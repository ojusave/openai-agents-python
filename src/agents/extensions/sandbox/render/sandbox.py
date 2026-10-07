"""Render Sandboxes backend using the optional Render Python SDK."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field

from ....sandbox._mount_security import redact_mount_error_data
from ....sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    MountConfigError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceWriteTypeError,
)
from ....sandbox.manifest import Manifest
from ....sandbox.session import SandboxSession, SandboxSessionState
from ....sandbox.session.base_sandbox_session import BaseSandboxSession
from ....sandbox.session.dependencies import Dependencies
from ....sandbox.session.manager import Instrumentation
from ....sandbox.session.runtime_helpers import RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript
from ....sandbox.session.sandbox_client import BaseSandboxClient, BaseSandboxClientOptions
from ....sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from ....sandbox.types import ExecResult, User
from ....sandbox.workspace_paths import sandbox_path_str


class RenderSandboxClientOptions(BaseSandboxClientOptions):
    """Provider settings. Command deadlines are separate from sandbox lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["render"] = "render"
    timeout_seconds: int = Field(default=900, gt=0)
    startup_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    network_policy: Literal["allow-all", "deny-all"] = "allow-all"


class RenderSandboxSessionState(SandboxSessionState):
    """Serializable identity and non-secret settings for a Render workspace."""

    type: Literal["render"] = "render"
    sandbox_id: str = ""
    owner_id: str
    timeout_seconds: int = Field(default=900, gt=0)
    startup_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    network_policy: Literal["allow-all", "deny-all"] = "allow-all"


def _validate_manifest(manifest: Manifest) -> None:
    if any(True for _ in manifest.mount_targets()):
        raise MountConfigError(message="Render does not support storage mounts")
    if manifest.users or manifest.groups:
        raise ValueError("Render does not support manifest users or groups")
    # Reserved credentials must remain in the application process.
    for name in ("OPENAI_API_KEY", "RENDER_API_KEY"):
        if name in manifest.environment.value:
            raise ValueError(f"Keep {name} outside the Render sandbox manifest")


class RenderSandboxSession(BaseSandboxSession):
    """Non-PTY session. A command timeout/cancellation terminates the whole sandbox."""

    state: RenderSandboxSessionState

    def __init__(self, state: RenderSandboxSessionState, provider: Any) -> None:
        self.state = state
        self._provider = provider
        self._terminated = False

    @property
    def sandbox_id(self) -> str:
        return self.state.sandbox_id

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    async def _wait_until_running(self) -> None:
        async def wait() -> None:
            while True:
                sandbox = await self._provider.from_id(
                    self.sandbox_id, owner_id=self.state.owner_id
                )
                if sandbox.status == "running":
                    return
                if sandbox.status not in {"creating", "starting"}:
                    raise WorkspaceStartError(path=Path(self.state.manifest.root))
                await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(wait(), timeout=self.state.startup_timeout_seconds)
        except asyncio.CancelledError:
            await asyncio.shield(self._shutdown_backend())
            raise
        except Exception:
            pass
        else:
            return
        await self._shutdown_backend()
        raise WorkspaceStartError(path=Path(self.state.manifest.root))

    async def start(self) -> None:
        try:
            await super().start()
        except asyncio.CancelledError:
            await asyncio.shield(self._shutdown_backend())
            raise

    def _prepare_exec_command(
        self, *command: str | Path, shell: bool | list[str], user: str | User | None
    ) -> list[str]:
        self._check_user(user)
        return super()._prepare_exec_command(*command, shell=shell, user=user)

    async def _prepare_backend_workspace(self) -> None:
        try:
            result = await self._collect("mkdir -p -- " + shlex.quote(self.state.manifest.root))
            if result.ok():
                return
        except Exception:
            pass
        raise WorkspaceStartError(path=Path(self.state.manifest.root))

    async def _after_start_failed(self) -> None:
        await self._shutdown_backend()

    async def _aclose_impl(self) -> None:
        try:
            await super()._aclose_impl()
        finally:
            # A failed snapshot must not skip provider resource cleanup.
            await self._shutdown_backend()

    async def _shutdown_backend(self) -> None:
        if self._terminated:
            return
        from render.experimental.sandbox import SandboxNotFoundError

        async def terminate() -> None:
            failed = False
            try:
                await self._provider.terminate(self.sandbox_id, owner_id=self.state.owner_id)
            except SandboxNotFoundError:
                pass
            except Exception:
                failed = True
            if failed:
                raise RuntimeError("Render sandbox termination failed; retry cleanup with its ID")
            self._terminated = True

        # Retain and finish the termination request even if its caller is cancelled.
        cleanup = asyncio.create_task(terminate())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
            raise

    async def running(self) -> bool:
        if self._terminated:
            return False
        from render.experimental.sandbox import SandboxNotFoundError

        try:
            sandbox = await self._provider.from_id(self.sandbox_id, owner_id=self.state.owner_id)
        except SandboxNotFoundError:
            return False
        except Exception:
            pass
        else:
            return bool(sandbox.status == "running")
        raise RuntimeError("Render sandbox status lookup failed")

    async def _collect(self, command: str) -> ExecResult:
        from render.experimental.sandbox import SandboxExecExit, SandboxExecOutput

        stdout = bytearray()
        stderr = bytearray()
        exit_code: int | None = None
        stream = self._provider.exec(self.sandbox_id, command, owner_id=self.state.owner_id)
        try:
            async for event in stream:
                if exit_code is not None:
                    raise ValueError("Unexpected event after exit")
                if isinstance(event, SandboxExecOutput):
                    if event.stream == "stdout":
                        stdout.extend(event.data.encode())
                    elif event.stream == "stderr":
                        stderr.extend(event.data.encode())
                    else:
                        raise ValueError("Unknown output stream")
                elif isinstance(event, SandboxExecExit):
                    exit_code = event.exit_code
                else:
                    raise ValueError("Unknown execution event")
        finally:
            await stream.aclose()
        if exit_code is None:
            raise ValueError("Missing execution exit event")
        return ExecResult(stdout=bytes(stdout), stderr=bytes(stderr), exit_code=exit_code)

    @redact_mount_error_data
    async def _exec_internal(
        self, *command: str | Path, timeout: float | None = None
    ) -> ExecResult:
        if self._terminated:
            raise ExecTransportError(command=command, message="Render sandbox was terminated")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a positive finite number")

        async def execute() -> ExecResult:
            env = await self.state.manifest.environment.resolve()
            assignments = [f"{key}={value}" for key, value in env.items()]
            remote_command = f"cd -- {shlex.quote(self.state.manifest.root)} && " + shlex.join(
                ["env", *assignments, *map(str, command)]
            )
            return await self._collect(remote_command)

        failure = False
        timed_out = False
        try:
            return await asyncio.wait_for(execute(), timeout=timeout)
        except asyncio.CancelledError:
            await asyncio.shield(self._shutdown_backend())
            raise
        except asyncio.TimeoutError:
            timed_out = True
        except Exception:
            failure = True
        # A broken stream may leave the remote command alive. Fail closed.
        if failure or timed_out:
            await self._shutdown_backend()
        if timed_out:
            raise ExecTimeoutError(command=command, timeout_s=timeout)
        raise ExecTransportError(command=command, message="Render command stream failed")

    @staticmethod
    def _check_user(user: str | User | None) -> None:
        if user is not None:
            raise NotImplementedError("Render file operations do not support user overrides")

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        self._check_user(user)
        path = await self._check_read_with_exec(path, user=None)
        from render.experimental.sandbox import SandboxFileNotFoundError

        missing = False
        try:
            with tempfile.TemporaryDirectory(prefix="agents-render-read-") as directory:
                local = Path(directory) / "payload"
                await self._provider.copy_from(
                    self.sandbox_id,
                    sandbox_path_str(path),
                    str(local),
                    owner_id=self.state.owner_id,
                )
                return io.BytesIO(local.read_bytes())
        except SandboxFileNotFoundError:
            missing = True
        except Exception:
            pass
        if missing:
            raise WorkspaceReadNotFoundError(path=path)
        raise WorkspaceArchiveReadError(path=path)

    async def _read_bounded(self, path: Path, *, max_bytes: int) -> bytes:
        path = await self._check_read_with_exec(path, user=None)
        result = await self.exec(
            "bash",
            "-o",
            "pipefail",
            "-c",
            f"head -c {max_bytes} -- {shlex.quote(sandbox_path_str(path))} | base64",
            shell=False,
        )
        if not result.ok():
            raise WorkspaceArchiveReadError(path=path)
        encoded = b"".join(result.stdout.split())
        if len(encoded) > 4 * ((max_bytes + 2) // 3):
            raise WorkspaceArchiveReadError(
                path=path, context={"reason": "bounded_read_wire_limit"}
            )
        return base64.b64decode(encoded, validate=True)[:max_bytes]

    async def write(self, path: Path, data: io.IOBase, *, user: str | User | None = None) -> None:
        self._check_user(user)
        path = await self._validate_path_access(path, for_write=True)
        content = data.read()
        if isinstance(content, str):
            content = content.encode()
        if not isinstance(content, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=path, actual_type=type(content).__name__)
        failed = False
        try:
            with tempfile.TemporaryDirectory(prefix="agents-render-write-") as directory:
                local = Path(directory) / "payload"
                local.write_bytes(content)
                await self._provider.copy_to(
                    self.sandbox_id,
                    str(local),
                    sandbox_path_str(path),
                    owner_id=self.state.owner_id,
                )
        except Exception:
            failed = True
        if failed:
            raise WorkspaceArchiveWriteError(path=path)

    async def persist_workspace(self) -> io.IOBase:
        # Keep the archive outside the workspace to avoid archiving itself.
        archive = f"/tmp/agents-render-{uuid.uuid4().hex}.tar"
        excludes = [
            f"--exclude=./{path.as_posix()}"
            for path in sorted(self._persist_workspace_skip_relpaths())
        ]
        try:
            result = await self.exec(
                "tar",
                "cf",
                archive,
                "--hard-dereference",
                "--no-wildcards",
                *excludes,
                ".",
                shell=False,
            )
            if not result.ok():
                raise WorkspaceArchiveReadError(path=Path(self.state.manifest.root))
            with tempfile.TemporaryDirectory(prefix="agents-render-snapshot-") as directory:
                local = Path(directory) / "workspace.tar"
                await self._provider.copy_from(
                    self.sandbox_id, archive, str(local), owner_id=self.state.owner_id
                )
                return io.BytesIO(local.read_bytes())
        except Exception:
            pass
        finally:
            if not self._terminated:
                await self.exec("rm", "-f", "--", archive, shell=False)
        raise WorkspaceArchiveReadError(path=Path(self.state.manifest.root))

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        await self._extract_tar_archive(
            archive_path=Path("render-workspace.tar"),
            destination_root=Path(self.state.manifest.root),
            data=data,
            archive_limits=self._archive_limits,
        )


class _RenderSessionWrapper(SandboxSession):
    async def _aclose_impl(self) -> None:
        try:
            await super()._aclose_impl()
        finally:
            await self._inner.shutdown()


class RenderSandboxClient(BaseSandboxClient[RenderSandboxClientOptions]):
    """Manage Render Sandboxes using application-side provider credentials."""

    backend_id = "render"

    def __init__(
        self,
        *,
        token: str | None = None,
        owner_id: str | None = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        super().__init__()
        try:
            from render import RenderAsync
        except ImportError:
            raise ImportError("Install openai-agents[render] to use RenderSandboxClient") from None
        self._owner_id = owner_id or os.environ.get("RENDER_WORKSPACE_ID")
        if not self._owner_id:
            raise ValueError("Set RENDER_WORKSPACE_ID or pass owner_id")
        self._provider = RenderAsync(token=token, owner_id=self._owner_id).experimental.sandboxes
        self._instrumentation = instrumentation
        self._dependencies = dependencies

    async def close(self) -> None:
        """Close the provider HTTP pool after all sessions have been cleaned up."""
        await self._provider.client.get_async_httpx_client().aclose()

    async def __aenter__(self) -> RenderSandboxClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _wrap_session(
        self, inner: BaseSandboxSession, *, instrumentation: Instrumentation | None = None
    ) -> SandboxSession:
        return _RenderSessionWrapper(
            inner, instrumentation=instrumentation, dependencies=self._resolve_dependencies()
        )

    async def _provision(self, state: RenderSandboxSessionState) -> RenderSandboxSession:
        failed = False
        try:
            sandbox = await self._provider.create(
                owner_id=self._owner_id,
                timeout_seconds=state.timeout_seconds,
                network_policy=state.network_policy,
            )
        except Exception:
            failed = True
        if failed:
            raise WorkspaceStartError(path=Path(state.manifest.root))
        state.sandbox_id = sandbox.id
        state.workspace_root_ready = False
        inner = RenderSandboxSession(state, self._provider)
        await inner._wait_until_running()
        return inner

    @redact_mount_error_data
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: RenderSandboxClientOptions,
    ) -> SandboxSession:
        options = RenderSandboxClientOptions.model_validate(options)
        manifest = manifest or Manifest()
        _validate_manifest(manifest)
        self._validate_manifest_for_create(manifest)
        session_id = uuid.uuid4()
        assert self._owner_id is not None
        state = RenderSandboxSessionState(
            session_id=session_id,
            owner_id=self._owner_id,
            manifest=manifest,
            snapshot=resolve_snapshot(snapshot, str(session_id)),
            timeout_seconds=options.timeout_seconds,
            startup_timeout_seconds=options.startup_timeout_seconds,
            network_policy=options.network_policy,
        )
        inner = await self._provision(state)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, RenderSandboxSession):
            raise TypeError("RenderSandboxClient.delete expects a RenderSandboxSession")
        if inner.state.owner_id != self._owner_id:
            raise ValueError("Render session belongs to another workspace")
        await inner.shutdown()
        return session

    @redact_mount_error_data
    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        from render.experimental.sandbox import SandboxNotFoundError

        if not isinstance(state, RenderSandboxSessionState):
            raise TypeError("RenderSandboxClient.resume expects RenderSandboxSessionState")
        if state.owner_id != self._owner_id:
            raise ValueError("Render session belongs to another workspace")
        state.assert_path_grants_rebound()
        _validate_manifest(state.manifest)
        self._validate_manifest_for_create(state.manifest)
        if state.exposed_ports:
            raise ValueError("Render does not support exposed ports")
        state = state.model_copy(deep=True)
        missing = False
        lookup_failed = False
        try:
            sandbox = await self._provider.from_id(state.sandbox_id, owner_id=self._owner_id)
            missing = sandbox.status == "terminated"
        except SandboxNotFoundError:
            missing = True
        except Exception:
            lookup_failed = True
        if lookup_failed:
            raise WorkspaceStartError(path=Path(state.manifest.root))
        if missing:
            inner = await self._provision(state)
        else:
            inner = RenderSandboxSession(state, self._provider)
            await inner._wait_until_running()
            inner._set_start_state_preserved(True, system=True)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, RenderSandboxSessionState)
