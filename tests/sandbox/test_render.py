"""Provider-wire tests for Render streaming, resource ownership, and file transfer."""

from __future__ import annotations

import asyncio
import base64
import io
import shlex
import tarfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from render.experimental.sandbox import (
    SandboxExecExit,
    SandboxExecOutput,
    SandboxNotFoundError,
)

from agents.extensions.sandbox import RenderSandboxClient, RenderSandboxClientOptions
from agents.extensions.sandbox.render import RenderSandboxSessionState
from agents.sandbox import Manifest
from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    InvalidManifestPathError,
    WorkspaceArchiveWriteError,
    WorkspaceStartError,
)
from agents.sandbox.manifest import Environment
from agents.sandbox.session import BaseSandboxClientOptions


class Provider:
    """Model the provider wire, including malformed streams and controlled cancellation."""

    def __init__(self):
        self.create = AsyncMock(return_value=SimpleNamespace(id="sbx-test", status="creating"))
        self.from_id = AsyncMock(return_value=SimpleNamespace(id="sbx-test", status="running"))
        self.terminate = AsyncMock()
        self.copy_to = AsyncMock()
        self.copy_from = AsyncMock()
        self.events = [
            SandboxExecOutput(stream="stdout", data="hello\n"),
            SandboxExecExit(exit_code=0),
        ]
        self.commands: list[str] = []
        self.started = asyncio.Event()
        self.block = False
        self.stream_closed = False

    async def exec(self, sandbox_id, command, *, owner_id):
        self.commands.append(command)
        self.started.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            for event in self.events:
                yield event
        finally:
            self.stream_closed = True


@pytest.fixture
def setup(monkeypatch):
    import render

    provider = Provider()
    monkeypatch.setattr(
        render,
        "RenderAsync",
        lambda **kwargs: SimpleNamespace(experimental=SimpleNamespace(sandboxes=provider)),
    )
    client = RenderSandboxClient(token="synthetic-render-token", owner_id="tea-test")
    return client, provider


async def new_session(setup, **kwargs):
    client, provider = setup
    return await client.create(options=RenderSandboxClientOptions(), **kwargs)


@pytest.mark.asyncio
async def test_exec_stream_exit_and_quoting(setup):
    session = await new_session(
        setup,
        manifest=Manifest(
            root="/workspace/a b", environment=Environment(value={"GREETING": "a b"})
        ),
    )
    _, provider = setup
    provider.events = [
        SandboxExecOutput(stream="stdout", data="out\n"),
        SandboxExecOutput(stream="stderr", data="err\n"),
        SandboxExecExit(exit_code=7),
    ]
    result = await session.exec("printf", "%s", "$(not-executed)", shell=False)
    assert (result.stdout, result.stderr, result.exit_code) == (b"out\n", b"err\n", 7)
    assert "cd -- '/workspace/a b'" in provider.commands[-1]
    assert "'GREETING=a b'" in provider.commands[-1]
    assert "'$(not-executed)'" in provider.commands[-1]
    provider.terminate.assert_not_awaited()
    assert provider.stream_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events", [[], [object()], [SandboxExecExit(exit_code=0), SandboxExecExit(exit_code=1)]]
)
async def test_malformed_stream_terminates_sandbox(setup, events):
    session = await new_session(setup)
    _, provider = setup
    provider.events = events
    with pytest.raises(ExecTransportError):
        await session.exec("echo hello")
    provider.terminate.assert_awaited_once_with("sbx-test", owner_id="tea-test")


@pytest.mark.asyncio
async def test_timeout_terminates_remote_sandbox(setup):
    session = await new_session(setup)
    _, provider = setup
    provider.block = True
    with pytest.raises(ExecTimeoutError):
        await session.exec("sleep 60", timeout=0.01)
    provider.terminate.assert_awaited_once()
    assert not await session.running()


@pytest.mark.asyncio
async def test_cancel_closes_stream_and_terminates_sandbox(setup):
    session = await new_session(setup)
    _, provider = setup
    provider.block = True
    task = asyncio.create_task(session.exec("sleep 60"))
    await provider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.stream_closed
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_failure_is_retryable_and_not_marked_terminated(setup):
    session = await new_session(setup)
    client, provider = setup
    provider.terminate.side_effect = RuntimeError("secret response body")
    with pytest.raises(RuntimeError, match="termination failed") as exc:
        await client.delete(session)
    assert "secret" not in str(exc.value)
    provider.terminate.side_effect = None
    await client.delete(session)
    await client.delete(session)
    assert provider.terminate.await_count == 2


@pytest.mark.asyncio
async def test_startup_failure_cleans_up(setup):
    client, provider = setup
    provider.from_id.return_value.status = "failed"
    with pytest.raises(WorkspaceStartError):
        await new_session(setup)
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_reattaches_and_does_not_mutate_original_state(setup):
    session = await new_session(setup)
    client, provider = setup
    state = client.deserialize_session_state(client.serialize_session_state(session.state))
    resumed = await client.resume(state)
    assert resumed.state.sandbox_id == "sbx-test"
    assert resumed.state is not state
    assert provider.create.await_count == 1
    await client.delete(resumed)
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_only_recreates_confirmed_missing_resource(setup):
    session = await new_session(setup)
    client, provider = setup
    provider.from_id.side_effect = [SandboxNotFoundError("gone"), SimpleNamespace(status="running")]
    provider.create.return_value.id = "sbx-new"
    resumed = await client.resume(session.state)
    assert resumed.state.sandbox_id == "sbx-new"
    assert session.state.sandbox_id == "sbx-test"
    provider.from_id.side_effect = RuntimeError("auth or transport failure")
    with pytest.raises(WorkspaceStartError):
        await client.resume(resumed.state)
    assert provider.create.await_count == 2


@pytest.mark.asyncio
async def test_foreign_workspace_rejected_before_lookup(setup):
    session = await new_session(setup)
    client, provider = setup
    session.state.owner_id = "tea-other"
    provider.from_id.reset_mock()
    with pytest.raises(ValueError, match="another workspace"):
        await client.resume(session.state)
    with pytest.raises(ValueError, match="another workspace"):
        await client.delete(session)
    provider.from_id.assert_not_awaited()
    provider.terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_credentials_are_not_in_state_or_provider_create(setup):
    session = await new_session(setup)
    client, provider = setup
    payload = client.serialize_session_state(session.state)
    assert "synthetic-render-token" not in str(payload)
    assert "env" not in provider.create.call_args.kwargs
    assert isinstance(client.deserialize_session_state(payload), RenderSandboxSessionState)


@pytest.mark.asyncio
async def test_reserved_credentials_rejected_before_provisioning(setup):
    client, provider = setup
    with pytest.raises(ValueError, match="outside"):
        await new_session(
            setup, manifest=Manifest(environment=Environment(value={"RENDER_API_KEY": "synthetic"}))
        )
    provider.create.assert_not_awaited()


def test_options_roundtrip_and_reject_unsupported_settings():
    options = RenderSandboxClientOptions(timeout_seconds=300, network_policy="deny-all")
    assert BaseSandboxClientOptions.parse(options.model_dump()) == options
    for kwargs in [{"timeout_seconds": 0}, {"exposed_ports": [8080]}, {"image": "anything"}]:
        with pytest.raises(ValidationError):
            RenderSandboxClientOptions(**kwargs)


@pytest.mark.asyncio
async def test_binary_file_transfer_uses_scoped_paths_and_cleans_temporary_files(
    setup, monkeypatch
):
    session = await new_session(setup)
    _, provider = setup

    # Path validation is owned by BaseSandboxSession; isolate the provider transfer boundary.
    async def validate(path, **kwargs):
        return Path("/workspace") / path

    monkeypatch.setattr(session._inner, "_validate_path_access", validate)
    content = b"\x00\xffbinary\n"
    copied = []

    async def upload(sandbox_id, local, remote, **kwargs):
        assert Path(local).read_bytes() == content
        assert remote == "/workspace/file with spaces.bin"
        copied.append(Path(local))

    async def download(sandbox_id, remote, local, **kwargs):
        Path(local).write_bytes(content)
        copied.append(Path(local))

    provider.copy_to.side_effect = upload
    provider.copy_from.side_effect = download
    await session.write(Path("file with spaces.bin"), io.BytesIO(content))
    with await session.read(Path("file with spaces.bin")) as stream:
        assert stream.read() == content
    assert all(not path.exists() for path in copied)


@pytest.mark.asyncio
async def test_file_operations_keep_remote_paths_posix_on_windows(setup, monkeypatch):
    import agents.sandbox.session.base_sandbox_session as base_session

    session = await new_session(setup)
    _, provider = setup
    remote = "/workspace/dir with spaces/file.bin"
    content = b"\x00\xffbinary"

    # Simulate the native path result on Windows without bypassing shared path validation.
    monkeypatch.setattr(base_session, "posix_path_as_path", PureWindowsPath)

    async def exec_events(sandbox_id, command, *, owner_id):
        provider.commands.append(command)
        # The provider wire supplies a resolved path or the requested file prefix.
        output = base64.b64encode(content[:2]).decode() if "head -c 2" in command else remote
        yield SandboxExecOutput(stream="stdout", data=output)
        yield SandboxExecExit(exit_code=0)

    async def upload(sandbox_id, local, destination, **kwargs):
        assert destination == remote
        assert Path(local).read_bytes() == content

    async def download(sandbox_id, source, local, **kwargs):
        assert source == remote
        Path(local).write_bytes(content)

    provider.exec = exec_events
    provider.copy_to.side_effect = upload
    provider.copy_from.side_effect = download
    path = PureWindowsPath("dir with spaces/file.bin")
    await session.write(path, io.BytesIO(content))
    with await session.read(path) as stream:
        assert stream.read() == content
    assert await session.read_bounded(path, max_bytes=2) == content[:2]
    bounded_command = shlex.split(provider.commands[-1])[-1]
    assert shlex.split(bounded_command) == ["head", "-c", "2", "--", remote, "|", "base64"]


@pytest.mark.asyncio
async def test_bounded_read_requests_only_prefix(setup, monkeypatch):
    session = await new_session(setup)
    _, provider = setup
    monkeypatch.setattr(
        session._inner, "_check_read_with_exec", AsyncMock(return_value=Path("/workspace/file"))
    )
    provider.events = [
        SandboxExecOutput(stream="stdout", data=base64.b64encode(b"\x00\xff").decode()),
        SandboxExecExit(exit_code=0),
    ]
    assert await session.read_bounded(Path("file"), max_bytes=2) == b"\x00\xff"
    assert "head -c 2" in provider.commands[-1]
    provider.copy_from.assert_not_awaited()


@pytest.mark.asyncio
async def test_lexical_path_escape_rejected(setup):
    session = await new_session(setup)
    with pytest.raises(InvalidManifestPathError):
        session._inner.normalize_path(Path("../escape"), for_write=True)


@pytest.mark.asyncio
async def test_archive_traversal_rejected_before_writes(setup):
    session = await new_session(setup)
    _, provider = setup
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        member = tarfile.TarInfo("../../escape")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    archive.seek(0)
    with pytest.raises(WorkspaceArchiveWriteError):
        await session._inner.hydrate_workspace(archive)
    provider.copy_to.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_cleans_up_when_persistence_fails(setup, monkeypatch):
    session = await new_session(setup)
    _, provider = setup
    monkeypatch.setattr(
        session._inner, "stop", AsyncMock(side_effect=RuntimeError("snapshot failed"))
    )
    with pytest.raises(RuntimeError, match="snapshot failed"):
        await session.aclose()
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_cancellation_cleans_up(setup, monkeypatch):
    session = await new_session(setup)
    _, provider = setup
    monkeypatch.setattr(
        session._inner, "_prepare_backend_workspace", AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await session.start()
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_override_rejected_without_command(setup):
    session = await new_session(setup)
    _, provider = setup
    with pytest.raises(NotImplementedError, match="user overrides"):
        await session.exec("id", user="root")
    assert provider.commands == []


@pytest.mark.asyncio
async def test_startup_exception_does_not_retain_provider_payload(setup):
    _, provider = setup
    provider.from_id.side_effect = RuntimeError("synthetic-sensitive-response")
    with pytest.raises(WorkspaceStartError) as error:
        await new_session(setup)
    assert error.value.__context__ is None
    provider.terminate.assert_awaited_once()


def test_optional_dependency_missing_has_actionable_error(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_render(name, *args, **kwargs):
        if name == "render":
            raise ImportError("render not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_render)
    # The public options remain usable without constructing the provider client.
    assert RenderSandboxClientOptions().type == "render"
    with pytest.raises(ImportError, match=r"openai-agents\[render\]"):
        RenderSandboxClient(owner_id="tea-test")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_environment_resolution_is_inside_execution_deadline(setup, monkeypatch, cancel):
    session = await new_session(setup)
    _, provider = setup
    entered = asyncio.Event()

    async def resolve(self):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(Environment, "resolve", resolve)
    task = asyncio.create_task(session.exec("true", timeout=None if cancel else 0.01))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ExecTimeoutError):
        await asyncio.wait_for(task, timeout=1)
    provider.terminate.assert_awaited_once()
    assert provider.commands == []


@pytest.mark.asyncio
async def test_workspace_start_errors_are_sanitized_in_instrumentation(setup, monkeypatch):
    from agents.sandbox.session import CallbackSink, Instrumentation

    client, provider = setup
    events = []
    client._instrumentation = Instrumentation(
        sinks=[CallbackSink(lambda event, _session: events.append(event), mode="sync")]
    )
    session = await new_session(setup)

    async def broken_stream(*args, **kwargs):
        raise RuntimeError("synthetic-sensitive-response")
        yield

    monkeypatch.setattr(provider, "exec", broken_stream)
    with pytest.raises(WorkspaceStartError) as error:
        await session.start()
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert "synthetic-sensitive-response" not in str(error.value)
    assert events
    assert "synthetic-sensitive-response" not in repr(events)
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["running", "persist_workspace"])
async def test_status_and_persistence_errors_discard_provider_context(setup, operation):
    from agents.sandbox.errors import WorkspaceArchiveReadError

    session = await new_session(setup)
    _, provider = setup
    provider.from_id.side_effect = RuntimeError("synthetic-sensitive-response")
    provider.copy_from.side_effect = RuntimeError("synthetic-sensitive-response")
    expected = RuntimeError if operation == "running" else WorkspaceArchiveReadError
    with pytest.raises(expected) as error:
        await getattr(session._inner, operation)()
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert "synthetic-sensitive-response" not in str(error.value)


@pytest.mark.asyncio
async def test_readiness_and_cleanup_failure_discard_both_provider_payloads(setup):
    _, provider = setup
    provider.from_id.side_effect = RuntimeError("synthetic-sensitive-readiness")
    provider.terminate.side_effect = RuntimeError("synthetic-sensitive-cleanup")
    with pytest.raises(RuntimeError, match="termination failed") as error:
        await new_session(setup)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert "synthetic-sensitive" not in str(error.value)
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["exec", "readiness"])
async def test_cancellation_during_failure_cleanup_finishes_termination(setup, phase):
    _, provider = setup
    session = await new_session(setup)
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def terminate(*args, **kwargs):
        entered.set()
        await release.wait()
        finished.set()

    provider.terminate.side_effect = terminate
    if phase == "exec":
        provider.events = []  # A stream without an exit event is a transport failure.
        operation = session.exec("true")
    else:
        provider.from_id.side_effect = RuntimeError("synthetic-sensitive-response")
        operation = new_session(setup)
    task = asyncio.create_task(operation)
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("create_fails", [False, True])
async def test_client_context_closes_http_pool_after_sessions_or_failed_create(create_fails):
    import httpx

    client = RenderSandboxClient(token="synthetic", owner_id="tea-test")
    http = httpx.AsyncClient(
        base_url="https://api.render.com/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(204)),
    )
    client._provider.client.set_async_httpx_client(http)
    client._provider.create = AsyncMock(
        side_effect=RuntimeError("synthetic-provider-response") if create_fails else None,
        return_value=SimpleNamespace(id="sbx-test"),
    )
    client._provider.from_id = AsyncMock(return_value=SimpleNamespace(status="running"))
    try:
        async with client:
            if create_fails:
                with pytest.raises(WorkspaceStartError):
                    await client.create(options=RenderSandboxClientOptions())
            else:
                first = await client.create(options=RenderSandboxClientOptions())
                second = await client.create(options=RenderSandboxClientOptions())
                await client.delete(first)
                assert not http.is_closed  # Sibling sessions still need the shared transport.
                await client.delete(second)
        assert http.is_closed
        await client.close()  # HTTP cleanup is safe to repeat.
    finally:
        await http.aclose()
