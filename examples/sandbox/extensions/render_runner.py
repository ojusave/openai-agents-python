"""Run an agent in Render and independently verify its output file before cleanup.

Run from the repository root:
    uv run --extra render python -m examples.sandbox.extensions.render_runner
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agents import ModelSettings, Runner
from agents.extensions.sandbox import RenderSandboxClient, RenderSandboxClientOptions
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.entries import File
from examples.sandbox.misc.workspace_shell import WorkspaceShellCapability


async def main() -> None:
    for name in ("OPENAI_API_KEY", "RENDER_API_KEY", "RENDER_WORKSPACE_ID"):
        if not os.environ.get(name):
            raise SystemExit(f"Set {name} before running this example.")
    manifest = Manifest(
        root="/workspace",
        entries={"README.md": File(content=b"Write generated files under outputs/.\n")},
    )
    agent = SandboxAgent(
        name="Render Sandbox Assistant",
        model="gpt-5.6-sol",
        instructions="Use the shell tool to complete the task and verify the result.",
        default_manifest=manifest,
        capabilities=[WorkspaceShellCapability()],
        model_settings=ModelSettings(tool_choice="required"),
    )
    async with RenderSandboxClient() as client:
        session = await client.create(
            manifest=manifest, options=RenderSandboxClientOptions(timeout_seconds=900)
        )
        try:
            async with session:
                result = await Runner.run(
                    agent,
                    "Read README.md. Create outputs/hello.txt containing exactly "
                    "RENDER_AGENTS_SDK_OK followed by a newline. Read it back.",
                    run_config=RunConfig(sandbox=SandboxRunConfig(session=session)),
                )
                print(result.final_output)
                with await session.read(Path("outputs/hello.txt")) as output:
                    content = output.read()
                if content != b"RENDER_AGENTS_SDK_OK\n":
                    raise RuntimeError("The sandbox file did not match the expected output.")
                print("Verified: outputs/hello.txt contains RENDER_AGENTS_SDK_OK")
        finally:
            await client.delete(session)


if __name__ == "__main__":
    asyncio.run(main())
