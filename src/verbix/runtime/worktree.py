from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


class WorktreeManager:
    """Manages isolated Git worktrees for concurrent mutating background tasks.

    Falls back to isolated temporary subdirectories if the workspace is not a Git repository.
    """

    def __init__(self, root_workspace: Path) -> None:
        self.root_workspace = root_workspace.resolve()
        self._worktrees_dir = self.root_workspace / ".verbix" / "worktrees"

    async def is_git_repo(self) -> bool:
        """Checks if the root workspace is inside a valid git repository."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--is-inside-work-tree",
            cwd=str(self.root_workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def create_worktree(self, task_id: str) -> Path:
        """Creates an isolated git worktree on branch worker/<task_id>.

        If not a git repository, creates a dedicated temporary directory.
        """
        if not await self.is_git_repo():
            fallback_dir = self._worktrees_dir / task_id
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir

        self._worktrees_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = self._worktrees_dir / task_id
        branch_name = f"worker/{task_id}"

        # Clean up stale worktree or branch if exists
        if worktree_path.exists():
            await self.remove_worktree(task_id, delete_branch=True)

        proc = await asyncio.create_subprocess_exec(
            "git",
            "worktree",
            "add",
            "-b",
            branch_name,
            str(worktree_path),
            "HEAD",
            cwd=str(self.root_workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, _stderr = await proc.communicate()
        if proc.returncode != 0:
            # Fall back to standalone directory if worktree add fails (e.g. empty repo without commits)
            worktree_path.mkdir(parents=True, exist_ok=True)
            return worktree_path

        return worktree_path

    async def remove_worktree(self, task_id: str, delete_branch: bool = False) -> None:
        """Removes the worktree and optionally deletes the worker branch."""
        worktree_path = self._worktrees_dir / task_id
        if not worktree_path.exists():
            return

        if await self.is_git_repo():
            proc = await asyncio.create_subprocess_exec(
                "git",
                "worktree",
                "remove",
                "--force",
                str(worktree_path),
                cwd=str(self.root_workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if delete_branch:
                branch_proc = await asyncio.create_subprocess_exec(
                    "git",
                    "branch",
                    "-D",
                    f"worker/{task_id}",
                    cwd=str(self.root_workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await branch_proc.communicate()

        # If directory remains, remove it
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
