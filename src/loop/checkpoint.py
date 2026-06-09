"""Deterministic git checkpoint for the autonomous tuning loop.

Stages only the experiments/ directory and creates a git commit so the LLM
can checkpoint with one deterministic command instead of hand-running git.

Safety guarantees:
- Only stages `experiments/` — never `git add -A` or `git add .`
- Never pushes to the remote.
- A "nothing to commit" exit code (1) from git commit is treated as success
  (idempotent — re-running after a clean tree is harmless).

Usage (from repo root with PYTHONPATH=src):
    uv run python -m loop.checkpoint --message "tick: adjudication done"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def run(message: str, cwd: str | None = None) -> None:
    """Stage experiments/ and commit with `message`.

    Args:
        message: Git commit message.
        cwd:     Working directory for git commands (defaults to repo root).
    """
    if cwd is None:
        # Default: repo root is two levels up from this file (src/loop/checkpoint.py).
        cwd = str(Path(__file__).resolve().parent.parent.parent)

    # Stage only the experiments/ directory.
    subprocess.run(["git", "add", "experiments/"], cwd=cwd, check=True)

    # Commit.  Exit code 1 means "nothing to commit" — treat as success.
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        # Unexpected failure — surface the error.
        sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, ["git", "commit"])

    if result.returncode == 1:
        print("checkpoint: nothing to commit (clean tree)")
    else:
        print(f"checkpoint: committed — {message}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage experiments/ and commit (never pushes)"
    )
    parser.add_argument(
        "--message", "-m",
        required=True,
        help="Git commit message",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory (default: repo root derived from this file's location)",
    )
    args = parser.parse_args(argv)
    run(message=args.message, cwd=args.cwd)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
