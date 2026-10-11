from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[4]


def files_under(root: Path):
    ignored = {".git", ".venv", "build", "dist", "node_modules", "third_party"}
    for path in root.rglob("*"):
        if any(part in ignored for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in {
            ".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".inl"
        }:
            yield path


def search(root: Path, pattern: str, exact: bool = False, limit: int = 50):
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--glob", "*.{h,hh,hpp,hxx,c,cc,cpp,cxx,inl}"]
        if exact:
            cmd += ["-w", "-F", pattern]
        else:
            cmd += ["-i", pattern]
        cmd += [str(root)]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        lines = (result.stdout or result.stderr).splitlines()
        print("\n".join(lines[:limit]) if lines else "No matches.")
        return
    regex = re.compile(re.escape(pattern) if exact else pattern, re.IGNORECASE)
    count = 0
    for path in files_under(root):
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    print(f"{path}:{line_no}:{line.strip()}")
                    count += 1
                    if count >= limit:
                        return
        except OSError:
            continue
    if count == 0:
        print("No matches.")


def main():
    parser = argparse.ArgumentParser(description="Search C/C++ source files.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Root of the C++ repository")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Search for an exact symbol")
    group.add_argument("--grep", help="Search using a regular expression")
    group.add_argument("--find", help="Search for a text fragment")
    group.add_argument("--read", help="Read a file relative to --root")
    parser.add_argument("--lines", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Code root does not exist: {root}")
    if args.read:
        path = (root / args.read).resolve()
        if root not in path.parents and path != root:
            raise ValueError("Refusing to read a path outside --root.")
        if not path.is_file():
            raise FileNotFoundError(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start, end = args.lines or (1, len(lines))
        for i in range(max(1, start), min(end, len(lines)) + 1):
            print(f"{i}: {lines[i-1]}")
        return
    pattern = args.symbol or args.grep or args.find
    search(root, pattern, exact=bool(args.symbol), limit=args.limit)


if __name__ == "__main__":
    main()
