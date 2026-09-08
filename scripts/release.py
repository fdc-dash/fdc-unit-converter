#!/usr/bin/env python3
"""
Release automation for fdc-unit-converter.

Bumps the version, runs the quality gates, builds the distribution and
publishes it to PyPI, then tags the release in git.

Usage:
    uv run scripts/release.py patch          # 0.2.0 -> 0.2.1
    uv run scripts/release.py minor          # 0.2.0 -> 0.3.0
    uv run scripts/release.py major          # 0.2.0 -> 1.0.0
    uv run scripts/release.py 1.2.3          # explicit version

Options:
    --dry-run       Do everything except publishing, tagging and pushing.
    --test-pypi     Publish to TestPyPI instead of PyPI.
    --skip-tests    Skip pytest and pre-commit (not recommended).
    --no-push       Commit and tag locally, but do not push.
    --yes           Do not ask for confirmation before publishing.

Authentication:
    Set UV_PUBLISH_TOKEN to a PyPI API token (starts with "pypi-"), or pass
    --token. For TestPyPI use a TestPyPI token.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DIST = ROOT / "dist"

TEST_PYPI_URL = "https://test.pypi.org/legacy/"
VERSION_RE = re.compile(r"^(?P<prefix>version\s*=\s*[\"'])(?P<version>[^\"']+)(?P<suffix>[\"'])", re.M)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([-.+][0-9A-Za-z.-]+)?$")


class ReleaseError(RuntimeError):
    pass


def log(message: str) -> None:
    print("\n==> " + message, flush=True)


def run(cmd: List[str], capture: bool = False) -> str:
    printable = " ".join(cmd)
    print("    $ " + printable, flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    if result.returncode != 0:
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise ReleaseError("command failed ({0}): {1}".format(result.returncode, printable))
    return (result.stdout or "").strip()


def read_version() -> str:
    match = VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise ReleaseError("could not find a version field in pyproject.toml")
    return match.group("version")


def write_version(new_version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    updated, count = VERSION_RE.subn(
        lambda m: m.group("prefix") + new_version + m.group("suffix"), text, count=1
    )
    if count != 1:
        raise ReleaseError("could not rewrite the version field in pyproject.toml")
    PYPROJECT.write_text(updated, encoding="utf-8")


def bump(current: str, part: str) -> str:
    if SEMVER_RE.match(part):
        return part
    if part not in ("major", "minor", "patch"):
        raise ReleaseError("invalid version argument: {0!r}".format(part))
    core = re.split(r"[-+]", current, maxsplit=1)[0]
    try:
        major, minor, patch = (int(n) for n in core.split("."))
    except ValueError as exc:
        raise ReleaseError(
            "current version {0!r} is not semver; pass an explicit version".format(current)
        ) from exc
    if part == "major":
        return "{0}.0.0".format(major + 1)
    if part == "minor":
        return "{0}.{1}.0".format(major, minor + 1)
    return "{0}.{1}.{2}".format(major, minor, patch + 1)


def check_git_state(allow_dirty: bool) -> None:
    log("Checking the git working tree")
    if run(["git", "rev-parse", "--is-inside-work-tree"], capture=True) != "true":
        raise ReleaseError("not inside a git repository")
    dirty = run(["git", "status", "--porcelain"], capture=True)
    if dirty and not allow_dirty:
        raise ReleaseError(
            "the working tree has uncommitted changes; commit or stash them first:\n" + dirty
        )
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True)
    print("    branch: " + branch)


def check_tag_free(tag: str) -> None:
    if run(["git", "tag", "--list", tag], capture=True):
        raise ReleaseError("tag {0} already exists locally".format(tag))
    if run(["git", "ls-remote", "--tags", "origin", tag], capture=True):
        raise ReleaseError("tag {0} already exists on origin".format(tag))


def quality_gates(skip: bool) -> None:
    if skip:
        log("Skipping tests and pre-commit (--skip-tests)")
        return
    log("Running the test suite")
    run(["uv", "run", "pytest", "-q"])
    log("Running pre-commit on all files")
    run(["uv", "run", "pre-commit", "run", "--all-files"])


def build() -> List[Path]:
    log("Building the distribution")
    for pattern in ("*.whl", "*.tar.gz"):
        for stale in DIST.glob(pattern):
            stale.unlink()
    run(["uv", "build"])
    artifacts = sorted(DIST.glob("*.whl")) + sorted(DIST.glob("*.tar.gz"))
    if not artifacts:
        raise ReleaseError("uv build produced no artifacts")
    for artifact in artifacts:
        print("    " + artifact.name)
    return artifacts


def smoke_test(version: str) -> None:
    log("Smoke-testing the built wheel in a clean environment")
    wheels = list(DIST.glob("*" + version.replace("-", "_") + "*.whl"))
    if not wheels:
        raise ReleaseError("no wheel found in dist/ for version {0}".format(version))
    check = (
        "import fdc_unit_converter as m; "
        "from fdc_unit_converter import UnitConverter, units as u; "
        "assert abs(UnitConverter.convert(1000, u.meter, u.kilometer) - 1.0) < 1e-12; "
        "print('    import OK ->', m.__file__)"
    )
    run(
        [
            "uv", "run", "--no-project", "--isolated",
            "--with", str(wheels[0]),
            "python", "-c", check,
        ]
    )


def publish(token: Optional[str], test_pypi: bool) -> None:
    log("Publishing to " + ("TestPyPI" if test_pypi else "PyPI"))
    cmd = ["uv", "publish"]
    if test_pypi:
        cmd += ["--publish-url", TEST_PYPI_URL]
    if token:
        cmd += ["--token", token]
    elif not os.environ.get("UV_PUBLISH_TOKEN"):
        raise ReleaseError("no PyPI credentials: set UV_PUBLISH_TOKEN or pass --token")
    run(cmd)


def commit_and_tag(version: str, tag: str, push: bool) -> None:
    log("Committing and tagging " + tag)
    run(["git", "add", "pyproject.toml", "uv.lock"])
    if run(["git", "diff", "--cached", "--name-only"], capture=True):
        run(["git", "commit", "-m", "Release " + version])
    else:
        # The version bump was already committed, e.g. when re-publishing a
        # version that is only missing from PyPI. Tag what is already there.
        print("    nothing to commit; tagging the current HEAD")
    run(["git", "tag", "-a", tag, "-m", "Release " + version])
    if push:
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True)
        run(["git", "push", "origin", branch])
        run(["git", "push", "origin", tag])
    else:
        print("    --no-push: remember to run `git push && git push --tags`")


def confirm(question: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise ReleaseError("not a TTY; pass --yes to confirm non-interactively")
    if input("\n" + question + " [y/N] ").strip().lower() not in ("y", "yes"):
        raise ReleaseError("aborted by the user")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, publish and tag a new release of fdc-unit-converter.",
    )
    parser.add_argument("version", help="major | minor | patch | an explicit version such as 1.2.3")
    parser.add_argument("--dry-run", action="store_true", help="build and verify, but do not publish, tag or push")
    parser.add_argument("--test-pypi", action="store_true", help="publish to TestPyPI")
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest and pre-commit")
    parser.add_argument("--no-push", action="store_true", help="do not push the commit and tag")
    parser.add_argument("--allow-dirty", action="store_true", help="allow an unclean working tree")
    parser.add_argument("--token", help="PyPI API token (defaults to $UV_PUBLISH_TOKEN)")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    current = read_version()
    new_version = bump(current, args.version)
    tag = "v" + new_version
    print("fdc-unit-converter: {0} -> {1}  (tag {2})".format(current, new_version, tag))

    check_git_state(args.allow_dirty)
    if not args.dry_run:
        check_tag_free(tag)
    quality_gates(args.skip_tests)

    write_version(new_version)
    print("    pyproject.toml version set to " + new_version)
    try:
        run(["uv", "lock"])
        build()
        smoke_test(new_version)

        if args.dry_run:
            log("Dry run complete - nothing was published, tagged or pushed")
            write_version(current)
            run(["uv", "lock"])
            print("    reverted pyproject.toml to " + current)
            return 0

        target = "TestPyPI" if args.test_pypi else "PyPI"
        confirm("Publish fdc-unit-converter {0} to {1}?".format(new_version, target), args.yes)
        publish(args.token, args.test_pypi)
        commit_and_tag(new_version, tag, push=not args.no_push)
    except Exception:
        if read_version() == new_version:
            write_version(current)
            print("\n    rolled back pyproject.toml to " + current, file=sys.stderr)
        raise

    log("Released fdc-unit-converter " + new_version)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReleaseError as exc:
        print("\nerror: {0}".format(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
