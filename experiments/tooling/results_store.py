#!/usr/bin/env python3
"""Run outputs live in one ``results/`` store outside git, symlinked back to where scripts look.

Each run-output location (``experiments/<dataset>/*_results*``, ``experiments/tooling/results``, the
benchmark timing cache) sits at the same relative path under ``results/`` and is replaced by a
relative symlink, so the sweep, analysis and figure scripts keep reading and writing the paths
they always have. ``results`` itself may be a symlink to a bigger disk.

    python experiments/tooling/results_store.py adopt      # move real run outputs into results/, link back
    python experiments/tooling/results_store.py link       # recreate the links from results/ (fresh clone)
    python experiments/tooling/results_store.py manifest   # write experiments/tooling/results_manifest.tsv
    python experiments/tooling/results_store.py verify     # check results/ against the manifest
    python experiments/tooling/results_store.py pack out.tar.gz --include '*.json'   # archive for publishing

Where the store is published and how to fetch it: experiments/tooling/RESULTS.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import os
import shutil
import sys
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

LOCATIONS: Final = (
    'experiments/*/*_results',
    'experiments/*/*_results_*',
    'experiments/tooling/results',
    'benchmarks/figures/paper_timings.json',
)
STORE: Final = Path('results')
MANIFEST: Final = Path('experiments/tooling/results_manifest.tsv')
MANIFEST_HEADER: Final = 'path\tbytes\tsha256\n'

type ManifestRow = tuple[str, int, str]


def locations(root: Path) -> list[Path]:
    """Run-output locations under *root* (real or symlinked), relative to it."""
    return sorted({p.relative_to(root) for pattern in LOCATIONS for p in root.glob(pattern)})


def _link(path: Path, target: Path) -> None:
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def adopt(repo: Path, store: Path) -> None:
    moved = 0
    for rel in locations(repo):
        src, dest = repo / rel, store / rel
        if src.is_symlink():
            continue
        if dest.exists():
            sys.exit(f'{dest} already exists: merge {src} into it by hand, then rerun adopt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        _link(src, dest)
        moved += 1
        print(f'{rel} -> {os.path.relpath(dest, repo)}')
    print(f'adopted {moved} location(s) into {store}')


def link(repo: Path, store: Path) -> None:
    created = 0
    for rel in locations(store):
        path, target = repo / rel, store / rel
        if path.is_symlink():
            if path.resolve() != target.resolve():
                sys.exit(f'{rel} links to {os.readlink(path)}, not into {store}')
            continue
        if path.exists():
            sys.exit(f'{rel} is a real path: run adopt to move it into {store}')
        path.parent.mkdir(parents=True, exist_ok=True)
        _link(path, target)
        created += 1
    print(f'linked {created} location(s) from {store}')


def _files(store: Path) -> Iterator[Path]:
    # Dot-entries are hosting state: the .cache/ that `hf download --local-dir` writes and the
    # dataset repo's .gitattributes.
    for dirpath, dirnames, filenames in os.walk(store):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for name in sorted(filenames):
            if not name.startswith('.'):
                yield Path(dirpath, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(store: Path, manifest: Path) -> None:
    rows: list[ManifestRow] = [
        (p.relative_to(store).as_posix(), p.stat().st_size, _sha256(p)) for p in _files(store)
    ]
    manifest.write_text(MANIFEST_HEADER + ''.join(f'{p}\t{n}\t{h}\n' for p, n, h in rows))
    total = sum(n for _, n, _ in rows)
    print(f'{len(rows)} files, {total / 2**30:.2f} GiB -> {manifest}')


def read_manifest(manifest: Path) -> dict[str, tuple[int, str]]:
    lines = manifest.read_text().splitlines()[1:]
    return {path: (int(size), sha) for path, size, sha in (line.split('\t') for line in lines)}


def _selected(path: str, include: Sequence[str]) -> bool:
    return not include or any(fnmatch.fnmatch(path, pattern) for pattern in include)


def verify(store: Path, manifest: Path, include: Sequence[str], hash_files: bool) -> None:
    expected = {p: v for p, v in read_manifest(manifest).items() if _selected(p, include)}
    actual = {
        rel: p for p in _files(store) if _selected(rel := p.relative_to(store).as_posix(), include)
    }
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = [
        p
        for p in sorted(expected.keys() & actual.keys())
        if actual[p].stat().st_size != expected[p][0]
        or (hash_files and _sha256(actual[p]) != expected[p][1])
    ]
    for label, paths in (('missing', missing), ('changed', changed), ('not in manifest', extra)):
        for p in paths:
            print(f'{label}: {p}')
    print(
        f'{len(expected)} expected, {len(missing)} missing, {len(changed)} changed, '
        f'{len(extra)} not in manifest'
    )
    if missing or changed:
        sys.exit(1)


def pack(store: Path, archive: Path, include: Sequence[str]) -> None:
    """Write the selected files to a ``.tar.gz`` of ``results/...`` paths, for the repo root.

    Owners, permissions and the gzip header are fixed, so an unchanged store packs to the same
    bytes and re-uploading it is a no-op.
    """
    files = [p for p in _files(store) if _selected(p.relative_to(store).as_posix(), include)]
    with (
        archive.open('wb') as raw,
        gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz,
    ):
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for path in files:
                info = tar.gettarinfo(path, arcname=(STORE / path.relative_to(store)).as_posix())
                info.mtime = int(info.mtime)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as f:
                    tar.addfile(info, f)
    print(f'{len(files)} files, {archive.stat().st_size / 2**20:.1f} MiB -> {archive}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Repository root; pass it when running a copy of this script from outside the repo.',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('adopt', help='Move real run-output paths into the store and link back.')
    commands.add_parser('link', help='Create the symlinks for everything in the store.')
    commands.add_parser('manifest', help='Write path, size and sha256 of every stored file.')
    check = commands.add_parser('verify', help='Compare the store with the manifest.')
    check.add_argument(
        '--include', nargs='*', default=[], help="Only check matching paths, e.g. '*.json'."
    )
    check.add_argument('--quick', action='store_true', help='Compare sizes only, no hashing.')
    packer = commands.add_parser('pack', help='Write stored files to a .tar.gz for publishing.')
    packer.add_argument('archive', type=Path, help='Output path, outside the store.')
    packer.add_argument(
        '--include', nargs='*', default=[], help="Only pack matching paths, e.g. '*.json'."
    )
    args = parser.parse_args()

    repo: Path = args.repo.resolve()
    store, manifest = repo / STORE, repo / MANIFEST
    match args.command:
        case 'adopt':
            adopt(repo, store)
        case 'link':
            link(repo, store)
        case 'manifest':
            write_manifest(store, manifest)
        case 'verify':
            verify(store, manifest, args.include, hash_files=not args.quick)
        case 'pack':
            pack(store, args.archive.resolve(), args.include)


if __name__ == '__main__':
    main()
