"""Annotate, strip and report the LLM-style audit overlay of the paper sources.

python paper/tools/llm_audit.py annotate      # wrap every catalogue item in \\llm / \\llmpt
python paper/tools/llm_audit.py strip --out D # write copies without any overlay macro to D
python paper/tools/llm_audit.py strip --in-place
python paper/tools/llm_audit.py status        # decisions recorded in the sources
"""

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PAPER_DIR = Path(__file__).resolve().parent.parent
CATALOGUE = Path(__file__).with_name('llm_audit.toml')
SOURCES = ('bispectrum.tex', 'checklist.tex')
CATEGORIES = 'VRJDSHCMX'

type Kind = Literal['span', 'point']

ID_RE = re.compile(rf'^[{CATEGORIES}]\d{{2}}[a-z]?$')
MACRO_RE = re.compile(r'\\(llmpt|llm)(?![A-Za-z@])')
FORBIDDEN_IN_NOTE = ('%', '#', '&', '\\cite', '\\ref', '\\label', '\\eqref', '\\pageref', '\\\\')


class AuditError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    file: str
    kind: Kind
    find: str
    note: str
    occurrence: int | None = None
    resolved: str | None = None


@dataclass(frozen=True, slots=True)
class Mark:
    macro: Literal['llm', 'llmpt']
    start: int
    end: int
    decision: str
    id: str
    text: str | None
    note: str


@dataclass(frozen=True, slots=True)
class Edit:
    start: int
    end: int
    replacement: str
    item_id: str


def _read_group(src: str, pos: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Read a balanced group starting at src[pos] == open_ch; return (content, end)."""
    if pos >= len(src) or src[pos] != open_ch:
        raise AuditError(f'expected {open_ch!r} at offset {pos}')
    depth = 0
    i = pos
    while i < len(src):
        ch = src[i]
        if ch == '\\':
            i += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return src[pos + 1 : i], i + 1
        i += 1
    raise AuditError(f'unbalanced {open_ch!r} group at offset {pos}')


def _skip_ws(src: str, pos: int) -> int:
    while pos < len(src) and src[pos] in ' \t\n':
        pos += 1
    return pos


def parse_marks(src: str) -> list[Mark]:
    """Outermost \\llm / \\llmpt calls in source order."""
    marks: list[Mark] = []
    pos = 0
    while (m := MACRO_RE.search(src, pos)) is not None:
        macro: Literal['llm', 'llmpt'] = 'llmpt' if m.group(1) == 'llmpt' else 'llm'
        i = _skip_ws(src, m.end())
        decision = ''
        if i < len(src) and src[i] == '[':
            decision, i = _read_group(src, i, '[', ']')
            i = _skip_ws(src, i)
        mark_id, i = _read_group(src, i, '{', '}')
        text: str | None = None
        if macro == 'llm':
            text, i = _read_group(src, _skip_ws(src, i), '{', '}')
        note, i = _read_group(src, _skip_ws(src, i), '{', '}')
        marks.append(Mark(macro, m.start(), i, decision.strip(), mark_id.strip(), text, note))
        pos = i
    return marks


def strip(src: str) -> str:
    out: list[str] = []
    pos = 0
    for mark in parse_marks(src):
        out.append(src[pos : mark.start])
        if mark.text is not None:
            out.append(strip(mark.text))
        pos = mark.end
    out.append(src[pos:])
    return ''.join(out)


def load_catalogue(path: Path = CATALOGUE) -> list[Item]:
    with path.open('rb') as fh:
        raw = tomllib.load(fh)
    items = [
        Item(
            id=entry['id'],
            file=entry.get('file', 'bispectrum.tex'),
            kind=entry.get('kind', 'span'),
            find=entry['find'],
            note=' '.join(entry['note'].split()),
            occurrence=entry.get('occurrence'),
            resolved=entry.get('resolved'),
        )
        for entry in raw['item']
    ]
    validate(items)
    return items


def validate(items: list[Item]) -> None:
    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not ID_RE.match(item.id):
            errors.append(f'{item.id}: malformed ID')
        if item.id in seen:
            errors.append(f'{item.id}: duplicate ID')
        seen.add(item.id)
        if item.kind not in ('span', 'point'):
            errors.append(f'{item.id}: kind must be span or point')
        if item.file not in SOURCES:
            errors.append(f'{item.id}: unknown file {item.file}')
        if not item.find.strip():
            errors.append(f'{item.id}: empty find')
        for bad in FORBIDDEN_IN_NOTE:
            if bad in item.note:
                errors.append(f'{item.id}: note contains {bad!r}')
        if not _balanced(item.note):
            errors.append(f'{item.id}: unbalanced braces or math delimiters in note')
        if item.resolved is not None and not item.resolved.strip():
            errors.append(f'{item.id}: empty resolved decision')
    if errors:
        raise AuditError('\n'.join(errors))


def _ws_regex(text: str) -> str:
    """Whitespace-insensitive regex for text; leading/trailing blanks become optional."""
    if not text.strip():
        return ''
    body = r'\s+'.join(re.escape(tok) for tok in text.split())
    lead = r'\s+' if text[:1].isspace() else ''
    trail = r'\s+' if text[-1:].isspace() else ''
    return lead + body + trail


def _find_pattern(find: str) -> re.Pattern[str]:
    """`find` may mark the target inside a longer context as before[[target]]after."""
    if '[[' not in find:
        return re.compile(f'({_ws_regex(find.strip())})')
    before, rest = find.split('[[', 1)
    target, after = rest.split(']]', 1)
    return re.compile(f'{_ws_regex(before)}({_ws_regex(target)}){_ws_regex(after)}')


def _balanced_braces(text: str) -> bool:
    try:
        _, end = _read_group('{' + text + '}', 0, '{', '}')
    except AuditError:
        return False
    return end == len(text) + 2


def _balanced(text: str) -> bool:
    return _balanced_braces(text) and len(re.findall(r'(?<!\\)\$', text)) % 2 == 0


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def plan_edits(src: str, items: list[Item]) -> list[Edit]:
    existing = parse_marks(src)
    present = {mark.id for mark in existing}
    masked = [(mark.start, mark.end) for mark in existing]
    edits: list[Edit] = []
    errors: list[str] = []
    for item in items:
        if item.id in present:
            continue
        hits = [
            h
            for h in _find_pattern(item.find).finditer(src)
            if not any(_overlaps((h.start(), h.end()), r) for r in masked)
        ]
        if not hits:
            errors.append(f'{item.id}: find string not found')
            continue
        if item.occurrence is None and len(hits) > 1:
            errors.append(f'{item.id}: find string matches {len(hits)} times; set occurrence')
            continue
        index = (item.occurrence or 1) - 1
        if index >= len(hits):
            errors.append(f'{item.id}: occurrence {item.occurrence} > {len(hits)} matches')
            continue
        start, end = hits[index].span(1)
        if item.kind == 'span':
            text = src[start:end]
            if not text:
                errors.append(f'{item.id}: empty span')
                continue
            if re.search(r'\n[ \t]*\n', text):
                errors.append(f'{item.id}: span crosses a paragraph break')
                continue
            if not _balanced(text):
                errors.append(f'{item.id}: span has unbalanced braces or math delimiters')
                continue
            edits.append(Edit(start, end, f'\\llm{{{item.id}}}{{{text}}}{{{item.note}}}', item.id))
        else:
            edits.append(Edit(end, end, f'\\llmpt{{{item.id}}}{{{item.note}}}', item.id))
    for i, a in enumerate(edits):
        for b in edits[i + 1 :]:
            same_point = a.start == a.end == b.start == b.end
            touches = a.start < b.end and b.start < a.end
            point_inside = (a.start == a.end and b.start < a.start < b.end) or (
                b.start == b.end and a.start < b.start < a.end
            )
            if touches or point_inside or same_point:
                errors.append(f'{a.item_id} and {b.item_id}: overlapping marks')
    if errors:
        raise AuditError('\n'.join(errors))
    return edits


def apply_edits(src: str, edits: list[Edit]) -> str:
    for edit in sorted(edits, key=lambda e: (e.start, e.end), reverse=True):
        src = src[: edit.start] + edit.replacement + src[edit.end :]
    return src


def annotate(items: list[Item], paper_dir: Path = PAPER_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    planned: dict[Path, tuple[str, list[Edit]]] = {}
    for name in SOURCES:
        path = paper_dir / name
        src = path.read_text()
        edits = plan_edits(src, [item for item in items if item.file == name and item.resolved is None])
        planned[path] = (src, edits)
        counts[name] = len(edits)
    for path, (src, edits) in planned.items():
        if edits:
            path.write_text(apply_edits(src, edits))
    return counts


def status(items: list[Item], paper_dir: Path = PAPER_DIR) -> list[str]:
    rows: list[str] = []
    found: set[str] = set()
    for name in SOURCES:
        src = (paper_dir / name).read_text()
        for mark in parse_marks(src):
            found.add(mark.id)
            line = src.count('\n', 0, mark.start) + 1
            snippet = ' '.join((mark.text or '').split())[:60]
            rows.append(f'{mark.id:6} {mark.decision or "-":8} {name}:{line:<5} {snippet}')
    for item in items:
        if item.id in found:
            continue
        label = f'done:{item.resolved}' if item.resolved else 'gone'
        rows.append(f'{item.id:6} {label:8} (not in sources)')
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('annotate')
    strip_parser = sub.add_parser('strip')
    target = strip_parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--out', type=Path)
    target.add_argument('--in-place', action='store_true')
    sub.add_parser('status')
    args = parser.parse_args(argv)

    try:
        match args.command:
            case 'annotate':
                counts = annotate(load_catalogue())
                for name, n in counts.items():
                    print(f'{name}: {n} marks added')
            case 'strip':
                for name in SOURCES:
                    stripped = strip((PAPER_DIR / name).read_text())
                    dest = PAPER_DIR / name if args.in_place else args.out / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(stripped)
                    print(f'{name} -> {dest}')
            case 'status':
                print('\n'.join(status(load_catalogue())))
    except AuditError as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
