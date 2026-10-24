"""Offline publication validation, PDF inventory and completeness reports."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bibtexparser
from bibliography_common import clean_link, publication_status, publication_links, is_visible

MANIFEST_FIELDS = ('citekey', 'title', 'source_url', 'pdf_url', 'filename', 'status', 'sha256', 'message')


def load_entries(root=ROOT):
    return bibtexparser.loads((Path(root) / 'citations.bib').read_text(encoding='utf-8')).entries


def normalized_title(value):
    value = unicodedata.normalize('NFKC', value)
    return re.sub(r'[^\w]', '', value.lower().replace('{', '').replace('}', ''))


def validate_entries(entries):
    ids = set()
    for entry in entries:
        key = entry['ID']
        if key in ids or not re.fullmatch(r'[A-Za-z0-9_.-]+', key):
            raise ValueError(f'Duplicate or unsafe bibliography key: {key}')
        ids.add(key)
        if not entry.get('title') or not entry.get('author') or not entry.get('year', '').isdigit():
            raise ValueError(f'{key}: title, author and numeric year are required')
        if publication_status(entry) not in ('published', 'accepted', 'preprint', 'unknown'):
            raise ValueError(f'{key}: invalid publication status')
        for field in ('url', 'paper_url', 'pdf_url', 'code_url', 'code', 'abstract_source', 'summary_source'):
            if entry.get(field) and not clean_link(entry[field]):
                raise ValueError(f'{key}: invalid {field}: {entry[field]}')
        for field in ('abstract', 'summary'):
            if entry.get(field) and not clean_link(entry.get(field + '_source')):
                raise ValueError(f'{key}: {field} needs a source URL')
    by_id = {entry['ID']: entry for entry in entries}
    for entry in entries:
        target = entry.get('duplicate_of')
        if target and (target not in ids or target == entry['ID'] or by_id[target].get('duplicate_of')):
            raise ValueError(f'{entry["ID"]}: duplicate_of must reference a canonical record')


def pdf_info(path):
    if not path.is_file():
        return False, '', 'No local PDF; external links remain available when known'
    with path.open('rb') as handle:
        if handle.read(5) != b'%PDF-':
            return False, '', 'Invalid PDF header'
    if path.stat().st_size < 1024:
        return False, '', 'Truncated PDF'
    return True, hashlib.sha256(path.read_bytes()).hexdigest(), ''


def write_if_changed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8')


def build_catalog(root=ROOT):
    root = Path(root)
    entries = load_entries(root)
    validate_entries(entries)
    old_path = root / 'pdfs/paper_pdf_manifest.csv'
    previous = {}
    if old_path.is_file():
        with old_path.open(encoding='utf-8-sig', newline='') as handle:
            previous = {row['citekey']: row for row in csv.DictReader(handle)}
    rows, records, missing = [], [], []
    groups = defaultdict(list)
    dois = defaultdict(list)
    for entry in entries:
        key = entry['ID']
        groups[normalized_title(entry['title'])].append(key)
        if entry.get('doi'):
            dois[entry['doi'].lower()].append(key)
        valid, digest, message = pdf_info(root / 'pdfs' / f'{key}.pdf')
        old = previous.get(key, {})
        status = 'exists' if valid else ('invalid' if (root / 'pdfs' / f'{key}.pdf').exists() else 'missing')
        if not valid and old.get('message'):
            message = old['message']
        links = publication_links(entry, root)
        rows.append(dict(zip(MANIFEST_FIELDS, (key, entry['title'], entry.get('pdf_source') or links['paper'],
                    links['pdf'], f'{key}.pdf', status, digest, message))))
        absent = [field for field in ('abstract', 'summary') if not entry.get(field)]
        if not links['paper'] and not links['pdf']:
            absent.append('paper_url')
        if not valid:
            absent.append('local_pdf')
        if absent:
            missing.append((key, absent))
        records.append({**entry, 'displayed': is_visible(entry), 'links': links, 'local_pdf': valid})
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    write_if_changed(old_path, buffer.getvalue())
    write_if_changed(root / '_generated/publications.json', json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    counts = {'total': len(entries), 'visible': sum(is_visible(e) for e in entries),
              'abstracts': sum(bool(e.get('abstract')) for e in entries),
              'summaries': sum(bool(e.get('summary')) for e in entries),
              'local_pdfs': sum(row['status'] == 'exists' for row in rows)}
    lines = ['# Publication audit', '', 'Generated offline from citations.bib. Do not edit this report.', '',
             *[f'- {key}: {value}' for key, value in counts.items()], '',
             'Status inferred from existing venue metadata is not independent acceptance verification.', '',
             '## Missing information', '', '| Citekey | Missing |', '| --- | --- |',
             *[f'| {key} | {", ".join(fields)} |' for key, fields in missing], '',
             '## Possible duplicates (retained for review)', '']
    for name, mapping in [('Title', groups), ('DOI', dois)]:
        lines += [f'- {name}: {", ".join(keys)}' for keys in mapping.values() if len(keys) > 1]
    expected = {f'{entry["ID"]}.pdf' for entry in entries}
    extras = sorted(path.name for path in (root / 'pdfs').glob('*.pdf') if path.name not in expected)
    if extras:
        lines += ['', '## Unreferenced PDFs (preserved)', '', *[f'- {name}' for name in extras]]
    write_if_changed(root / '_generated/publication_audit.md', '\n'.join(lines) + '\n')
    return counts


def update_fields(path, changes):
    """Patch selected entries only, preserving comments, order and existing formatting."""
    path = Path(path)
    original = path.read_text(encoding='utf-8')
    text = original
    for key, fields in changes.items():
        match = re.search(r'^@(?!comment\b)\w+\s*\{\s*' + re.escape(key) + r'\s*,', text, re.M | re.I)
        if not match:
            raise ValueError(f'Entry not found: {key}')
        start = text.index('{', match.start())
        depth, end = 0, None
        for pos in range(start, len(text)):
            if text[pos] in '{}':
                backslashes, before = 0, pos - 1
                while before >= 0 and text[before] == '\\':
                    backslashes += 1
                    before -= 1
                if backslashes % 2:
                    continue
                depth += 1 if text[pos] == '{' else -1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            raise ValueError(f'Unbalanced entry: {key}')
        block = text[match.end():end]
        for field, value in fields.items():
            pattern = r'^\s*' + re.escape(field) + r'\s*=\s*\{'
            existing = re.search(pattern, block, re.M | re.I)
            # Updating existing fields uses parsed balanced braces, not a line regex.
            if existing:
                brace = block.index('{', existing.start())
                level, stop = 1, brace + 1
                while level and stop < len(block):
                    if block[stop] in '{}' and block[stop - 1] != '\\':
                        level += 1 if block[stop] == '{' else -1
                    stop += 1
                tail = stop
                if tail < len(block) and block[tail] == ',':
                    tail += 1
                block = block[:existing.start()] + '\n' + block[tail:]
            if value is not None:
                value = re.sub(r'\s+', ' ', value).strip()
                value = value.replace('{', r'\{').replace('}', r'\}')
                block = block.rstrip().rstrip(',') + f',\n {field} = {{{value}}},\n'
        text = text[:match.end()] + block + text[end:]
    old_entries = bibtexparser.loads(original).entries
    new_entries = bibtexparser.loads(text).entries
    if [e['ID'] for e in old_entries] != [e['ID'] for e in new_entries]:
        raise ValueError('BibTeX patch changed entry identities; refusing to write')
    old_map = {entry['ID']: entry for entry in old_entries}
    for entry in new_entries:
        allowed = changes.get(entry['ID'], {})
        for field in set(entry) | set(old_map[entry['ID']]):
            if field not in allowed and entry.get(field) != old_map[entry['ID']].get(field):
                raise ValueError(f'Unexpected mutation: {entry["ID"]}.{field}')
    temporary = path.with_suffix('.bib.tmp')
    temporary.write_text(text, encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows previews/watchers may briefly hold a read handle.
            if attempt == 7:
                raise
            time.sleep(0.1 * (attempt + 1))


if __name__ == '__main__':
    print(json.dumps(build_catalog(), indent=2))
