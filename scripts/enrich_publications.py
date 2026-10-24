"""Incrementally retrieve sourced abstracts, links and optional open-access PDFs.

Network access is explicit: this script is never called by generation or CI.
Existing abstracts/summaries and PDFs are preserved. Failed lookups remain in the audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import logging
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.publication_catalog import ROOT, load_entries, normalized_title, update_fields, build_catalog
from bibliography_common import clean_link, publication_status

CACHE = ROOT / '.venv/publication-cache'
_locks = {}
_guard = threading.Lock()
_last = {}


def request(url, binary=False):
    key = hashlib.sha256(url.encode()).hexdigest()
    path = CACHE / (key + ('.pdf' if binary else '.txt'))
    if path.is_file():
        data = path.read_bytes()
        return data if binary else data.decode('utf-8')
    host = urllib.parse.urlsplit(url).netloc
    with _guard:
        lock = _locks.setdefault(host, threading.Lock())
    with lock:
        delay = 3.1 if 'arxiv' in host else 0.4
        time.sleep(max(0, delay - (time.monotonic() - _last.get(host, 0))))
        _last[host] = time.monotonic()
        req = urllib.request.Request(url, headers={'User-Agent': 'Academic-publication-catalog/1.0 (https://vanoracai.github.io/)'})
        with urllib.request.urlopen(req, timeout=25) as response:
            # Never cache arbitrarily large responses or HTML posing as a PDF.
            data = response.read((80 if binary else 8) * 1024 * 1024 + 1)
            if len(data) > (80 if binary else 8) * 1024 * 1024:
                raise ValueError('Response exceeds size limit')
            if binary and not data.startswith(b'%PDF-'):
                raise ValueError('Response is not a PDF')
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data if binary else data.decode('utf-8', errors='replace')


def safe_request(url, binary=False):
    try:
        return request(url, binary)
    except (OSError, ValueError, urllib.error.URLError) as error:
        return None


def plain(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', html.unescape(re.sub(r'<[^>]+>', ' ', value or '')))).strip()


def title_matches(a, b):
    a, b = normalized_title(a), normalized_title(b)
    return bool(a and b) and (a == b or SequenceMatcher(None, a, b).ratio() >= 0.94)


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}
        self.depth = 0
        self.abstract = []
        self.abstract_links = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'meta':
            self.meta[(attrs.get('name') or attrs.get('property') or '').lower()] = attrs.get('content', '')
        if self.depth:
            if tag == 'a' and attrs.get('href'):
                self.abstract_links.append(attrs['href'])
            if tag not in ('br', 'img', 'hr', 'meta', 'input', 'link'):
                self.depth += 1
        elif (attrs.get('id') == 'abstract' or
              'abstract' in attrs.get('class', '').split() or
              'acl-abstract' in attrs.get('class', '').split()):
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth and tag not in ('br', 'img', 'hr', 'meta', 'input', 'link'):
            self.depth -= 1

    def handle_data(self, value):
        if self.depth:
            self.abstract.append(value)


def from_page(url, title):
    if not url or '/pdf/' in url or url.lower().split('?')[0].endswith('.pdf') or '/pdf?' in url:
        return {}
    text = safe_request(url)
    if not text:
        return {}
    page = Metadata()
    page.feed(text)
    actual_title = page.meta.get('citation_title') or page.meta.get('dc.title') or page.meta.get('og:title', '')
    actual_title = re.sub(r'^\[\d{4}\.\d+\]\s*', '', actual_title)
    if not title_matches(title, actual_title):
        return {}
    abstract = plain(' '.join(page.abstract))
    abstract = re.sub(r'^\s*Abstract\s*:?\s*', '', abstract, flags=re.I)
    if len(abstract) < 100:
        abstract = ''
    code = next((clean_link(link) for link in page.abstract_links
                 if re.match(r'https://github\.com/[\w.-]+/[\w.-]+', link)), '')
    return {k: v for k, v in {'abstract': abstract, 'abstract_source': url if abstract else '',
             'paper_url': url, 'pdf_url': clean_link(page.meta.get('citation_pdf_url')),
             'code_url': code, 'code_source': url if code else ''}.items() if v}


def from_openreview(url, title):
    note_id = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get('id', [''])[0]
    if not note_id:
        return {}
    for api in ('https://api2.openreview.net/notes?id=', 'https://api.openreview.net/notes?id='):
        raw = safe_request(api + urllib.parse.quote(note_id))
        if not raw:
            continue
        for note in json.loads(raw).get('notes', []):
            def value(field):
                result = note.get('content', {}).get(field, '')
                return result.get('value', '') if isinstance(result, dict) else result
            if not title_matches(title, value('title')):
                continue
            source = 'https://openreview.net/forum?id=' + note_id
            result = {'paper_url': source, 'pdf_url': 'https://openreview.net/pdf?id=' + note_id}
            if value('abstract'):
                result.update(abstract=plain(value('abstract')), abstract_source=source)
            code = value('code') or value('code_url')
            if isinstance(code, str) and clean_link(code):
                result['code_url'] = clean_link(code)
            return result
    return {}


def from_doi(doi, title):
    raw = safe_request('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe=''))
    if not raw:
        return {}
    data = json.loads(raw).get('message', {})
    if not title_matches(title, ' '.join(data.get('title', []))):
        return {}
    result = {'paper_url': 'https://doi.org/' + doi}
    if data.get('abstract'):
        result.update(abstract=plain(data['abstract']), abstract_source='https://doi.org/' + doi)
    for link in data.get('link', []):
        if link.get('content-type') == 'application/pdf' and clean_link(link.get('URL')):
            result['pdf_url'] = link['URL']
            break
    return result


def from_openalex(entry):
    doi = entry.get('doi')
    if doi:
        url = 'https://api.openalex.org/works/https://doi.org/' + urllib.parse.quote(doi, safe='/')
    else:
        url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': entry['title'], 'per-page': 3})
    raw = safe_request(url)
    if not raw:
        return {}
    data = json.loads(raw)
    candidates = data.get('results', [data])
    for item in candidates:
        if not title_matches(entry['title'], item.get('title', '')):
            continue
        # Avoid accepting a near-identical title from unrelated authors.
        names = ' '.join(a.get('author', {}).get('display_name', '') for a in item.get('authorships', []))
        if 'cai' not in names.lower():
            continue
        locations = item.get('locations') or []
        result = {}
        for location in locations:
            landing = clean_link(location.get('landing_page_url'))
            if landing and any(domain in landing for domain in ('arxiv.org', 'openaccess.thecvf.com', 'aclanthology.org', 'proceedings.neurips.cc', 'proceedings.mlr.press')):
                result.update(from_page(landing, entry['title']))
                if result.get('abstract'):
                    break
        # OpenAlex supplies publisher-derived abstracts; record the actual metadata source.
        index = item.get('abstract_inverted_index')
        if not result.get('abstract') and index:
            words = {pos: word for word, positions in index.items() for pos in positions}
            result.update(abstract=' '.join(words[pos] for pos in sorted(words)),
                          abstract_source=item['id'], abstract_method='openalex_metadata')
        if not result.get('paper_url'):
            result['paper_url'] = clean_link((item.get('primary_location') or {}).get('landing_page_url')) or ''
        if not result.get('pdf_url'):
            result['pdf_url'] = next((clean_link(loc.get('pdf_url')) for loc in locations if loc.get('is_oa') and clean_link(loc.get('pdf_url'))), '')
        return {k: v for k, v in result.items() if v}
    return {}


def summary_from_abstract(abstract):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract)
    selected = []
    for sentence in sentences[:3]:
        if selected and len((' '.join(selected) + ' ' + sentence).split()) > 110:
            break
        selected.append(sentence)
    return ' '.join(selected)


def import_local_pdfs(entries):
    """Copy confidently matched legacy files into the canonical archive, preserving originals."""
    from pypdf import PdfReader
    copied = []
    for source in sorted((ROOT / 'Paper').rglob('*.pdf')):
        try:
            reader = PdfReader(source)
            opening = ' '.join((page.extract_text() or '') for page in reader.pages[:1])
        except Exception:
            continue
        words = set(re.findall(r'[a-z0-9]+', opening.lower()))
        for entry in entries:
            target = ROOT / 'pdfs' / f'{entry["ID"]}.pdf'
            if target.exists():
                continue
            title_words = set(re.findall(r'[a-z0-9]+', entry['title'].lower()))
            if len(title_words) < 5 or len(title_words & words) / len(title_words) < 0.95:
                continue
            if 'cai' not in words:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.pdf.tmp')
            temporary.write_bytes(source.read_bytes())
            temporary.replace(target)
            copied.append((entry['ID'], source.relative_to(ROOT).as_posix()))
    return copied


def abstract_from_pdf(data):
    """Require both abstract and end headings; do not guess from an arbitrary snippet."""
    import io
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        text = unicodedata.normalize('NFKC', '\n'.join((page.extract_text(extraction_mode='layout') or '') for page in reader.pages[:2]))
        match = re.search(r'(?is)\babstract\b\s*[:\-\u2014]?\s*(.+?)(?=\n\s*(?:1\.?\s+introduction|i\.\s+introduction|index terms|keywords|ccs concepts))', text)
        if match:
            text = plain(match[1])
            if 60 <= len(text.split()) <= 450:
                return text
    except Exception:
        pass
    return ''


def enrich(entry, peer_entries, download):
    original = dict(entry)
    changes = {}
    for field in ('url', 'code_url', 'code', 'pdf_url', 'paper_url'):
        if field in entry:
            cleaned = clean_link(entry[field])
            if cleaned != entry[field]:
                changes[field] = cleaned or None
                entry[field] = cleaned
    if not entry.get('status'):
        changes.update(status=publication_status(entry), status_source='existing_metadata')
    if (entry.get('abstract') and entry.get('summary') and entry.get('abstract_source')
            and (entry.get('paper_url') or clean_link(entry.get('url')))
            and (not download or (ROOT / 'pdfs' / f'{entry["ID"]}.pdf').is_file())):
        return changes
    candidates = [entry.get('paper_url'), entry.get('url')]
    for peer in peer_entries:
        if title_matches(entry['title'], peer.get('title', '')) and 'cai' in peer.get('author', '').lower():
            candidates.append(clean_link(peer.get('url')))
            if not entry.get('doi') and peer.get('doi'):
                # DOI is still checked by title below before use.
                doi_result = from_doi(peer['doi'], entry['title'])
                if doi_result:
                    changes['doi'] = peer['doi']
                    changes.update(doi_result)
    for field in ('arxivid', 'journal', 'eprint'):
        match = re.search(r'\d{4}\.\d{4,5}', entry.get(field, ''))
        if match:
            candidates.append('https://arxiv.org/abs/' + match[0])
    for url in dict.fromkeys(filter(None, candidates)):
        url = clean_link(url)
        if 'arxiv.org/pdf/' in url:
            url = url.replace('/pdf/', '/abs/')
        if 'openaccess.thecvf.com' in url and url.endswith('.pdf'):
            url = url.replace('/papers/', '/html/').replace('.pdf', '.html')
        result = from_openreview(url, entry['title']) if 'openreview.net' in url else from_page(url, entry['title'])
        changes.update({k: v for k, v in result.items() if v and not original.get(k)})
        if changes.get('abstract') or entry.get('abstract'):
            break
    if not (entry.get('abstract') or changes.get('abstract')) and entry.get('doi'):
        changes.update({k: v for k, v in from_doi(entry['doi'], entry['title']).items() if not original.get(k)})
    if not (entry.get('abstract') or changes.get('abstract')) or not (changes.get('pdf_url') or entry.get('pdf_url') or (ROOT / 'pdfs' / f'{entry["ID"]}.pdf').exists()):
        changes.update({k: v for k, v in from_openalex({**entry, **changes}).items() if v and not original.get(k) and not changes.get(k)})
    abstract = entry.get('abstract') or changes.get('abstract')
    if abstract and not entry.get('summary'):
        changes.update(summary=summary_from_abstract(abstract), summary_method='extractive',
                       summary_source=entry.get('abstract_source') or changes.get('abstract_source'))
    if abstract and not clean_link(entry.get('code_url') or entry.get('code')):
        match = re.search(r'(?:code|implementation)[^.]{0,100}?(https?://github\.com/[\w.-]+/[\w.-]+)', abstract, re.I)
        if match:
            changes.update(code_url=match[1].rstrip('.'),
                           code_source=entry.get('abstract_source') or changes.get('abstract_source'))
    if changes.get('abstract') and not changes.get('abstract_method'):
        changes['abstract_method'] = 'source_text'
    if changes.get('abstract') and not entry.get('abstract'):
        changes['abstract_retrieved'] = time.strftime('%Y-%m-%d', time.gmtime())
    combined = {**entry, **{k: v for k, v in changes.items() if v is not None}}
    if download and not (ROOT / 'pdfs' / f'{entry["ID"]}.pdf').exists():
        candidates_pdf = [combined.get('pdf_url')]
        candidates_pdf += ['https://wangywust.github.io/pdfs/' + peer['ID'] + '.pdf'
                           for peer in peer_entries if title_matches(entry['title'], peer.get('title', ''))
                           and 'cai' in peer.get('author', '').lower()]
        url = clean_link(combined.get('url'))
        if 'arxiv.org/abs/' in url:
            candidates_pdf.append(url.replace('/abs/', '/pdf/'))
        if url.endswith('.pdf') or 'openreview.net/pdf?' in url:
            candidates_pdf.append(url)
        for pdf_url in dict.fromkeys(filter(None, candidates_pdf)):
            data = safe_request(pdf_url, binary=True)
            if not data:
                continue
            # Compare the downloaded first page with the expected title before publishing.
            import io
            from pypdf import PdfReader
            try:
                reader = PdfReader(io.BytesIO(data))
                text = unicodedata.normalize('NFKC', ' '.join((page.extract_text() or '') for page in reader.pages[:2]))
                title_words = set(re.findall(r'[a-z0-9]+', entry['title'].lower()))
                words = set(re.findall(r'[a-z0-9]+', text.lower()))
                if not title_words or len(title_words & words) / len(title_words) < 0.85:
                    continue
            except Exception:
                continue
            target = ROOT / 'pdfs' / f'{entry["ID"]}.pdf'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.pdf.tmp')
            temporary.write_bytes(data)
            if not target.exists():
                temporary.replace(target)
            else:
                temporary.unlink()
            changes.update(pdf_url=pdf_url, pdf_source=pdf_url)
            break
    target = ROOT / 'pdfs' / f'{entry["ID"]}.pdf'
    if download and not abstract and target.is_file():
        extracted = abstract_from_pdf(target.read_bytes())
        if extracted:
            source = changes.get('pdf_source') or entry.get('pdf_source') or 'https://vanoracai.github.io/pdfs/' + entry['ID'] + '.pdf'
            changes.update(abstract=extracted, abstract_source=source, abstract_method='pdf_text',
                           abstract_retrieved=time.strftime('%Y-%m-%d', time.gmtime()))
            if not entry.get('summary'):
                changes.update(summary=summary_from_abstract(extracted), summary_source=source, summary_method='extractive')
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download and verify open PDFs; requires pypdf')
    parser.add_argument('--keys', nargs='*', help='Limit to selected citation keys')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.download:
        import pypdf
        logging.getLogger('pypdf').setLevel(logging.ERROR)
    # Peer metadata is a discovery aid only; never copy its AI summaries into abstract.
    peer_url = 'https://raw.githubusercontent.com/wangywUST/wangywust.github.io/master/MetaData/data/citations.bib'
    peer = safe_request(peer_url)
    import bibtexparser
    peers = bibtexparser.loads(peer).entries if peer else []
    entries = [entry for entry in load_entries() if not args.keys or entry['ID'] in args.keys]
    if args.download:
        for key, source in import_local_pdfs(entries):
            update_fields(ROOT / 'citations.bib', {key: {'pdf_source': 'https://vanoracai.github.io/' + source}})
            print(f'{key}: imported existing {source}', flush=True)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        futures = {pool.submit(enrich, dict(entry), peers, args.download): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                changes = future.result()
                update_fields(ROOT / 'citations.bib', {entry['ID']: changes})
                print(f'{entry["ID"]}: ' + ', '.join(changes), flush=True)
            except Exception as error:
                print(f'{entry["ID"]}: lookup failed: {type(error).__name__}: {error}', flush=True)
    print(json.dumps(build_catalog(), indent=2))


if __name__ == '__main__':
    main()
