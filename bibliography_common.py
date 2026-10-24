"""One set of venue names and conference ordering for website and CV outputs."""
import json
import re
from pathlib import Path

RULES = json.loads((Path(__file__).resolve().parent / '_data' / 'venues.json').read_text(encoding='utf-8'))
VENUE_ABBREVIATIONS = RULES['abbreviations']
CONFERENCE_TIME_ORDER = RULES['conference_order']
CCF_A_VENUES = set(RULES['ccf_a'])
JCR_Q1_JOURNALS = set(RULES['jcr_q1_journals'])
ICORE_A_STAR_VENUES = set(RULES['icore_a_star'])


def normalize_venue(venue):
    return re.sub(r'\s+', ' ', venue.replace('\\', '').replace('&', 'and')
                  .replace('{', '').replace('}', '')).strip()


def abbreviate_venue(venue):
    """Match full names before shorter names, with case-insensitive word boundaries."""
    cleaned = normalize_venue(venue)
    aliases = dict(VENUE_ABBREVIATIONS)
    for name in set(VENUE_ABBREVIATIONS.values()):
        aliases.setdefault(name, name)
    for keyword in sorted(aliases, key=lambda name: len(normalize_venue(name)), reverse=True):
        pattern = r'(?<!\w)' + re.escape(normalize_venue(keyword)) + r'(?!\w)'
        if re.search(pattern, cleaned, re.IGNORECASE):
            return aliases[keyword]
    return cleaned


def is_arxiv(entry):
    """Hide arXiv-only records, keeping published papers with arXiv PDF links."""
    if entry.get('status'):
        return entry['status'] == 'preprint'
    venue = entry.get('booktitle', entry.get('journal', ''))
    if re.search(r'\barxiv\b', venue, re.IGNORECASE):
        return True
    return not venue.strip() and bool(entry.get('arxivid') or
                                      re.search(r'arxiv\.org/', entry.get('url', ''), re.IGNORECASE))


def publication_status(entry):
    """Preserve explicit editorial status; infer only from existing venue metadata."""
    if entry.get('status'):
        return entry['status']
    if is_arxiv(entry):
        return 'preprint'
    return 'published' if entry.get('booktitle') or entry.get('journal') else 'unknown'


def is_visible(entry):
    return not entry.get('duplicate_of') and publication_status(entry) in ('accepted', 'published')


def publication_sort_key(entry):
    name = abbreviate_venue(entry.get('booktitle', entry.get('journal', '')))
    rank = CONFERENCE_TIME_ORDER.index(name) if name in CONFERENCE_TIME_ORDER else len(CONFERENCE_TIME_ORDER)
    return (-int(entry.get('year', 0)), rank, name.lower(), entry.get('ID', ''))


def clean_link(value):
    """Reject placeholders and unsafe URLs; repair old arXiv URL formatting."""
    from urllib.parse import urlsplit
    value = (value or '').strip().replace(r'\_', '_')
    if not value or value in ('#', 'TODO', 'TBD') or value.endswith('#'):
        return ''
    if 'arxiv.org/' in value.lower():
        match = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', value)
        if not match:
            return ''
        kind = 'pdf' if '/pdf/' in value else 'abs'
        return f'https://arxiv.org/{kind}/{match[0]}'
    parsed = urlsplit(value)
    if parsed.scheme in ('http', 'https') and parsed.netloc and not re.search(r'\s', value):
        return value
    return ''


def publication_links(entry, root=None):
    """Prefer an existing canonical PDF; otherwise use a verified external PDF/page."""
    root = Path(root) if root else Path(__file__).resolve().parent
    local = root / 'pdfs' / f'{entry["ID"]}.pdf'
    pdf = ''
    if local.is_file():
        with local.open('rb') as handle:
            if handle.read(5) == b'%PDF-':
                pdf = f'/pdfs/{entry["ID"]}.pdf'
    external = clean_link(entry.get('pdf_url'))
    url = clean_link(entry.get('url'))
    if not pdf:
        pdf = external
    if not pdf and url:
        if 'arxiv.org/abs/' in url:
            pdf = url.replace('/abs/', '/pdf/')
        elif url.lower().split('?')[0].endswith('.pdf') or 'openreview.net/pdf?' in url:
            pdf = url
    paper = clean_link(entry.get('paper_url')) or url or clean_link('https://doi.org/' + entry['doi'] if entry.get('doi') else '')
    return {'pdf': pdf, 'paper': paper, 'code': clean_link(entry.get('code_url') or entry.get('code'))}


def publication_ratings(entry):
    """Classify the venue, even when conference papers use the BibTeX journal field."""
    venue = entry.get('booktitle', entry.get('journal', ''))
    name = abbreviate_venue(venue)
    labels = []
    # CCF covers regular/full papers, excluding Findings and companion formats.
    excluded = re.search(r'\b(arxiv|preprint|findings|workshops?|short paper|demo|'
                         r'technical brief|summary)\b',
                         venue + ' ' + entry.get('note', ''), re.IGNORECASE)
    if name in CCF_A_VENUES and not excluded:
        labels.append('CCF-A')
    # Use a journal whitelist: @article and `journal` also store conferences here.
    if name in JCR_Q1_JOURNALS and not excluded:
        labels.append('JCR Q1')
    if name in ICORE_A_STAR_VENUES and not excluded:
        labels.append('ICORE A*')
    return labels
