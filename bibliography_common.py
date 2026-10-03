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
    venue = entry.get('booktitle', entry.get('journal', ''))
    if re.search(r'\barxiv\b', venue, re.IGNORECASE):
        return True
    return not venue.strip() and bool(entry.get('arxivid') or
                                      re.search(r'arxiv\.org/', entry.get('url', ''), re.IGNORECASE))


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
