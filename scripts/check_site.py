"""Check essential generated page links, assets, and homepage-only profile."""
from html.parser import HTMLParser
import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / '_site'
PAGES = ('index.html', 'publications.html', 'students.html', 'teaching.html',
         'services.html', 'awards.html', 'sponsors.html')


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.profile = False
        self.images = []
        self.nav_depth = 0
        self.nav_links = []
        self.cv_links = {}
        self.publication_depth = 0
        self.publication_links = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        classes = attrs.get('class', '').split()
        if tag == 'ol' and ('bibliography' in classes or self.publication_depth):
            self.publication_depth += 1
        if tag == 'a' and self.publication_depth:
            self.publication_links.append(attrs.get('href', ''))
        if 'profile-section' in classes:
            self.profile = True
        if tag == 'img':
            self.images.append(attrs.get('src', ''))
        if tag == 'nav':
            if 'site-nav' in classes or self.nav_depth:
                self.nav_depth += 1
        if tag == 'a' and self.nav_depth:
            self.nav_links.append(attrs.get('href', ''))
        if tag == 'a' and attrs.get('data-cv-language'):
            self.cv_links[attrs['data-cv-language']] = attrs.get('href', '')

    def handle_endtag(self, tag):
        if tag == 'ol' and self.publication_depth:
            self.publication_depth -= 1
        if tag == 'nav' and self.nav_depth:
            self.nav_depth -= 1


def main():
    global SITE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=SITE)
    SITE = parser.parse_args().site.resolve()
    profile = json.loads((ROOT / '_data/profile.json').read_text(encoding='utf-8'))
    for filename in PAGES:
        page = Page()
        page.feed((SITE / filename).read_text(encoding='utf-8'))
        if filename == 'publications.html' and 'publication-details' in (SITE / filename).read_text(encoding='utf-8'):
            raise ValueError('Publication abstracts/summaries must remain local records')
        if page.profile != (filename == 'index.html'):
            raise ValueError(f'{filename}: incorrect profile visibility')
        if len(page.nav_links) != 7:  # Brand link plus the six visible section links.
            raise ValueError(f'{filename}: expected six navigation sections')
        for link in page.nav_links:
            destination = link.lstrip('/') or 'index.html'
            if not (SITE / destination).is_file():
                raise ValueError(f'{filename}: missing navigation target {link}')
        if profile['school_logo'] not in page.images:
            raise ValueError(f'{filename}: missing school logo')
        for image in page.images:
            if not image:
                raise ValueError(f'{filename}: empty image URL')
        for link in page.publication_links:
            parsed = urlsplit(link)
            if not link or link == '#' or link.endswith('#') or parsed.scheme not in ('', 'http', 'https'):
                raise ValueError(f'{filename}: invalid publication link {link}')
            if not parsed.scheme and not parsed.netloc:
                target = (SITE / unquote(parsed.path).lstrip('/')).resolve()
                if not target.is_relative_to(SITE) or not target.is_file():
                    raise ValueError(f'{filename}: missing publication target {link}')
        if filename == 'index.html':
            for language, key in [('en', 'cv_link'), ('zh', 'cv_zh_link')]:
                link = page.cv_links.get(language, '')
                parsed = urlsplit(link)
                expected = profile.get(key)
                if not expected or unquote(parsed.path) != expected:
                    raise ValueError(f'{language}: homepage CV icon has the wrong PDF link')
                source = ROOT / expected.lstrip('/')
                built = SITE / expected.lstrip('/')
                if not built.is_file() or not built.read_bytes().startswith(b'%PDF-'):
                    raise ValueError(f'{language}: CV PDF is missing from the built site')
                if built.read_bytes() != source.read_bytes():
                    raise ValueError(f'{language}: website CV PDF is stale')
                print(f'{language} CV: icon link and current PDF OK')
        print(f'{filename}: navigation, logo and profile OK')
    if not (SITE / profile['school_logo'].lstrip('/')).is_file():
        raise ValueError('Built school logo file is missing')
    for forbidden in ('reference-wangywust', '.venv', 'scripts', 'tests', 'citations.bib'):
        if (SITE / forbidden).exists():
            raise ValueError(f'Build contains development directory {forbidden}')


if __name__ == '__main__':
    main()
