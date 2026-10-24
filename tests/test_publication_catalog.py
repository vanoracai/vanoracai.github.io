"""Protect sourced metadata, non-destructive updates and publication link fallbacks."""
import csv
import tempfile
import unittest
from pathlib import Path

from bibliography_common import clean_link, is_visible, publication_links
from bib_to_html import generate_html
from scripts.publication_catalog import build_catalog, update_fields, validate_entries
from scripts.enrich_publications import from_page, title_matches
from unittest.mock import patch


class PublicationCatalogTests(unittest.TestCase):
    def test_discovery_rejects_a_different_paper(self):
        with patch('scripts.enrich_publications.safe_request', return_value='<meta name="citation_title" content="Unrelated paper"><div id="abstract">' + 'Wrong abstract. ' * 20 + '</div>'):
            self.assertEqual(from_page('https://example.org/paper', 'Expected paper'), {})
        self.assertTrue(title_matches('A {Nested} Title', 'A Nested Title.'))
        self.assertTrue(title_matches('Node Classification', 'Node Classiﬁcation'))

    def entry(self, **fields):
        return dict(ID='paper', title='A paper', author='Cai, Yujun', year='2025',
                    booktitle='ICML', **fields)

    def test_explicit_status_controls_both_exports(self):
        self.assertTrue(is_visible(self.entry(status='accepted')))
        self.assertFalse(is_visible(self.entry(status='preprint')))
        self.assertFalse(is_visible(self.entry(duplicate_of='other')))

    def test_placeholder_and_unsafe_links_are_never_rendered(self):
        for link in ('#', 'https://example.org/publications#', 'javascript:alert(1)'):
            self.assertEqual(clean_link(link), '')
            rendered = generate_html(self.entry(code_url=link))
            self.assertNotIn('>Code</a>', rendered)
        self.assertNotIn('>PDF</a>', generate_html(self.entry()))

    def test_missing_local_pdf_falls_back_to_publisher_page(self):
        with tempfile.TemporaryDirectory() as directory:
            links = publication_links(self.entry(url='https://doi.org/10.1234/test'), directory)
            self.assertFalse(links['pdf'])
            self.assertEqual(links['paper'], 'https://doi.org/10.1234/test')

    def test_local_pdf_preferred_and_bad_header_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pdfs/paper.pdf'
            path.parent.mkdir()
            entry = self.entry(pdf_url='https://example.org/paper.pdf')
            path.write_bytes(b'<html>error</html>')
            self.assertEqual(publication_links(entry, directory)['pdf'], entry['pdf_url'])
            path.write_bytes(b'%PDF-1.7\n')
            self.assertEqual(publication_links(entry, directory)['pdf'], '/pdfs/paper.pdf')

    def test_abstract_and_summary_are_local_records_only(self):
        entry = self.entry(abstract='Recorded original abstract', summary='Recorded short summary',
                           summary_source='https://example.org/paper')
        with self.assertRaisesRegex(ValueError, 'source'):
            validate_entries([entry])
        entry['abstract_source'] = 'https://example.org/paper'
        validate_entries([entry])
        rendered = generate_html(entry)
        self.assertNotIn(entry['abstract'], rendered)
        self.assertNotIn(entry['summary'], rendered)
        self.assertNotIn('<details', rendered)
        self.assertEqual(entry['abstract'], 'Recorded original abstract')
        self.assertEqual(entry['summary'], 'Recorded short summary')

    def test_surgical_patch_preserves_comments_and_unrelated_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'citations.bib'
            source = ('@comment{@article{hidden, title={Hidden}}}\n'
                      '@article{paper,\n title = {A {Nested} Title},\n author={Cai, Yujun},\n'
                      ' year={2025}\n}\n')
            path.write_text(source, encoding='utf-8')
            update_fields(path, {'paper': {'abstract': 'An abstract with {braces}.',
                                           'abstract_source': 'https://example.org/paper'}})
            result = path.read_text(encoding='utf-8')
            self.assertIn('@comment{@article{hidden, title={Hidden}}}', result)
            self.assertIn('title = {A {Nested} Title}', result)
            update_fields(path, {'paper': {'status': 'published'}})
            self.assertEqual(result.count('abstract ='), 1)

    def test_manifest_reconciles_against_bib_and_keeps_existing_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'citations.bib').write_text('@article{paper, title={Paper}, author={Cai, Yujun}, year={2025}, journal={ICML}}')
            (root / 'pdfs').mkdir()
            (root / 'pdfs/paper_pdf_manifest.csv').write_text('citekey,title,status\nghost,Ghost,exists\npaper,Wrong,missing\npaper,Wrong,missing\n')
            pdf = root / 'pdfs/paper.pdf'
            pdf.write_bytes(b'%PDF-1.7\n' + b' ' * 1024)
            before = pdf.read_bytes()
            counts = build_catalog(root)
            with (root / 'pdfs/paper_pdf_manifest.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(r['citekey'], r['title'], r['status']) for r in rows], [('paper', 'Paper', 'exists')])
            self.assertEqual(counts['local_pdfs'], 1)
            self.assertEqual(pdf.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
