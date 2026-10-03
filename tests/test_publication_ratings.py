"""Verify real venue aliases and prevent conference/journal misclassification."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bib_to_html import generate_html, generate_bibliography_html
from bib_to_latex import bib_to_paper_list
from bibliography_common import abbreviate_venue, publication_ratings, is_arxiv


class PublicationRatingsTests(unittest.TestCase):
    def test_tpami_full_name_and_aliases_receive_both_labels(self):
        for venue in ('IEEE transactions on pattern analysis and machine intelligence',
                      'TPAMI', 'PAMI'):
            with self.subTest(venue=venue):
                self.assertEqual(publication_ratings({'journal': venue}), ['CCF-A', 'JCR Q1'])

    def test_acm_multimedia_full_name_and_aliases(self):
        for venue in ('Proceedings of the ACM International Conference on Multimedia (ACM MM)',
                      'ACMMM', 'acm mm', 'ACM Multimedia'):
            with self.subTest(venue=venue):
                self.assertEqual(publication_ratings({'booktitle': venue}), ['CCF-A'])

    def test_conferences_stored_as_articles_do_not_receive_jcr_labels(self):
        for venue in ('NeurIPS', 'CVPR', 'ACL', 'ICLR'):
            with self.subTest(venue=venue):
                self.assertEqual(publication_ratings({'journal': venue}), ['CCF-A'])

    def test_naacl_is_not_misread_as_acl(self):
        venue = 'Proceedings of the North American Chapter of the Association for Computational Linguistics'
        self.assertEqual(abbreviate_venue(venue), 'NAACL')
        self.assertEqual(publication_ratings({'booktitle': venue}), [])

    def test_unranked_and_companion_formats_do_not_receive_ccf_a(self):
        for venue in ('BMVC', 'CoRL', 'COLM', 'arXiv preprint arXiv:2601.12345',
                      'Findings of the Association for Computational Linguistics: ACL 2026',
                      'CVPR Workshops', 'Unlisted Venue', 'NotACL'):
            with self.subTest(venue=venue):
                self.assertEqual(publication_ratings({'booktitle': venue}), [])
        self.assertEqual(publication_ratings({'booktitle': 'CVPR', 'note': 'Workshop'}), [])

    def test_supplementary_icore_ratings(self):
        for venue in ('ECCV', 'European Conference on Computer Vision', 'eccv',
                      'EMNLP', 'Empirical Methods in Natural Language Processing',
                      'SIGGRAPH Asia 2022 conference papers'):
            with self.subTest(venue=venue):
                self.assertEqual(publication_ratings({'booktitle': venue}), ['ICORE A*'])
        self.assertEqual(publication_ratings({'booktitle': 'ECCV Workshops'}), [])
        self.assertEqual(publication_ratings({'booktitle': 'Findings of EMNLP'}), [])

    def test_bmc_bioinformatics_case_insensitive(self):
        self.assertEqual(publication_ratings({'journal': 'BMC bioinformatics'}), ['JCR Q1'])

    def test_arxiv_filter_keeps_published_papers_with_arxiv_links(self):
        published = {'ID': 'published', 'author': 'Cai, Yujun', 'title': 'Published paper',
                     'year': '2025', 'journal': 'NeurIPS', 'url': 'https://arxiv.org/abs/1234.5678'}
        preprint = dict(published, ID='preprint', title='Hidden preprint', year='2024',
                        journal='arXiv preprint arXiv:1234.5678')
        self.assertFalse(is_arxiv(published))
        self.assertTrue(is_arxiv(preprint))
        self.assertTrue(is_arxiv({'booktitle': 'arXiv'}))
        self.assertTrue(is_arxiv({'url': 'https://arxiv.org/abs/1234.5678'}))
        rendered = generate_bibliography_html([published, preprint])
        self.assertIn('Published paper', rendered)
        self.assertNotIn('Hidden preprint', rendered)
        self.assertNotIn('>2024</h3>', rendered)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'example.bib'
            source.write_text('@article{published, author={Cai, Yujun}, title={Published paper}, '
                              'journal={NeurIPS}, year={2025}, url={https://arxiv.org/abs/1234.5678}}\n'
                              '@article{preprint, author={Cai, Yujun}, title={Hidden preprint}, '
                              'journal={arXiv}, year={2024}}', encoding='utf-8')
            rendered = bib_to_paper_list(source)
            self.assertIn('Published paper', rendered)
            self.assertNotIn('Hidden preprint', rendered)

    def test_html_and_latex_export_same_labels(self):
        entry = {'ID': 'example', 'author': 'Cai, Yujun', 'year': '2026',
                 'title': 'Example', 'journal': 'TPAMI'}
        for has_image in (True, False):
            with self.subTest(has_image=has_image), patch('bib_to_html.os.path.exists', return_value=has_image):
                rendered = generate_html(entry)
                self.assertIn('<span class="publication-rating">(CCF-A)</span>', rendered)
                self.assertIn('<span class="publication-rating">(JCR Q1)</span>', rendered)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'example.bib'
            source.write_text('@article{example, author={Cai, Yujun}, title={Example}, '
                              'journal={TPAMI}, year={2026}}', encoding='utf-8')
            rendered = bib_to_paper_list(source)
            self.assertIn(r'\textcolor{red}{(CCF-A)}', rendered)
            self.assertIn(r'\textcolor{red}{(JCR Q1)}', rendered)


if __name__ == '__main__':
    unittest.main()
