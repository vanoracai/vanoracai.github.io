"""Check malformed data is caught before publication and bilingual rendering is preserved."""
import importlib.util
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_all', ROOT / 'scripts/generate_all.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class MetadataTests(unittest.TestCase):
    def test_current_sources_are_valid(self):
        *_, count = pipeline.validate()
        self.assertGreater(count, 0)

    def test_bilingual_section_uses_language_specific_fields(self):
        source = {'title': '教学', 'title_en': 'Teaching', 'sections': [
            {'title': '课程', 'title_en': 'Courses', 'items': [
                {'name': '机器学习', 'name_en': 'Machine Learning', 'details': '东南大学',
                 'details_en': 'Southeast University', 'date': '2026'}]}]}
        self.assertIn('东南大学', pipeline.section_md(source))
        english = pipeline.section_md(source, 'en')
        self.assertIn('Machine Learning', english)
        self.assertIn('Southeast University', english)
        self.assertNotIn('机器学习', english)

    def test_empty_sections_do_not_invent_entries(self):
        rendered = pipeline.section_md({'title': '荣誉', 'sections': [{'items': []}]})
        self.assertIn('待补充', rendered)
        self.assertNotIn('- **', rendered)

    def test_write_does_not_rewrite_unchanged_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'section.md'
            pipeline.write(target, 'Example')
            before = target.stat().st_mtime_ns
            pipeline.write(target, 'Example\n')
            self.assertEqual(before, target.stat().st_mtime_ns)

    def test_latex_special_characters_are_escaped(self):
        self.assertEqual(pipeline.tex_escape('A&B_1'), r'A\&B\_1')

    def test_english_cv_is_standalone_and_contains_no_untranslated_chinese(self):
        rendered = pipeline.english_cv_document(pipeline.read('profile'), pipeline.read('research'))
        self.assertIn(r'\documentclass[11pt,letterpaper]{article}', rendered)
        self.assertIn(r'\end{document}', rendered)
        self.assertNotRegex(rendered, r'\\(?:input|include)\s*\{')
        self.assertNotRegex(rendered, r'[\u4e00-\u9fff]')
        papers = (ROOT / 'paper_list.tex').read_text(encoding='utf-8')
        self.assertEqual(len(re.findall(r'\\item ', papers)), len(re.findall(r'\\item ', rendered)))

    def test_english_cv_combines_meta_roles_and_preserves_legacy_teaching(self):
        rendered = pipeline.english_cv_document(pipeline.read('profile'), pipeline.read('research'))
        self.assertNotIn('Senior Research Scientist at Meta Reality Labs', rendered)
        self.assertEqual(rendered.count('cvEntry{Research Scientist at Meta Reality Labs'), 1)
        self.assertIn('{2022 - 2024}{}', rendered)
        self.assertIn('{2017 - 2021}', rendered)
        self.assertIn('{2013 - 2017}', rendered)
        self.assertIn(r'Lecturer at The University of Queensland (QS \#42)', rendered)
        self.assertNotIn(r'\section*{Research Interests}', rendered)
        self.assertIn('Engineering Mathematics II', rendered)
        self.assertIn('Rank: 1/229', rendered)

    def test_homepage_allows_research_without_description(self):
        section = {'title': '工作经历', 'sections': [{'items': []}]}
        education = {'title': '教育背景', 'sections': [{'items': []}]}
        rendered = pipeline.homepage({'email': 'test@example.com'},
                                     {'items': [{'name': '海洋信号处理', 'details': ''}]},
                                     [], {'text': '欢迎联系'}, section, education)
        self.assertIn('海洋信号处理', rendered)
        self.assertNotIn('  - \n', rendered)
        self.assertNotIn('最新动态', rendered)


if __name__ == '__main__':
    unittest.main()
