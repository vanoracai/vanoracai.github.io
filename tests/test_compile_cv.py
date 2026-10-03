"""Protect published PDFs against compilation failures and stale cached output."""
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import compile_cv


class CompileCVTests(unittest.TestCase):
    def test_incremental_compile_and_saved_source_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, pdf = compile_cv.cv_paths(root)['en']
            source.parent.mkdir(parents=True)
            source.write_text('first source', encoding='utf-8')
            compiler = root / 'tectonic'
            compiler.touch()
            runs = []

            def run(args, **kwargs):
                runs.append(args)
                output = Path(args[args.index('--outdir') + 1])
                (output / 'main.pdf').write_bytes(b'%PDF-1.5\n' + source.read_bytes())
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=run)
            compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=run)
            self.assertEqual(len(runs), 1)
            source.write_text('edited source', encoding='utf-8')
            compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=run)
            self.assertEqual(len(runs), 2)
            self.assertIn(b'edited source', pdf.read_bytes())
            self.assertEqual(len(list((root / '.venv/cv-previews').glob('*.pdf'))), 2)
            pdf.write_bytes(b'%PDF-1.5\nobsolete PDF')
            compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=run)
            self.assertEqual(len(runs), 3)

    def test_compiler_failure_does_not_replace_existing_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, pdf = compile_cv.cv_paths(root)['en']
            source.parent.mkdir(parents=True)
            source.write_text('invalid source', encoding='utf-8')
            pdf.write_bytes(b'%PDF-1.5\nlast good PDF')
            compiler = root / 'tectonic'
            compiler.touch()
            failed = lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout='', stderr='TeX error')
            with self.assertRaisesRegex(RuntimeError, 'compilation failed'):
                compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=failed)
            self.assertEqual(pdf.read_bytes(), b'%PDF-1.5\nlast good PDF')

    def test_missing_glyphs_and_edits_during_compile_preserve_old_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, pdf = compile_cv.cv_paths(root)['en']
            source.parent.mkdir(parents=True)
            source.write_text('source', encoding='utf-8')
            pdf.write_bytes(b'%PDF-1.5\nlast good PDF')
            compiler = root / 'tectonic'
            compiler.touch()

            def run(args, **kwargs):
                output = Path(args[args.index('--outdir') + 1])
                (output / 'main.pdf').write_bytes(b'%PDF-1.5\nbad PDF')
                (output / 'main.log').write_text('Missing character: unsupported glyph', encoding='utf-8')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with self.assertRaisesRegex(RuntimeError, 'missing glyphs'):
                compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=run)
            self.assertEqual(pdf.read_bytes(), b'%PDF-1.5\nlast good PDF')

            def edit_during_run(args, **kwargs):
                output = Path(args[args.index('--outdir') + 1])
                (output / 'main.pdf').write_bytes(b'%PDF-1.5\nstale PDF')
                (output / 'main.log').write_text('', encoding='utf-8')
                source.write_text('new save', encoding='utf-8')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            with self.assertRaisesRegex(RuntimeError, 'changed during compilation'):
                compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=edit_during_run)
            self.assertEqual(pdf.read_bytes(), b'%PDF-1.5\nlast good PDF')

    def test_locked_pdf_is_preserved_and_temporary_file_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            built, destination = root / 'built.pdf', root / 'live.pdf'
            built.write_bytes(b'%PDF-1.5\nnew')
            destination.write_bytes(b'%PDF-1.5\nold')
            with patch.object(compile_cv.os, 'replace', side_effect=PermissionError('open PDF')):
                with self.assertRaisesRegex(RuntimeError, 'previous PDF is preserved'):
                    compile_cv.publish_pdf(built, destination)
            self.assertEqual(destination.read_bytes(), b'%PDF-1.5\nold')
            self.assertEqual(list(root.glob('*.pending')), [])

    def test_locked_website_pdf_publishes_a_new_version_and_updates_its_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / '_data/profile.json'
            profile_path.parent.mkdir()
            profile_path.write_text(json.dumps({'cv_link': '/Latex/英文简历/main.pdf'}), encoding='utf-8')
            source, old_pdf = compile_cv.cv_paths(root)['en']
            source.parent.mkdir(parents=True)
            source.write_text('updated source', encoding='utf-8')
            old_pdf.write_bytes(b'%PDF-1.5\nold version')
            compiler = root / 'tectonic'
            compiler.touch()

            def run(args, **kwargs):
                output = Path(args[args.index('--outdir') + 1])
                (output / 'main.pdf').write_bytes(b'%PDF-1.5\nnew version')
                return SimpleNamespace(returncode=0, stdout='', stderr='')

            replace = compile_cv.os.replace
            def locked_replace(temporary, destination):
                if destination == old_pdf:
                    raise PermissionError('open website PDF')
                return replace(temporary, destination)

            with patch.object(compile_cv.os, 'replace', side_effect=locked_replace):
                outputs = compile_cv.compile_cvs(('en',), root=root, compiler=compiler, runner=run)
            self.assertEqual(old_pdf.read_bytes(), b'%PDF-1.5\nold version')
            self.assertEqual(outputs[0].read_bytes(), b'%PDF-1.5\nnew version')
            linked = json.loads(profile_path.read_text(encoding='utf-8'))['cv_link']
            self.assertEqual(root / linked.lstrip('/'), outputs[0])
            self.assertEqual(compile_cv.cv_paths(root)['en'][1], outputs[0])


if __name__ == '__main__':
    unittest.main()
