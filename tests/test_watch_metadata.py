"""Verify save detection, invalid-data recovery and no output feedback loop."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('watch_metadata', ROOT / 'scripts/watch_metadata.py')
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class WatchTests(unittest.TestCase):
    def test_manual_tex_save_compiles_without_regenerating_or_watching_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '_data').mkdir()
            source = root / 'Latex/en/main.tex'
            source.parent.mkdir(parents=True)
            source.write_text('generated source', encoding='utf-8')
            generations, compilations = [], []
            stop = threading.Event()

            def generate():
                generations.append(1)
                source.write_text('generated source', encoding='utf-8')

            def compile_saved():
                compilations.append(source.read_text(encoding='utf-8'))
                (source.parent / 'main.pdf').write_bytes(b'%PDF-1.5\nnew output')

            def until(condition):
                deadline = time.monotonic() + 3
                while not condition() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(condition())

            thread = threading.Thread(target=watcher.watch, kwargs={
                'root': root, 'regenerate': generate, 'compile_saved': compile_saved,
                'stop': stop, 'interval': 0.01, 'debounce': 0.04}, daemon=True)
            thread.start()
            try:
                until(lambda: generations and watcher.cv_snapshot(root))
                # Let startup finish taking its baseline before the manual save.
                time.sleep(0.05)
                source.write_text('manual editor change', encoding='utf-8')
                until(lambda: compilations)
                self.assertEqual(compilations, ['manual editor change'])
                self.assertEqual(len(generations), 1)
                before = watcher.cv_snapshot(root)
                (source.parent / 'main.pdf').write_bytes(b'%PDF-1.5\nchanged PDF')
                self.assertEqual(before, watcher.cv_snapshot(root))
            finally:
                stop.set()
                thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    def test_save_error_recovery_and_output_is_not_watched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '_data').mkdir()
            source = root / '_data/profile.json'
            source.write_text('{"name": "first"}', encoding='utf-8')
            output = root / 'generated.md'
            attempts = []
            stop = threading.Event()

            def generate():
                attempts.append(1)
                data = json.loads(source.read_text(encoding='utf-8'))
                output.write_text(data['name'], encoding='utf-8')

            def until(condition):
                deadline = time.monotonic() + 3
                while not condition() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(condition())

            thread = threading.Thread(target=watcher.watch, kwargs={
                'root': root, 'regenerate': generate, 'stop': stop,
                'interval': 0.01, 'debounce': 0.04}, daemon=True)
            thread.start()
            try:
                until(lambda: output.exists())
                source.write_text('{"name": "second"}', encoding='utf-8')
                until(lambda: output.read_text(encoding='utf-8') == 'second')
                source.write_text('{broken', encoding='utf-8')
                until(lambda: len(attempts) == 3)
                self.assertEqual(output.read_text(encoding='utf-8'), 'second')
                source.write_text('{"name": "fixed"}', encoding='utf-8')
                until(lambda: output.read_text(encoding='utf-8') == 'fixed')
                before = watcher.snapshot(root)
                output.write_text('changed output', encoding='utf-8')
                self.assertEqual(before, watcher.snapshot(root))
            finally:
                stop.set()
                thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
