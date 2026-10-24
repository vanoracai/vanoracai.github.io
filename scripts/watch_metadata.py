"""Watch source metadata using only the Python standard library."""
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    paths = list((root / '_data').glob('*.json'))
    paths += list((root / '_data').glob('*.yml'))
    paths.append(root / 'citations.bib')
    paths += list((root / 'pdfs').glob('*.pdf'))
    result = {}
    for path in sorted(paths):
        try:
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).digest()
        except FileNotFoundError:
            result[str(path.relative_to(root))] = None
    return result


def cv_snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).digest()
            for path in sorted((root / 'Latex').glob('*/main.tex'))}


def watch(root=ROOT, regenerate=None, stop=None, interval=0.5, debounce=0.5,
          compile_saved=None, compile_pdfs=True):
    if regenerate is None:
        def regenerate():
            args = [sys.executable, '-u', str(root / 'scripts/generate_all.py')]
            if not compile_pdfs:
                args.append('--skip-cv')
            result = subprocess.run(args, cwd=root)
            if result.returncode:
                raise RuntimeError('Generation failed. Correct the source file and save again.')

    if compile_saved is None:
        def compile_saved():
            result = subprocess.run([sys.executable, '-u', str(root / 'scripts/compile_cv.py')], cwd=root)
            if result.returncode:
                raise RuntimeError('PDF compilation failed. Correct the saved source and retry.')

    def update(action):
        try:
            action()
            print('Content/PDF updated. Refresh your existing Jekyll preview.', flush=True)
        except Exception as error:
            print(f'Update error: {error}\nStill watching; save corrected source to retry.',
                  file=sys.stderr, flush=True)

    print('Watching metadata, citations.bib and saved CV main.tex files. Ctrl+C to stop.', flush=True)
    previous = snapshot(root)
    update(regenerate)
    previous_cv = cv_snapshot(root)
    pending = None
    pending_metadata = False
    pending_cv = False
    while stop is None or not stop.is_set():
        time.sleep(interval)
        try:
            current = snapshot(root)
            current_cv = cv_snapshot(root)
        except OSError as error:
            print(f'Cannot read metadata yet: {error}', file=sys.stderr, flush=True)
            continue
        if current != previous:
            previous = current
            pending_metadata = True
            pending = time.monotonic()
        if current_cv != previous_cv:
            previous_cv = current_cv
            pending_cv = True
            pending = time.monotonic()
        if pending is not None and time.monotonic() - pending >= debounce:
            pending = None
            if pending_metadata:
                update(regenerate)
            elif pending_cv and compile_pdfs:
                # Preserve manual changes in the editor: compile without regeneration.
                update(compile_saved)
            previous_cv = cv_snapshot(root)
            pending_metadata = pending_cv = False


if __name__ == '__main__':
    try:
        watch()
    except KeyboardInterrupt:
        print('\nMetadata watcher stopped.')
