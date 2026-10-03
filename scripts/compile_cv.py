"""Compile saved standalone CVs and publish PDFs only after successful compilation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {'en': '英文简历', 'zh': '中文简历'}


def cv_paths(root=ROOT):
    profile_path = root / '_data' / 'profile.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8-sig')) if profile_path.exists() else {}
    paths = {}
    for language, folder in FOLDERS.items():
        directory = root / 'Latex' / folder
        key = 'cv_link' if language == 'en' else 'cv_zh_link'
        configured = root / profile.get(key, '').lstrip('/')
        destination = (configured if configured.suffix == '.pdf'
                       and configured.parent.resolve() == directory.resolve() else directory / 'main.pdf')
        paths[language] = directory / 'main.tex', destination
    return paths


def find_compiler(root=ROOT, explicit=None):
    candidate = explicit or os.environ.get('TECTONIC_BIN') or shutil.which('tectonic')
    if candidate:
        path = Path(candidate).resolve()
        if not path.is_file():
            raise RuntimeError(f'Tectonic compiler does not exist: {path}')
        return path
    saved = root / '.venv' / 'tectonic-path.txt'
    if saved.is_file():
        path = Path(saved.read_text(encoding='utf-8').strip())
        if path.is_file():
            return path
    if sys.platform == 'win32':
        command = ("$taskPackage = Get-AppxPackage -Name OpenAI.Codex | Select-Object -First 1; "
                   "if ($taskPackage) { Join-Path $taskPackage.InstallLocation "
                   "'app\\resources\\tectonic\\tectonic.exe' }")
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=subprocess.CREATE_NO_WINDOW, check=True)
        path = Path(result.stdout.strip()) if result.stdout.strip() else None
        if path and path.is_file():
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(str(path), encoding='utf-8')
            return path
    raise RuntimeError('Tectonic is unavailable. Use the installed Codex desktop compiler on '
                       'Windows, put tectonic on PATH, or set TECTONIC_BIN to its executable.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_current(source, pdf, stamp):
    if not (source.is_file() and pdf.is_file() and stamp.is_file()):
        return False
    try:
        state = json.loads(stamp.read_text(encoding='utf-8'))
        return state == {'source': digest(source), 'pdf': digest(pdf)}
    except (OSError, ValueError):
        return False


def publish_pdf(built, destination):
    """Never truncate a live PDF; a failed replacement leaves the old file intact."""
    if not built.is_file() or not built.read_bytes().startswith(b'%PDF-'):
        raise RuntimeError(f'Compiler did not produce a PDF: {built}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.pending', delete=False) as file:
            temporary = Path(file.name)
        shutil.copyfile(built, temporary)
        os.replace(temporary, destination)
    except PermissionError as error:
        raise RuntimeError(f'Cannot update the open PDF {destination}. Close its desktop PDF tab '
                           'and retry; the previous PDF is preserved. Open the preview snapshot '
                           'printed by this compiler to avoid locking the website PDF.') from error
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def preview_copy(pdf, language, root=ROOT):
    preview = root / '.venv' / 'cv-previews' / f'cv-{language}-{digest(pdf)[:12]}.pdf'
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not preview.exists():
        shutil.copyfile(pdf, preview)
    return preview


def compile_cvs(languages=('en', 'zh'), root=ROOT, compiler=None, force=False, runner=subprocess.run):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    outputs = []
    executable = None
    for language in languages:
        source, destination = cv_paths(root)[language]
        if not source.is_file():
            raise RuntimeError(f'CV source is missing: {source}')
        build = root / '.venv' / f'cv-build-{language}'
        stamp = build / 'state.json'
        if force or not is_current(source, destination, stamp):
            source_digest = digest(source)
            executable = executable or find_compiler(root, compiler)
            build.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env['TECTONIC_CACHE_DIR'] = str(root / '.venv' / 'tectonic-cache')
            result = runner([str(executable), '-X', 'compile', '--untrusted', '--keep-logs',
                             '--outdir', str(build), str(source)], cwd=root, env=env,
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode:
                raise RuntimeError(f'{language} CV compilation failed:\n{result.stdout}\n{result.stderr}')
            log = build / 'main.log'
            if log.exists() and 'Missing character:' in log.read_text(encoding='utf-8', errors='replace'):
                raise RuntimeError(f'{language} CV has missing glyphs. Check {log}; previous PDF preserved.')
            if digest(source) != source_digest:
                raise RuntimeError(f'{source} changed during compilation. Retry with the saved source; '
                                   'previous PDF preserved.')
            built = build / 'main.pdf'
            try:
                publish_pdf(built, destination)
            except RuntimeError as error:
                if not isinstance(error.__cause__, PermissionError):
                    raise
                profile_path = root / '_data' / 'profile.json'
                if not profile_path.is_file():
                    raise
                # Keep the open file intact and point the website at a new version.
                destination = source.parent / f'main-{digest(built)[:12]}.pdf'
                if not destination.exists():
                    publish_pdf(built, destination)
                profile = json.loads(profile_path.read_text(encoding='utf-8-sig'))
                profile['cv_link' if language == 'en' else 'cv_zh_link'] = '/' + destination.relative_to(root).as_posix()
                profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                print(f'{language} website PDF link updated because the previous file is open.', flush=True)
            stamp.write_text(json.dumps({'source': source_digest, 'pdf': digest(destination)}), encoding='utf-8')
            print(f'{language} CV compiled: {destination}', flush=True)
        else:
            print(f'{language} CV is current: {destination}', flush=True)
        preview = preview_copy(destination, language, root)
        print(f'Preview snapshot: {preview}', flush=True)
        outputs.append(destination)
    return outputs


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=('en', 'zh', 'all'), default='all')
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    try:
        compile_cvs(('en', 'zh') if args.language == 'all' else (args.language,),
                    compiler=args.compiler, force=args.force)
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
