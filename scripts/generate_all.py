"""Validate shared metadata and generate website content and CV fragments."""

from __future__ import annotations

import argparse
from datetime import date
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '_data'
OUTPUT = ROOT / '_includes' / 'generated'
CV = ROOT / '_generated'
sys.path.insert(0, str(ROOT))

SECTIONS = ('students', 'teaching', 'services', 'awards', 'grants', 'work', 'education')


def read(name):
    try:
        return json.loads((DATA / f'{name}.json').read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError as error:
        raise ValueError(f'{name}.json, line {error.lineno}: {error.msg}. '
                         'Check quotes and commas; JSON does not support // comments.') from error


def field(item, key, language='zh'):
    return item.get(f'{key}_en', item.get(key, '')) if language == 'en' else item.get(key, '')


def label(name, url=''):
    return f'[{name}]({url})' if url else name


def item_md(item, language='zh', bold_names=True):
    name = label(field(item, 'name', language), item.get('url', ''))
    details = field(item, 'details', language)
    when = field(item, 'date', language)
    suffix = ' · '.join(part for part in (details, when) if part)
    return (f'- **{name}**' if bold_names else f'- {name}') + (f' — {suffix}' if suffix else '')


def section_md(data, language='zh', bold_names=True):
    lines = [f'## {field(data, "title", language)}', '']
    count = 0
    for group in data['sections']:
        if not group['items']:
            continue
        heading = field(group, 'title', language)
        if heading:
            lines.extend([f'### {heading}', ''])
        lines.extend(item_md(item, language, bold_names) for item in group['items'])
        lines.append('')
        count += len(group['items'])
    if not count:
        lines.append('本页内容待补充。' if language == 'zh' else 'Details will be added soon.')
    return '\n'.join(lines).rstrip() + '\n'


def teaching_md(data):
    """Use the reference's two sections with unbolded linked course names."""
    tutorials = []
    courses = []
    for group in data['sections']:
        tutorial = group.get('title_en') == 'Tutorial in Academic Conferences'
        for item in group['items']:
            name = field(item, 'name', 'en')
            if group.get('title_en') == 'Teaching Assistant':
                name = 'Teaching Assistant, ' + name
            suffix = ', '.join(filter(None, (field(item, 'details', 'en'), field(item, 'date', 'en'))))
            line = '- ' + label(name, item.get('url', '')) + (f' — {suffix}' if suffix else '')
            (tutorials if tutorial else courses).append(line)
    return '\n'.join(['## Tutorial in Academic Conferences', '', *tutorials,
                      '', '## Teaching', '', *courses, ''])


def services_md(data):
    cards = []
    for group in data['sections']:
        if not group['items']:
            continue
        lines = ['<div class="service-card">',
                 f'<h3>{html.escape(field(group, "title", "en"))}</h3>', '<ul>']
        for item in group['items']:
            text = field(item, 'name', 'en')
            suffix = ', '.join(filter(None, (field(item, 'details', 'en'), field(item, 'date', 'en'))))
            if suffix:
                text += ' — ' + suffix
            lines.append(f'<li>{html.escape(text)}</li>')
        lines.extend(['</ul>', '</div>'])
        cards.append('\n'.join(lines))
    return '\n\n'.join(cards) or 'Details will be added soon.'


def validate():
    profile = read('profile')
    for key in ('name', 'position', 'affiliation', 'email', 'avatar', 'school_logo'):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f'profile.json: {key} must be a nonempty string')
    from scripts.compile_cv import cv_paths
    generated_pdfs = {pdf.resolve() for _, pdf in cv_paths().values()}
    for key in ('avatar', 'school_logo', 'cv_link', 'cv_zh_link'):
        path = profile.get(key, '')
        if (path and not re.match(r'https?://', path)
                and not (ROOT / path.lstrip('/')).is_file()
                and (ROOT / path.lstrip('/')).resolve() not in generated_pdfs):
            raise ValueError(f'profile.json: {key} points to missing file {path}')
    for language in ('zh', 'en'):
        if not isinstance(profile.get(f'bio_{language}', []), list):
            raise ValueError(f'profile.json: bio_{language} must be a list')
    research = read('research')
    if not isinstance(research.get('items'), list):
        raise ValueError('research.json: items must be a list')
    for item in research['items']:
        if not isinstance(item.get('name'), str) or not item['name'].strip():
            raise ValueError('research.json: every item needs a name')
        if not isinstance(item.get('details', ''), str):
            raise ValueError('research.json: details must be text, or left empty')
    for name in SECTIONS:
        data = read(name)
        if not data.get('title') or not isinstance(data.get('sections'), list):
            raise ValueError(f'{name}.json: title and sections are required')
        for group in data['sections']:
            if not isinstance(group.get('items'), list):
                raise ValueError(f'{name}.json: each section needs an items list')
            for item in group['items']:
                if not isinstance(item.get('name'), str) or not item['name'].strip():
                    raise ValueError(f'{name}.json: each item needs a name')
    news = read('news')
    if not isinstance(news, list):
        raise ValueError('news.json must be a list')
    for item in news:
        date.fromisoformat(item['date'] + '-01')
        if not item.get('text'):
            raise ValueError('news.json: each item needs text')
    recruitment = read('recruitment')
    if not recruitment.get('text'):
        raise ValueError('recruitment.json: text is required')
    import bibtexparser
    with (ROOT / 'citations.bib').open(encoding='utf-8') as handle:
        entries = bibtexparser.load(handle).entries
    ids = set()
    if not entries:
        raise ValueError('citations.bib contains no publications')
    for entry in entries:
        if entry['ID'] in ids:
            raise ValueError(f'Duplicate bibliography key: {entry["ID"]}')
        ids.add(entry['ID'])
        if not entry.get('title') or not entry.get('author') or not entry.get('year', '').isdigit():
            raise ValueError(f'Publication {entry["ID"]} needs title, author, and numeric year')
    return profile, research, news, recruitment, len(entries)


def write(path, content):
    # Avoid unnecessary watcher reloads when the rendered output is unchanged.
    content = content.rstrip() + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8')


def homepage(profile, research, news, recruitment, work, education):
    def linked_name(item):
        name = html.escape(item['name'])
        if item.get('url'):
            return f'<a href="{html.escape(item["url"], quote=True)}">{name}</a>'
        return name

    def experience(data):
        lines = ['<div class="home-section">', f'<h2>{html.escape(data["title"])}</h2>']
        for group in data['sections']:
            if group.get('title'):
                lines.append(f'<h3>{html.escape(group["title"])}</h3>')
            lines.append('<ul class="experience-list">')
            for item in group['items']:
                when = item.get('date', '')
                period = re.fullmatch(r'(\d{4})\s*[-–]\s*(\d{4}|至今|现在)', when)
                if period:
                    end = period[2] + ('年' if period[2].isdigit() else '')
                    when = f'{period[1]}年 - {end}'
                lines.extend(['<li class="experience-row">', '<div class="experience-main">',
                              f'<span class="experience-name">{linked_name(item)}</span>'])
                if item.get('details'):
                    lines[-1] += f'，<span class="experience-detail">{html.escape(item["details"])}</span>'
                if when:
                    lines[-1] += f' — <span class="experience-date">{html.escape(when)}</span>'
                lines.extend(['</div>', '</li>'])
            lines.append('</ul>')
        lines.append('</div>')
        return '\n'.join(lines)

    separator = '<span class="research-divider" aria-hidden="true">／</span>'
    topics = separator.join(f'<span class="research-name">{linked_name(item)}</span>'
                            for item in research['items'])
    lines = ['<div class="home-section">', '<h2>研究领域</h2>',
             f'<p class="research-topics">{topics}</p>']
    for item in research['items']:
        if item.get('details'):
            lines.append(f'<p class="research-description">{linked_name(item)}：{html.escape(item["details"])}</p>')
    lines.extend(['<div class="home-recruitment" markdown="1">', '',
                  '<span class="recruitment-label">🔥 招生与科研合作：</span>' + recruitment['text'], '', '</div>',
                  '</div>', experience(work), experience(education)])
    if profile.get('address'):
        lines.extend(['<div class="home-section">', '<h2>联系方式</h2>',
                      f'<p>通讯地址：{html.escape(profile["address"])}</p>', '</div>'])
    return '\n'.join(lines)


def plain(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text.replace('**', ''))


def tex_escape(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in plain(text))


def english_cv_year_range(text):
    dates = re.findall(r'(?<!\d)\d{4}(?!\d)|\bPresent\b', text, flags=re.IGNORECASE)
    return ' - '.join('Present' if value.lower() == 'present' else value for value in dates)


def cv_fragments(profile, research, recruitment):
    for language in ('zh', 'en'):
        title = field(profile, 'name', language)
        affiliation = field(profile, 'affiliation', language)
        position = field(profile, 'position', language)
        summary = f'{position}, {affiliation}'
        lines = [f'# {title}', '', summary, '', profile['email'], '',
                 *sum(([p, ''] for p in profile.get(f'bio_{language}', [])), []),
                 section_md({'title': '研究方向', 'title_en': 'Research Interests',
                             'sections': [{'title': '', 'items': research['items']}]}, language)]
        tex = [f'\\section*{{{tex_escape(title)}}}', tex_escape(summary), '',
               tex_escape(profile['email']), '']
        tex.extend(tex_escape(paragraph) + '\n' for paragraph in profile.get(f'bio_{language}', []))
        tex.extend([r'\section*{' + ('研究方向' if language == 'zh' else 'Research Interests') + '}', r'\begin{itemize}'])
        for item in research['items']:
            tex.append(r'\item ' + tex_escape(field(item, 'name', language) + ': ' + field(item, 'details', language)))
        tex.append(r'\end{itemize}')
        for name in ('work', 'education', 'services', 'teaching', 'awards', 'grants', 'students'):
            data = read(name)
            if language == 'en' and name in ('work', 'education'):
                for group in data['sections']:
                    for item in group['items']:
                        item['date_en'] = english_cv_year_range(item.get('cv_date_en') or field(item, 'date', 'en'))
            lines.append(section_md(data, language))
            items = [item for group in data['sections'] for item in group['items']]
            if not items:
                continue
            tex.extend([f'\\section*{{{tex_escape(field(data, "title", language))}}}', r'\begin{itemize}'])
            for item in items:
                text = ' — '.join(filter(None, [field(item, 'name', language),
                                               field(item, 'details', language), field(item, 'date', language)]))
                tex.append(r'\item ' + tex_escape(text))
            tex.append(r'\end{itemize}')
        # The same canonical BibTeX file also supplies the CV publication list.
        import bibtexparser
        from bibliography_common import publication_ratings, is_arxiv
        with (ROOT / 'citations.bib').open(encoding='utf-8') as handle:
            entries = bibtexparser.load(handle).entries
        lines.extend(['## ' + ('论文发表' if language == 'zh' else 'Publications'), ''])
        for entry in sorted(entries, key=lambda item: int(item['year']), reverse=True):
            if is_arxiv(entry):
                continue
            labels = ' '.join(f'({label})' for label in publication_ratings(entry))
            lines.append(f'- {entry["author"]}. {entry["title"]}. {entry.get("booktitle", entry.get("journal", ""))}, {entry["year"]}.' +
                         (' ' + labels if labels else ''))
        tex.extend([r'\section*{' + ('论文发表' if language == 'zh' else 'Publications') + '}',
                    (ROOT / 'paper_list.tex').read_text(encoding='utf-8')])
        write(CV / f'cv_{language}.md', '\n'.join(lines))
        write(CV / f'cv_{language}.tex', '\n'.join(tex))


def chinese_cv_document(profile, research):
    """Standalone Chinese CV using Yiwei Wang's ruled-section ctex layout."""
    lines = [r'''% 自动生成：修改 _data/*.json 或 citations.bib 后运行 .\generate.ps1。
% 版式参考 reference-wangywust/Latex/中文简历/main.tex。
\documentclass[UTF8,a4paper,11pt,fontset=fandol]{ctexart}
\usepackage[left=1.5cm,right=1.5cm,top=2cm,bottom=2cm]{geometry}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{tabularx}
\setCJKmainfont{FandolKai-Regular.otf}[BoldFont=FandolKai-Regular.otf]
\setmainfont{texgyretermes-regular.otf}[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]
\hypersetup{hidelinks,pdftitle={蔡雨君中文简历},pdfauthor={Yujun Cai}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyfoot[C]{\small\thepage}
\titleformat{\section}{\large\normalfont}{}{0em}{}[\titlerule]
\titlespacing*{\section}{0pt}{16pt}{5pt}
\newcommand{\rSection}[1]{\section*{#1}}
\setlist[itemize]{leftmargin=1.5em,noitemsep,topsep=2pt}
\setlist[itemize,1]{label=\textbullet}
\newlist{experience}{itemize}{1}
\setlist[experience]{label=\textbullet,leftmargin=1.6em,itemsep=3pt,topsep=0pt,parsep=0pt}
\newcommand{\cvDateRange}[2]{\makebox[4.4em][l]{#1}\makebox[1.4em][c]{-}\makebox[4.4em][l]{#2}}
\newcommand{\cvExperience}[3]{\item\begin{tabularx}{\linewidth}[t]{@{}Xr@{}}#1 & #2\end{tabularx}\if\relax\detokenize{#3}\relax\else\\[-1pt]#3\fi}
\setlist[enumerate]{leftmargin=2.3em,label={[\arabic*]},itemsep=5pt,topsep=3pt}
\emergencystretch=3em
\begin{document}
\begin{center}''']
    name = profile['name']
    chinese = re.search(r'[（(]([^）)]+)[）)]', name)
    if chinese:
        name = chinese.group(1)
    lines.extend([r'{\LARGE\bfseries ' + tex_escape(name) + r'}\\[6pt]',
                  r'\href{mailto:' + profile['email'] + '}{' + tex_escape(profile['email']) + r'}',
                  r'\quad\href{https://vanoracai.github.io/}{个人主页}', r'\end{center}'])
    for name in ('work', 'education'):
        data = read(name)
        lines.append(r'\rSection{' + tex_escape(data['title']) + '}')
        lines.append(r'\begin{experience}')
        for group in data['sections']:
            for item in group['items']:
                description = '，'.join(filter(None, [item['name'], item.get('details', '')]))
                date_text = item.get('date', '')
                date_range = re.fullmatch(r'(\d{4})\s*[-–]\s*(\d{4}|至今)', date_text)
                if date_range:
                    start, end = date_range.groups()
                    when = r'\cvDateRange{' + start + ' 年}{' + end + (' 年' if end.isdigit() else '') + '}'
                else:
                    when = tex_escape(date_text)
                department = item.get('department', '')
                lines.append(r'\cvExperience{' + tex_escape(description) + '}{' + when +
                             '}{' + tex_escape(department) + '}')
        lines.append(r'\end{experience}')
    lines.append(r'\rSection{研究方向}')
    lines.append(tex_escape('、'.join(item['name'] for item in research['items'])))
    for item in research['items']:
        if item.get('details'):
            lines.append(tex_escape(item['name'] + '：' + item['details']) + r'\par')
    lines.append(r'\rSection{研究论文发表}')
    papers = (ROOT / 'paper_list.tex').read_text(encoding='utf-8')
    papers = papers.replace('Yujun Cai', r'\textbf{Yujun Cai}')
    lines.append(papers)
    services = read('services')
    for group in services['sections']:
        if not group['items']:
            continue
        lines.extend([r'\rSection{' + tex_escape(group.get('title') or services['title']) + '}',
                      r'\begin{itemize}'])
        for item in group['items']:
            lines.append(r'\item ' + tex_escape(' — '.join(filter(None, [item['name'].strip(),
                               item.get('details', ''), item.get('date', '')]))))
        lines.append(r'\end{itemize}')
    for name in ('teaching', 'awards'):
        data = read(name)
        for group in data['sections']:
            if not group['items']:
                continue
            heading = group.get('title') or data['title']
            if heading == '学术会议中主办Tutorial':
                heading = '学术会议教程'
            lines.extend([r'\rSection{' + tex_escape(heading) + '}', r'\begin{itemize}'])
            for item in group['items']:
                display_name = item['name']
                course = re.fullmatch(r'([A-Z]+\d+(?:/\d+)?)\s+(.+)', display_name)
                if name == 'teaching' and course:
                    display_name = course[2] + '（' + course[1] + '）'
                when = item.get('date', '')
                if name == 'awards' and re.fullmatch(r'\d{4}(?:,\s*\d{4})*', when):
                    when = when.replace(', ', '、') + ' 年'
                text = '，'.join(filter(None, [display_name, item.get('details', ''), when]))
                lines.append(r'\item ' + tex_escape(text))
            lines.append(r'\end{itemize}')
    lines.append(r'\end{document}')
    return '\n'.join(lines)


def english_cv_document(profile, research):
    """Standalone version of the reference resume.cls layout, with shared data."""
    lines = [r'''% Generated from _data/*.json and citations.bib by .\generate.ps1.
% Layout adapted from Yiwei Wang's English CV and the user's earlier CV.
% Resume template copyright (C) 2010 Trey Hunner; modified by LaTeXTemplates.com.
% Copying and distribution, with or without modification, are permitted in any
% medium without royalty provided the copyright notice and this notice remain.
% This template is offered as-is, without any warranty.
% The layout is embedded so the built-in editor needs no resume.cls or inputs.
\documentclass[11pt,letterpaper]{article}
\usepackage[left=0.7in,right=0.7in,top=1.2in,bottom=1.2in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tabularx}
\usepackage{needspace}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{hidelinks,pdftitle={Yujun Cai -- Curriculum Vitae},pdfauthor={Yujun Cai}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\titleformat{\section}{\normalsize\bfseries}{}{0pt}{\MakeUppercase}[\vspace{4pt}\titlerule]
\titlespacing*{\section}{0pt}{12pt}{7pt}
\newcommand{\cvEntry}[3]{\Needspace{4\baselineskip}\par
  \begin{tabularx}{\linewidth}{@{}Xr@{}}\textbf{#1} & #2\end{tabularx}\nopagebreak
  \if\relax\detokenize{#3}\relax\else\par\nopagebreak #3\fi\par}
\newcommand{\cvService}[2]{\par\begin{tabularx}{\linewidth}{@{}Xr@{}}#1 & #2\end{tabularx}\par}
\setlist[enumerate]{leftmargin=2.1em,label={[\arabic*]},itemsep=5pt,parsep=0pt,topsep=3pt,before=\interlinepenalty10000}
\setlist[itemize]{leftmargin=1.3em,itemsep=3pt,parsep=0pt,topsep=3pt}
\emergencystretch=3em
\begin{document}
\begin{center}''']
    lines.append(r'{\huge\bfseries ' + tex_escape(profile['name_en'].upper()) + r'}\\[10pt]')
    lines.append(r'Email: \href{mailto:' + profile['email'] + '}{' + tex_escape(profile['email']) +
                 r'}\quad$\diamond$\quad\href{https://vanoracai.github.io/}{webpage}\\[4pt]')
    links = [(profile.get('google_scholar'), 'Google Scholar'), (profile.get('linkedin'), 'LinkedIn')]
    lines.append(r'\quad$\diamond$\quad'.join(r'\href{' + url + '}{' + title + '}'
                                            for url, title in links if url))
    lines.append(r'\end{center}')
    for name in ('work', 'education'):
        data = read(name)
        lines.append(r'\section*{' + tex_escape(field(data, 'title', 'en')) + '}')
        for group in data['sections']:
            for item in group['items']:
                institution = field(item, 'name', 'en').strip()
                role = field(item, 'details', 'en').strip()
                if name == 'work':
                    title = item.get('cv_title_en') or role + ' at ' + institution
                    detail = ''
                else:
                    title = role
                    detail = institution
                    if item.get('note_en'):
                        detail += ', ' + item['note_en']
                lines.append(r'\cvEntry{' + tex_escape(title) + '}{' +
                             tex_escape(english_cv_year_range(item.get('cv_date_en') or field(item, 'date', 'en'))) +
                             '}{' + tex_escape(detail) + '}')
    lines.extend([r'\section*{Publications}',
                  (ROOT / 'paper_list.tex').read_text(encoding='utf-8').replace(
                      'Yujun Cai', r'\textbf{Yujun Cai}')])
    services = read('services')
    for group in services['sections']:
        if not group['items']:
            continue
        lines.append(r'\section*{' + tex_escape(group.get('cv_title_en') or field(group, 'title', 'en')) + '}')
        for item in group['items'] + group.get('cv_extra_items_en', []):
            title = field(item, 'name', 'en').strip().replace('（', '(').replace('）', ')')
            detail = field(item, 'details', 'en')
            if detail:
                title += ' -- ' + detail
            lines.append(r'\cvService{' + tex_escape(title) + '}{' + tex_escape(field(item, 'date', 'en')) + '}')
    teaching = read('teaching')
    tutorials = [group for group in teaching['sections'] if group.get('title_en') == 'Tutorial in Academic Conferences']
    courses = [group for group in teaching['sections'] if group not in tutorials]
    for heading, groups in [('Conference Tutorials', tutorials), ('Teaching', courses)]:
        if not any(group['items'] for group in groups):
            continue
        lines.append(r'\section*{' + heading + '}')
        for group in groups:
            role = 'Teaching Assistant' if group.get('title_en') == 'Teaching Assistant' else 'Instructor'
            for item in group['items'] + group.get('cv_extra_items_en', []):
                title = tex_escape(field(item, 'name', 'en'))
                if item.get('url'):
                    title = r'\href{' + item['url'] + '}{' + title + '}'
                if heading == 'Teaching':
                    title = role + ', ' + title
                when = ', '.join(filter(None, [field(item, 'details', 'en'), field(item, 'date', 'en')]))
                # Tutorials and long course names keep dates on a separate line.
                lines.append(r'\Needspace{3\baselineskip}' + title + r'\par\nopagebreak')
                lines.append(tex_escape(when) + r'\par')
    awards = read('awards')
    if any(group['items'] for group in awards['sections']):
        lines.append(r'\section*{Honors and Awards}')
        for group in awards['sections']:
            for item in group['items']:
                title = field(item, 'name', 'en')
                if field(item, 'details', 'en'):
                    title += ', ' + field(item, 'details', 'en')
                lines.append(r'\cvService{' + tex_escape(title) + '}{' + tex_escape(field(item, 'date', 'en')) + '}')
    lines.append(r'\end{document}')
    # T1 resume fonts need plain punctuation for metadata copied from webpages.
    return '\n'.join(lines).translate(str.maketrans({'–': '-', '—': '-', '’': "'", '‘': "'",
                                                  '“': '``', '”': "''"}))


def generate():
    profile, research, news, recruitment, paper_count = validate()
    from bib_to_html import convert_bib_to_html
    from bib_to_latex import bib_to_paper_list
    convert_bib_to_html(ROOT / 'citations.bib', ROOT / '_includes' / 'output_file.md', profile)
    write(ROOT / 'paper_list.tex', bib_to_paper_list(ROOT / 'citations.bib'))
    banner = '<!-- 自动生成：修改 _data/*.json 后运行 python scripts/generate_all.py。 -->\n\n'
    write(OUTPUT / 'homepage.md', banner + homepage(profile, research, news, recruitment,
                                                 read('work'), read('education')))
    for name in SECTIONS:
        language = 'en' if name in ('students', 'teaching', 'services', 'awards', 'grants') else 'zh'
        content = teaching_md(read(name)) if name == 'teaching' else section_md(read(name), language)
        if name == 'services':
            content = services_md(read(name))
        write(OUTPUT / f'{name}.md', banner + content)
    cv_fragments(profile, research, recruitment)
    write(ROOT / 'Latex' / '中文简历' / 'main.tex', chinese_cv_document(profile, research))
    write(ROOT / 'Latex' / '英文简历' / 'main.tex', english_cv_document(profile, research))
    print(f'Generated seven section fragments, homepage, publications ({paper_count}), and bilingual CV sources.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='Validate sources without changing generated files')
    group.add_argument('--build', action='store_true', help='Generate content and build Jekyll without opening a preview')
    group.add_argument('--watch', action='store_true', help='Generate on metadata changes until Ctrl+C')
    parser.add_argument('--skip-cv', action='store_true', help='Generate text only, without compiling CV PDFs')
    args = parser.parse_args()
    if args.watch:
        from watch_metadata import watch
        try:
            watch(compile_pdfs=not args.skip_cv)
        except KeyboardInterrupt:
            print('\nMetadata watcher stopped.')
        return
    if args.check:
        *_, count = validate()
        print(f'Metadata and {count} publications are valid.')
        return
    generate()
    if not args.skip_cv:
        from scripts.compile_cv import compile_cvs
        compile_cvs()
    if args.build:
        bundle = shutil.which('bundle')
        if not bundle:
            raise SystemExit('Bundler is unavailable. Install Ruby and run bundle install first.')
        subprocess.run([bundle, 'exec', 'jekyll', 'build'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
