from bibliography_common import CONFERENCE_TIME_ORDER, abbreviate_venue, publication_ratings, is_visible, publication_links, publication_status, publication_sort_key
import html
import os
import bibtexparser
from bibtexparser.bparser import BibTexParser

# Dictionary to map venue keywords to abbreviations (CCF Recommended Conferences and more specific examples)

# Function to convert 'Last, First' to 'First Last'
def convert_author_format(author_name):
    if ',' in author_name:
        parts = author_name.split(',')
        return f"{parts[1].strip()} {parts[0].strip()}"
    return author_name  # If no comma, return as is

# Function to clean venue for full name (without abbreviating)
def clean_full_venue(venue):
    # Remove special characters like backslashes and replace '&' with 'and'
    return venue.replace("\\", "").replace("&", "and")

# Function to clean and remove special characters from title, including replacing \textquotesingle and \ast
def clean_title(title):
    return title.replace("{", "").replace("}", "").replace("\\textquotesingle", "\u0027").replace("\\ast", "*")


# Function to clean LaTeX-related artifacts in URLs, especially for escaping sequences like \_
def clean_url_latex_artifacts(url):
    # 将 \_ 替换为普通的 _
    return url.replace("\\_", "_")

# Function to generate image path based on the bib entry
def generate_image_filename(entry):
    # Get the entry ID, which is typically the bibtex citation key
    entry_id = entry.get('ID', 'default')
    return f"Image/{entry_id}.png"

# Function to check if a paper is a Preprint or ArXiv
def is_preprint(entry):
    # Preprint is defined as having no 'journal' or 'booktitle' field
    return 'journal' not in entry and 'booktitle' not in entry

# Function to sort papers by the abbreviation of their venue
def get_venue_abbreviation(entry):
    venue = entry.get('journal', entry.get('booktitle', ''))
    return abbreviate_venue(venue)

# Function to generate HTML from bib entry, adjusting image size and center-aligning content
def generate_html(entry):
    esc = lambda value: html.escape(str(value), quote=True)
    authors = ', '.join(convert_author_format(author) for author in entry.get('author', '').split(' and '))
    venue = entry.get('booktitle', entry.get('journal', ''))
    links = publication_links(entry)
    title = esc(clean_title(entry.get('title', 'Untitled')))
    destination = links['paper'] or links['pdf']
    if destination:
        title = f'<a href="{esc(destination)}">{title}</a>'
    badges = ' '.join(f'<span class="publication-rating">({esc(label)})</span>'
                      for label in publication_ratings(entry))
    image = generate_image_filename(entry)
    thumbnail = ''
    if os.path.exists(image):
        thumbnail = (f'<div class="publication-image"><img src="/{esc(image)}" '
                     f'alt="Figure for {esc(clean_title(entry.get("title", "")))}" loading="lazy">'
                     f'<span class="publication-venue">{esc(abbreviate_venue(venue))}</span></div>')
    buttons = []
    for key, label in [('pdf', 'PDF'), ('paper', 'Paper'), ('code', 'Code')]:
        if links[key] and (key != 'paper' or links[key] != links['pdf']):
            buttons.append(f'<a href="{esc(links[key])}" class="publication-link" '
                           f'target="_blank" rel="noopener noreferrer">{label}</a>')
    status = ' <span class="publication-status">Accepted</span>' if publication_status(entry) == 'accepted' else ''
    return (f'<li id="{esc(entry["ID"])}"><div class="pub-row">{thumbnail}'
            f'<div class="publication-text"><div class="title">{title}</div>'
            f'<div class="author">{esc(authors)}.</div>'
            f'<div class="periodical"><em>{esc(clean_full_venue(venue))}, {esc(entry.get("year", ""))}.</em> {badges}{status}</div>'
            f'<div class="links">{" ".join(buttons)}</div></div></div></li>\n')

# Load the .bib file
def load_bib_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as bib_file:
        bib_database = bibtexparser.load(bib_file, parser=BibTexParser(common_strings=True))
    return bib_database.entries

# 会议时间排序列表，基于缩写

# 获取会议时间的排序
def get_conference_time_rank(venue):
    abbreviated_venue = abbreviate_venue(venue)
    if abbreviated_venue in CONFERENCE_TIME_ORDER:
        return CONFERENCE_TIME_ORDER.index(abbreviated_venue)
    return len(CONFERENCE_TIME_ORDER)  # 未在列表中的会议放在最后

def sort_entries_by_conference_time(entries):
    def get_entry_type_rank(entry):
        # 根据 venue abbreviation 判断是否为会议，优先检查会议
        venue = entry.get('booktitle', entry.get('journal', ''))
        abbreviation = abbreviate_venue(venue)
        
        # 如果是会议，返回 0，表示会议优先；如果是期刊，返回 1；预印本返回 2
        if abbreviation in CONFERENCE_TIME_ORDER:
            return 0  # 会议条目优先
        elif 'journal' in entry:
            return 1  # 期刊条目次优
        else:
            return 2  # 预印本/ArXiv 条目最后
        
    return sorted(
        entries,
        key=lambda x: (
            get_entry_type_rank(x),  # 按条目类型排序，会议 > 期刊 > 预印本
            get_conference_time_rank(x.get('booktitle', x.get('journal', ''))),  # 按会议时间顺序排序
        )
    )

# 按年份和会议时间排序
def sort_entries_by_year(entries):
    return sorted(entries, key=publication_sort_key)

# Generate HTML for all entries in the bib file, adding a divider for each year
def generate_bibliography_html(entries, profile=None):
    # Hide arXiv-only records before grouping so empty year headings are omitted.
    sorted_entries = sort_entries_by_year([entry for entry in entries if is_visible(entry)])

    html_content = '''<h2 id="publications" style="margin: 2px 0px -15px;">Publications <temp style="font-size:15px;">[</temp><a href="https://scholar.google.com/citations?hl=en&user=TE7lbQwAAAAJ" target="_blank" style="font-size:15px;">Google Scholar</a><temp style="font-size:15px;">]</temp><temp style="font-size:15px;">[</temp><a href="https://dblp.org/pid/227/4399.html" target="_blank" style="font-size:15px;">DBLP</a><temp style="font-size:15px;">]</temp></h2>

<div class="publications">
<ol class="bibliography">
'''
    if profile is not None:
        links = []
        for key, name in (('google_scholar', 'Google Scholar'), ('dblp', 'DBLP')):
            if profile.get(key):
                links.append(f'<a href="{html.escape(profile[key], quote=True)}">{name}</a>')
        html_content = '<h2 id="publications">Publications <span class="publication-links">' + ' · '.join(links) + '</span></h2>\n<div class="publications">\n<ol class="bibliography">\n'
    last_year = None
    for entry in sorted_entries:
        current_year = entry.get('year', 'Unknown')
        if current_year != last_year:
            # Add a divider for each new year
            html_content += f'<h3 style="margin-top: 20px; margin-bottom: 5px;">{current_year}</h3><hr style="margin-bottom: 5px;">'
            last_year = current_year
        html_content += generate_html(entry)
    html_content += '</ol>\n</div>'
    return html_content

# Main function to execute the conversion
def convert_bib_to_html(bib_file_path, output_html_path, profile=None):
    entries = load_bib_file(bib_file_path)
    html_content = generate_bibliography_html(entries, profile)
    
    # Write the HTML to the output file
    with open(output_html_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)
    
    print(f"HTML file successfully written to {output_html_path}")

if __name__ == '__main__':
    convert_bib_to_html('citations.bib', '_includes/output_file.md')
