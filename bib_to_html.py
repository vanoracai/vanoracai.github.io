import os
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding

# Dictionary to map venue keywords to abbreviations (CCF Recommended Conferences and more specific examples)
VENUE_ABBREVIATIONS = {
    'ACM SIGKDD': 'KDD',
    'Computer Vision and Pattern Recognition': 'CVPR',
    'arXiv preprint': 'Preprint',
    'International Conference on Learning Representations': 'ICLR',
    'Advances in Neural Information Processing Systems': 'NeurIPS',
    'Association for Computational Linguistics': 'ACL',
    'International Conference on Data Mining': 'ICDM',
    'International Joint Conference on Artificial Intelligence': 'IJCAI',
    'Conference on Computer Vision and Pattern Recognition': 'CVPR',
    'International Conference on Computer Vision': 'ICCV',
    'International Conference on Database Systems for Advanced Applications': 'DASFAA',
    'Symposium on Theory of Computing': 'STOC',
    'International Symposium on Computer Architecture': 'ISCA',
    'ACM Conference on Information and Knowledge Management': 'CIKM',
    'International Conference on Web Search and Data Mining': 'WSDM',
    'International Conference on Very Large Data Bases': 'VLDB',
    'Annual Meeting of the Association for Computational Linguistics': 'ACL',
    'International Conference on Automated Planning and Scheduling': 'ICAPS',
    'International Symposium on Information Theory': 'ISIT',
    'IEEE International Conference on Robotics and Automation': 'ICRA',
    'European Conference on Computer Vision': 'ECCV',
    'International Symposium on Software Testing and Analysis': 'ISSTA',
    'International Symposium on Theoretical Aspects of Software Engineering': 'TASE',
    'International Conference on Artificial Intelligence and Statistics': 'AISTATS',
    'IEEE International Conference on Data Engineering': 'ICDE',
    'International Conference on Principles of Knowledge Representation and Reasoning': 'KR',
    'Conference on Uncertainty in Artificial Intelligence': 'UAI',
    'International Conference on Knowledge Capture': 'K-CAP',
    'IEEE International Conference on Computer Communications': 'INFOCOM',
    'IEEE Transactions on Signal Processing': 'TSP',
    'Proceedings of the 2017 ACM on Conference on Information and Knowledge Management': 'CIKM',
    'Proceedings of the Web Conference': 'WWW',
    'International Symposium on Software Reliability Engineering': 'ISSRE',
    'ECCV': 'ECCV',
    'Machine Learning and Knowledge Discovery in Databases: European': 'ECML',
    'Learning on Graph': 'LOG',
    'AAAI': 'AAAI',
    'Transactions on Knowledge and Data Engineering': 'TKDE',
    'European Conference on Machine Learning': 'ECML',
    'CONLL': 'CONLL',
    'EMNLP': 'EMNLP',
    'Empirical Methods': 'EMNLP',
    ' CHI Conference on Human Factors in Computing Systems': 'CHI',
    'European Conference on Computer Vision': 'ECCV',
    'pattern analysis and machine intelligence':'TPAMI',
    'Conference on Computational Linguistics':'COLING',
    # Add more mappings for common conferences/journals
}

# Function to convert 'Last, First' to 'First Last'
def convert_author_format(author_name):
    if ',' in author_name:
        parts = author_name.split(',')
        return f"{parts[1].strip()} {parts[0].strip()}"
    return author_name  # If no comma, return as is

# Function to clean and abbreviate the venue for the image badge using keyword matching
def abbreviate_venue(venue):
    # Clean venue by removing backslashes and special characters
    venue_cleaned = venue.replace("\\", "").replace("&", "and")
    
    # Match based on keywords in the venue name
    for keyword, short_name in VENUE_ABBREVIATIONS.items():
        if keyword in venue_cleaned:
            return short_name
    return venue_cleaned  # Return cleaned venue if no abbreviation is found

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

# Function to check if the paper is specifically an ArXiv preprint
def is_arxiv(entry):
    # ArXiv is identified by 'journal' field containing 'arXiv'
    return 'journal' in entry and 'arxiv' in entry['journal'].lower()

# Function to sort papers by the abbreviation of their venue
def get_venue_abbreviation(entry):
    venue = entry.get('journal', entry.get('booktitle', ''))
    return abbreviate_venue(venue)

# Function to generate HTML from bib entry, adjusting image size and center-aligning content
def generate_html(entry):
    # Handling multiple authors by splitting and formatting them
    authors = entry.get('author', 'Unknown Author').split(" and ")
    formatted_authors = ', '.join([convert_author_format(author) for author in authors])

    # Clean the title to remove special characters
    clean_title_text = clean_title(entry.get('title', 'Untitled'))

    # Get the full venue (journal or booktitle) and its abbreviation for the image badge
    full_venue = entry.get('journal', entry.get('booktitle', 'Preprint'))
    abbreviated_venue = abbreviate_venue(full_venue)
    cleaned_full_venue = clean_full_venue(full_venue)  # Cleaned full venue without abbreviations

    # Use the entry's ID to generate the image filename
    img_src = generate_image_filename(entry)
    
    # Check if the image file exists
    image_exists = os.path.exists(img_src)
    
    # Get code URL or default code link (if 'code_url' field is present in the bib file, use it)
    code_url = entry.get('code_url', entry.get('code', '#'))

    # Clean the URL to remove LaTeX artifacts (e.g., \_ to _)
    pdf_url = clean_url_latex_artifacts(entry.get('url', '#'))
    
    # Define the image style to make it slightly longer and adjust layout
    image_style = "width: 110px; height: 130px; object-fit: cover; margin-bottom: 0;"  # Slightly taller image
    
    # If the image exists, include the image and layout using flexbox
    if image_exists:
        html = f'''
<li>
<div class="pub-row" style="display: flex; align-items: center;"> <!-- Center-align the content -->
  <div class="image-container" style="flex: 0 0 auto; margin-right: 15px; position: relative;">
    <img src="{img_src}" class="teaser img-fluid z-depth-1" style="{image_style}">
    <abbr class="badge" style="position: absolute; top: 5px; left: 5px; background-color: #007bff; color: white; padding: 5px;">{abbreviated_venue}</abbr>
  </div>
  <div class="text-container" style="flex: 1; display: flex; flex-direction: column; justify-content: center;">
      <div class="title"><a href="{pdf_url}">{clean_title_text}</a></div>
      <div class="author"><strong>{formatted_authors}</strong>.</div>
      <div class="periodical"><em>{cleaned_full_venue}, {entry.get('year', '2024')}.</em></div>
      <div class="links" style="margin-top: 10px;">
        <a href="{pdf_url}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
        <a href="{code_url}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Code</a>
      </div>
  </div>
</div>
</li>
'''
    else:
        # If the image doesn't exist, remove the image section and extend the text
        html = f'''
<li>
<div class="pub-row" style="display: flex; align-items: center;">
  <div class="text-container" style="flex: 1; display: flex; flex-direction: column; justify-content: center;">
      <div class="title"><a href="{pdf_url}">{clean_title_text}</a></div>
      <div class="author"><strong>{formatted_authors}</strong>.</div>
      <div class="periodical"><em>{cleaned_full_venue}, {entry.get('year', '2024')}.</em></div>
      <div class="links" style="margin-top: 10px;">
        <a href="{pdf_url}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
        <a href="{code_url}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Code</a>
      </div>
  </div>
</div>
</li>
'''

    return html

# Load the .bib file
def load_bib_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as bib_file:
        bib_database = bibtexparser.load(bib_file, parser=BibTexParser(customization=homogenize_latex_encoding))
    return bib_database.entries

# 会议时间排序列表，基于缩写
CONFERENCE_TIME_ORDER = [
    'NeurIPS',
    'COLING',
    'EMNLP',
    'ACL',
    'ECCV',
    'CVPR',
    'NAACL',
    'ICCV',
    'ICLR',
    'AAAI',
    'CHI',
    'CONLL',
    # 你可以根据需要添加更多的会议
]

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
    # 按年份分组
    entries_by_year = {}

    for entry in entries:
        year = int(entry.get('year', 0))
        if year not in entries_by_year:
            entries_by_year[year] = []
        entries_by_year[year].append(entry)

    # 在每个年份内按会议时间排序
    for year, year_entries in entries_by_year.items():
        entries_by_year[year] = sort_entries_by_conference_time(year_entries)

    # 按年份从新到旧返回所有的条目
    sorted_entries = []
    for year in sorted(entries_by_year.keys(), reverse=True):
        sorted_entries.extend(entries_by_year[year])

    return sorted_entries

# Generate HTML for all entries in the bib file, adding a divider for each year
def generate_bibliography_html(entries):
    # Sort entries by year from newest to oldest, grouped by venue, with Preprints/arXiv at the bottom for each year
    sorted_entries = sort_entries_by_year(entries)

    html_content = '''<h2 id="publications" style="margin: 2px 0px -15px;">Selected Publications <temp style="font-size:15px;">[</temp><a href="https://scholar.google.com/citations?hl=en&user=TE7lbQwAAAAJ" target="_blank" style="font-size:15px;">Google Scholar</a><temp style="font-size:15px;">]</temp><temp style="font-size:15px;">[</temp><a href="https://dblp.org/pid/227/4399.html" target="_blank" style="font-size:15px;">DBLP</a><temp style="font-size:15px;">]</temp></h2>

<div class="publications">
<ol class="bibliography">
'''
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
def convert_bib_to_html(bib_file_path, output_html_path):
    entries = load_bib_file(bib_file_path)
    html_content = generate_bibliography_html(entries)
    
    # Write the HTML to the output file
    with open(output_html_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)
    
    print(f"HTML file successfully written to {output_html_path}")

# Example usage:
convert_bib_to_html('citations.bib', '_includes/output_file.md')
