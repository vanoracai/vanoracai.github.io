from bibliography_common import CONFERENCE_TIME_ORDER, abbreviate_venue, publication_ratings, is_arxiv
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding

# Dictionary to map venue keywords to abbreviations (CCF Recommended Conferences and more specific examples)

# 定义会议时间顺序

# 函数：将作者名字从 "Last Name, First Name" 改为 "First Name Last Name"
def format_author_names(authors_str):
    authors = authors_str.split(' and ')
    formatted_authors = []
    for author in authors:
        parts = author.split(', ')
        if len(parts) == 2:
            # 改为 "First Name Last Name" 格式
            formatted_authors.append(f"{parts[1]} {parts[0]}")
        else:
            formatted_authors.append(author)
    return formatted_authors

# 函数：获取会议在排序中的位置
def get_conference_order(venue):
    if venue in CONFERENCE_TIME_ORDER:
        return CONFERENCE_TIME_ORDER.index(venue)
    else:
        return len(CONFERENCE_TIME_ORDER)

# 函数：将.bib转换为LaTeX格式的论文列表，使用enumerate环境
def bib_to_paper_list(bib_file):
    # 读取.bib文件并使用BibTexParser
    with open(bib_file, encoding='utf-8') as bibtex_file:
        parser = BibTexParser(common_strings=True)  # 保留特殊字符并禁用多余的转义
        bib_database = bibtexparser.load(bibtex_file, parser=parser)
    
    paper_list = []

    for entry in bib_database.entries:
        # 跳过arXiv论文
        if is_arxiv(entry):
            continue

        authors_str = entry.get('author', 'Unknown')
        authors = format_author_names(authors_str)  # 改为 "First Name Last Name" 格式
        title = entry.get('title', 'Title not available').rstrip('.')
        year = entry.get('year', '')
        venue = entry.get('booktitle', entry.get('journal', ''))
        venue = abbreviate_venue(venue)  # 使用缩写规则

        # 生成格式化条目，去除页码，保持特殊字符和LaTeX格式
        formatted_entry = f"{authors[0]}, {', '.join(authors[1:])}. {title}, \\textit{{{venue} {year}}}.".strip()
        labels = [rf'\textcolor{{red}}{{({label})}}' for label in publication_ratings(entry)]
        if labels:
            formatted_entry += ' ' + ' '.join(labels)

        # 存储年份、venue和条目，用于排序
        paper_list.append((int(year), venue, formatted_entry))

    # 按年份降序排序，然后按会议时间顺序进行排序
    paper_list.sort(key=lambda x: (-x[0], get_conference_order(x[1]), x[1].lower() if x[1] else ''))

    # 为每篇论文添加编号 [1], [2], 等
    numbered_papers = [f"\\item {entry}" for i, (_, _, entry) in enumerate(paper_list)]
    
    # 返回带有 \begin{enumerate} 和 \end{enumerate} 的论文列表
    return "\\begin{enumerate}\n" + "\n\n".join(numbered_papers) + "\n\\end{enumerate}"

if __name__ == '__main__':
    with open('paper_list.tex', 'w', encoding='utf-8') as handle:
        handle.write(bib_to_paper_list('citations.bib'))
