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
    'Proceedings of the Web Conference': 'WWW',
    'Machine Learning and Knowledge Discovery in Databases: European': 'ECML',
    'Learning on Graph': 'LOG',
    'AAAI': 'AAAI',
    'Transactions on Knowledge and Data Engineering': 'TKDE',
    'European Conference on Machine Learning': 'ECML',
    'CONLL': 'CONLL',
    'EMNLP': 'EMNLP',
    'Empirical Methods': 'EMNLP',
    'CHI Conference on Human Factors in Computing Systems': 'CHI',
    'European Conference on Computer Vision': 'ECCV',
    'pattern analysis and machine intelligence':'TPAMI',
    'Conference on Computational Linguistics':'COLING',
    # Add more mappings for common conferences/journals
}

# 定义会议时间顺序
CONFERENCE_TIME_ORDER = [
    'TPAMI'
    'NeurIPS',
    'EMNLP',
    'COLING'
    'ACL',
    'ECCV',
    'ICCV',
    'CVPR',
    'NAACL',
    'ICLR',
    'AAAI'
]

# 函数：根据缩写表替换venue，不区分大小写
def abbreviate_venue(venue):
    venue_lower = venue.lower()  # 将输入venue转换为小写
    for key, abbreviation in VENUE_ABBREVIATIONS.items():
        if key.lower() in venue_lower:  # 对字典中的key也转换为小写进行匹配
            return abbreviation
    return venue

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
    with open(bib_file) as bibtex_file:
        parser = BibTexParser(common_strings=True)  # 保留特殊字符并禁用多余的转义
        bib_database = bibtexparser.load(bibtex_file, parser=parser)
    
    paper_list = []

    for entry in bib_database.entries:
        # 跳过arXiv论文
        if 'arxiv' in entry.get('journal', '').lower():
            continue

        authors_str = entry.get('author', 'Unknown')
        authors = format_author_names(authors_str)  # 改为 "First Name Last Name" 格式
        title = entry.get('title', 'Title not available').rstrip('.')
        year = entry.get('year', '')
        venue = entry.get('booktitle', entry.get('journal', ''))
        venue = abbreviate_venue(venue)  # 使用缩写规则

        # 生成格式化条目，去除页码，保持特殊字符和LaTeX格式
        formatted_entry = f"{authors[0]}, {', '.join(authors[1:])}. {title}, \\textit{{{venue} {year}}}.".strip()

        # 存储年份、venue和条目，用于排序
        paper_list.append((int(year), venue, formatted_entry))

    # 按年份降序排序，然后按会议时间顺序进行排序
    paper_list.sort(key=lambda x: (-x[0], get_conference_order(x[1]), x[1].lower() if x[1] else ''))

    # 为每篇论文添加编号 [1], [2], 等
    numbered_papers = [f"\\item {entry}" for i, (_, _, entry) in enumerate(paper_list)]
    
    # 返回带有 \begin{enumerate} 和 \end{enumerate} 的论文列表
    return "\\begin{enumerate}\n" + "\n\n".join(numbered_papers) + "\n\\end{enumerate}"

# 示例用法
bib_file = "citations.bib"
output_file = "paper_list.tex"  # 生成的 LaTeX 文件

# 将结果写入到 LaTeX 文件
with open(output_file, "w") as f:
    f.write(bib_to_paper_list(bib_file))
