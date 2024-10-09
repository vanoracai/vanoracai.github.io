from habanero import Crossref
import bibtexparser
import requests
from bs4 import BeautifulSoup
import difflib
import re

# 从 DOI 获取 URL
def get_url_from_doi(doi):
    return f"https://doi.org/{doi}"

# 从 Arxiv ID 获取 URL
def get_url_from_arxiv(arxiv_id):
    return f"https://arxiv.org/abs/{arxiv_id}"

# 标准化会议名称
def normalize_conference_name(conference):
    conference = conference.lower()
    if 'acl' in conference:
        return 'Association for Computational Linguistics'
    elif 'iclr' in conference:
        return 'International Conference on Learning Representations'
    return conference

# 简化标题以提高匹配效果
def simplify_title(title):
    # 移除标点符号和常见无意义词
    title = re.sub(r'[^\w\s]', '', title)  # 移除标点
    return ' '.join(title.split())  # 移除多余空格

# 使用 ACL Anthology 查询论文 URL
def get_acl_anthology_url(title):
    simplified_title = simplify_title(title)
    search_url = f"https://aclanthology.org/search/?q={simplified_title.replace(' ', '+')}"
    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 获取多个搜索结果
            results = soup.find_all('a', href=True)
            for result in results:
                if 'papers' in result['href']:
                    # 获取链接文本进行比较
                    result_title = result.text.strip().lower()
                    if compare_titles(title.lower(), result_title):
                        return f"https://aclanthology.org{result['href']}"
        print(f"No ACL Anthology URL found for title: {title}")
    except Exception as e:
        print(f"Error searching ACL Anthology for title '{title}': {e}")
    return None

# 使用 Crossref API 获取 DOI 或 URL
def get_doi_from_title_and_conference(title, conference=None):
    cr = Crossref()
    query = title
    if conference:
        normalized_conference = normalize_conference_name(conference)
        query = f"{title} {normalized_conference}"
    try:
        result = cr.works(query=query, limit=3)
        for item in result['message']['items']:
            fetched_title = item.get('title', [''])[0]
            if compare_titles(title, fetched_title):
                return item.get('DOI', None)
    except Exception as e:
        print(f"Error fetching DOI for title '{title}' (conference '{conference}'): {e}")
    return None

# 标题相似度比较
def compare_titles(original_title, fetched_title, threshold=0.85):
    original_title = clean_title(original_title)
    fetched_title = clean_title(fetched_title)
    similarity = difflib.SequenceMatcher(None, original_title, fetched_title).ratio()
    return similarity >= threshold

# 清理标题中的特殊符号
def clean_title(title):
    return title.lower().replace('{', '').replace('}', '').replace('\n', '').strip()

# 从 journal 字段中提取 Arxiv ID
def extract_arxiv_id(entry):
    if 'journal' in entry:
        journal = entry['journal']
        arxiv_match = re.search(r'arxiv[:\s]*([\d.]+)', journal, re.IGNORECASE)
        if arxiv_match:
            return arxiv_match.group(1)
    return None

# 解析 .bib 文件
def fill_bib_urls(bib_file):
    with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    # 遍历每个条目
    for entry in bib_database.entries:
        if 'url' not in entry:
            # 先检查是否为 Arxiv 论文
            arxiv_id = extract_arxiv_id(entry)
            if arxiv_id:
                entry['url'] = get_url_from_arxiv(arxiv_id)
                print(f"Arxiv URL added for {entry.get('title', 'No Title')}")
            else:
                booktitle = entry.get('booktitle', None)
                title = entry.get('title', None)
                if booktitle and title:
                    # 先通过 ACL Anthology 查询 URL
                    url = get_acl_anthology_url(title)
                    if url:
                        entry['url'] = url
                        print(f"ACL Anthology URL added for {entry.get('title', 'No Title')}")
                    else:
                        # 如果 ACL Anthology 未找到结果，使用 Crossref API 查询 DOI
                        doi = get_doi_from_title_and_conference(title, booktitle)
                        if doi:
                            entry['doi'] = doi
                            entry['url'] = get_url_from_doi(doi)
                            print(f"Conference URL added for {entry.get('title', 'No Title')}")
                        else:
                            print(f"DOI not found for title: {title} (conference: {booktitle})")
                else:
                    # 如果没有 Arxiv ID 或会议信息，尝试通过标题查找 DOI
                    title = entry.get('title', None)
                    if title:
                        doi = get_doi_from_title_and_conference(title)
                        if doi:
                            entry['doi'] = doi
                            entry['url'] = get_url_from_doi(doi)
                        else:
                            print(f"DOI not found for title: {title}")

    # 保存修改后的 .bib 文件
    with open(bib_file, 'w', encoding='utf-8') as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)

# 示例：自动填充 url 字段
fill_bib_urls('citations.bib')