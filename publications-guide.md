# 论文资料维护

`citations.bib` 是论文唯一数据源。网页、中英文 CV、PDF 清单与完整性报告都由它生成，不另建一套手工论文列表。

## 数据与文件

| 内容 | 位置 |
| --- | --- |
| 论文、摘要、状态与来源 | `citations.bib` |
| 按 citation key 命名的开放全文 | `pdfs/{key}.pdf` |
| 按 citation key 命名的可选配图 | `Image/{key}.png` |
| 自动对账的下载清单 | `pdfs/paper_pdf_manifest.csv` |
| 含全部记录的机器可读档案 | `_generated/publications.json` |
| 缺失字段及疑似重复项 | `_generated/publication_audit.md` |
| 原有历史 PDF | `Paper/`（保留） |

## BibTeX 扩展字段

- `status`：`published`、`accepted`、`preprint` 或 `unknown`。网页和 CV 仅展示前两种；所有状态保留在档案里。迁移时按原有 venue 推断的状态带 `status_source={existing_metadata}`，不代表已联网核验录用。
- `abstract`：从原始来源取得的完整摘要，不存 AI 改写。`abstract_source` 必填；`abstract_method` 记录原站文本或聚合元数据，`abstract_retrieved` 记录取得日期。
- `summary`：本地记录的简短介绍，与原始摘要分开，不在网页展示。`summary_source` 必填。自动补全默认摘取摘要开头的完整句子，标记 `summary_method={extractive}`，不编写没有依据的结论。可自行改成更好的简介并标记 `summary_method={manual}`。
- `paper_url`：论文详情页；`pdf_url`：外部全文；`pdf_source`：本地全文的下载源；`doi`：DOI。
- `code_url`：真实项目代码地址。没有就省略，勿填 `#`。
- `duplicate_of`：确认重复后可指向保留的 citation key，隐藏该重复条目的网页/CV 展示，但保留档案。疑似重复只在报告列出，不会自动删除。

每条 PDF 优先链接本地 `pdfs/{key}.pdf`；文件不存在则使用外部全文；只有详情页时显示 Paper。没有可用地址时不生成对应按钮。摘要、简介及其来源仅保存在本地 BibTeX 和生成的完整档案中，不输出到网页 HTML，也不加入 CV。`citations.bib` 不复制到构建后的网站。

## 日常命令

```powershell
# 离线校验资料
.\generate.ps1 -Check

# 生成网站内容和 CV，并编译中英文 CV
.\generate.ps1

# 仅更新文本和 PDF 档案清单
.\generate.ps1 -SkipCV

# 显式联网补缺：保留已有摘要/简介，按标题核对，下载并核验开放 PDF
.venv\Scripts\python.exe scripts/enrich_publications.py --download

# 只补选定条目；不下载 PDF 时省略 --download
.venv\Scripts\python.exe scripts/enrich_publications.py --keys cai2018weakly

# 只重新对账并生成完整性报告
.venv\Scripts\python.exe scripts/publication_catalog.py
```

联网补全从现有链接、arXiv、OpenReview、Crossref 与 OpenAlex 找资料。同题匹配需要达到严格阈值；聚合搜索额外检查 Cai 作者。参考网站的 BibTeX 只提供查找线索，不把其 AI 摘要复制为原摘要。下载先验证 PDF 格式并比较前两页的标题词，再原子写入；既有 PDF 不覆盖。网络缓存放在 `.venv/publication-cache/`，可复用。遇到付费墙、限流或未索引论文，保留缺失状态供下次补充，不伪造数据。

普通生成、监听和 GitHub Actions 不执行联网搜索。清单每次按 BibTeX 重建，自动去掉幽灵行、重复行并更新标题；未被引用的 PDF 保留且在报告列出。原有 `fill_bib_url.py` 为历史辅助脚本，新资料优先使用上述增量入口。
