# 个人学术主页：资料编辑与统一生成

日常更新步骤见 [主页更新指南](主页更新指南.md)，涵盖资料编辑、自动编译、本地检查和发布。

论文摘要、状态、开放 PDF 归档和自动补缺见 [论文资料维护](publications-guide.md)。统一源文件仍为 `citations.bib`；生成时同步更新完整论文档案、PDF 对账清单和缺失信息报告。

以后主要编辑 `_data/` 下的资料文件和 `citations.bib`。根目录各页面只是入口，`_includes/generated/`、`_includes/output_file.md`、`paper_list.tex` 和 `_generated/` 是自动生成内容。

## 信息在哪里修改

| 信息 | 编辑文件 |
| --- | --- |
| 姓名、职位、学校、学院、邮箱、通讯地址、照片、校标、个人简介、Scholar/CV/LinkedIn 链接 | `_data/profile.json` |
| 研究方向 | `_data/research.json` |
| 招生与科研合作说明 | `_data/recruitment.json` |
| 新闻动态（目前不在首页展示，保留资料） | `_data/news.json` |
| 学生与校友 | `_data/students.json` |
| 课程、专题教程和助教经历 | `_data/teaching.json` |
| 编辑、领域主席和审稿服务 | `_data/services.json` |
| 荣誉与奖项 | `_data/awards.json` |
| 主持经费（导航暂隐藏，资料保留） | `_data/grants.json` |
| 工作经历 | `_data/work.json` |
| 教育经历 | `_data/education.json` |
| 论文作者、标题、年份、会议、链接和代码地址 | `citations.bib` |
| 会议简称与排序规则 | `_data/venues.json` |
| 导航栏目名称与链接 | `_data/navigation.yml` |
| 网站地址、构建排除项、图标和站点级设置 | `_config.yml` |
| 页面字号、字体、正文布局 | `assets/css/homepage.css` |
| 导航背景、校标展示、导航字号 | `assets/css/navigation.css` |

### 先补充个人信息

`profile.json` 中的 `name` 可填写中文姓名，`name_en` 填英文姓名。`department` 和 `address` 目前留空，填写后会显示在首页；`position` 已填“教授”。`bio_zh` 为中文简介段落，`bio_en` 为英文简介段落。

首页个人介绍分为“研究领域、工作经历、教育背景”，分别修改 `_data/research.json`、`_data/work.json`、`_data/education.json`。工作和教育填写 `sections` 中的 `items`：`name` 是单位或学校，`details` 是职位、学位或补充说明，`date` 是起止时间，`url` 是可选链接。按希望显示的顺序填写，通常最新经历在前。时间未知时可以留空。研究领域填写 `items` 中的 `name`（方向名称）与 `details`（具体介绍）。

顶部当前职位与学校仍来自 `profile.json`；`bio_zh` 和 `bio_en` 保留供中英文简历使用，首页改用上述三个栏目，不再重复显示这两段经历介绍。

`avatar`、`school_logo` 和 `cv_link` 是文件路径。例如 `/photo.png`、`/assets/img/seu_logo.png`、`/yujun_cv.pdf`。校标保留原图，CSS 的 `mix-blend-mode: multiply` 让图片白底融入蓝色导航。

### JSON 怎么编辑

中文字段直接填写中文；对应的 `_en` 字段用于英文简历以及学生、教学、服务、荣誉和经费页面。首页研究、工作和教育使用中文。未提供英文字段时，生成器会使用相同的中文或原始内容，不会自动翻译。JSON 不支持 `//` 注释；想隐藏小标题可把 `title` 和 `title_en` 填成空字符串，或删除这两个字段。

通用栏目使用 `sections` → `items`。每条资料的 `name` 是标题；`details` 是说明；`date` 是时间；`url` 是可选链接。例如在 `awards.json` 的 `items` 中添加：

```json
{
  "name": "你的奖项名称",
  "name_en": "English award name",
  "details": "授奖单位或补充说明",
  "details_en": "Organization or details",
  "date": "2026"
}
```

新闻的 `date` 格式为 `YYYY-MM`，生成时会自动按时间从新到旧排序。学生名字放在 `name`，背景、去向和成果放在 `details`。添加多条 JSON 项目时，用逗号分隔，最后一项后面不要加逗号。

论文的 BibTeX key 必须唯一。迁移时发现两处重复 key，已给第二条加 `_2` 后缀，保留全部 82 条论文记录；请在整理论文时核对对应记录是否确实需要两条。

## 论文分级标签

生成器参照 Yiwei Wang 的做法，在 venue 后以红色显示 `(CCF-A)` 和 `(JCR Q1)`，网页与 LaTeX 使用同一份 `_data/venues.json` 规则。支持会议/期刊全称、简称与大小写变化；即使会议论文存为 `@article`，也不会因此获得 JCR 标签。

ECCV、EMNLP 和 SIGGRAPH Asia 另显示 `(ICORE A*)`，依据 ICORE2026 官方评级：[ECCV](https://portal.core.edu.au/conf-ranks/479/)、[EMNLP](https://portal.core.edu.au/conf-ranks/448/)、[SIGGRAPH Asia](https://portal.core.edu.au/conf-ranks/957/)。2026 官方名称为 ICORE（原 CORE）；NAACL 为 A，不标 A*。此标签独立于 CCF，维护在 `icore_a_star`；同样不会给 Workshop/Findings 添加主会标签。标签按当前配置评级展示，不代表论文发表当年的评级。

BMC Bioinformatics 的 Q1 依据为 2025 年 JCR 的 Mathematical & Computational Biology 类别，来源记录在 `rating_sources.jcr`。JCR 分区与中科院分区不同。

网站和中英文简历均隐藏 arXiv 预印本，原始记录保留在 `citations.bib`。已发表论文即使 PDF 链接指向 arXiv 仍保留；判断依据是发表场所，而非单独的链接。

`ccf_a` 按 [CCF 2026 第七版目录](https://www.ccf.org.cn/Academic_Evaluation/By_category/) 维护；Findings、Workshop 等不加 CCF-A 标签。`jcr_q1_journals` 沿用参考网站明确列出的 TPAMI、TKDE、TSP，当前仓库涉及的是 TPAMI。JCR 标签依据维护的期刊清单，不逐篇查询发表当年的分区；新增期刊或分区变动时，确认后更新清单。来源信息保存在 `rating_sources`。

LaTeX 简历模板需加载 `\usepackage{xcolor}` 以显示红色标签。修改规则或 BibTeX 后运行 `.\generate.ps1`；已启动 `-Watch` 时保存会自动更新。

## 本地使用：一个入口

当前电脑已经准备好项目专用 `.venv`。在仓库目录用 PowerShell 执行：

```powershell
# 编辑资料/简历时启动一次：保存后自动生成和编译，Ctrl+C 停止
.\generate.ps1 -Watch

# 修改资料后生成所有网页片段、简历源文件和中英文 PDF
.\generate.ps1

# 生成内容、编译 PDF 并构建网站
.\generate.ps1 -Build

# 只校验资料，不生成或编译
.\generate.ps1 -Check

# 单独生成文本时，明确跳过 PDF 编译
.\generate.ps1 -SkipCV
```

在其他电脑上首次安装：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
bundle install
```

跨平台入口为 `python scripts/generate_all.py`，可以加 `--check`、`--build` 或 `--watch`。新电脑需要先激活 Python 虚拟环境。

保持现有 Jekyll 服务运行，在另一个终端运行 `.\generate.ps1 -Watch`，然后保存资料、刷新现有页面即可。监听覆盖 `_data/*.json`、`_data/*.yml`、`citations.bib` 和 `Latex/*/main.tex`，保存稳定约半秒后更新。资料变化会生成网页与简历源文件并编译 PDF；直接保存 LaTeX 则只编译已保存的源文件，不重新生成、不覆盖编辑器修改。生成器自己写出的 `.tex` 不触发第二轮编译，PDF 和预览快照也不受监听，避免循环。资料或编译错误会提示并继续监听，旧 PDF 保留；修正后保存会自动重试。若 Jekyll 以 `--no-watch` 启动，需要改用默认的 `bundle exec jekyll serve` 才会自动重建；需要浏览器自动刷新时可加 `--livereload`。

修改 `_config.yml` 后需重启自己的 Jekyll 服务；个人资料现在来自 `_data/profile.json`，可避免把校标等信息放在服务器缓存的配置里。

## 生成了什么

- `_includes/generated/*.md`：首页和各栏目内容。
- `_includes/output_file.md`：从 BibTeX 生成的网站论文列表。
- `paper_list.tex`：CV 论文列表。
- `_generated/cv_zh.md`、`_generated/cv_en.md`：中英文简历内容。
- `_generated/cv_zh.tex`、`_generated/cv_en.tex`：LaTeX 简历片段，供简历模板引用；不是完整可独立编译的文档。
- `Latex/中文简历/main.tex`：完整中文简历，参考 Yiwei Wang 的中文简历版式；同一次生成更新，使用 XeLaTeX 和 Fandol 字体，可在 Codex 内置 LaTeX 编辑器编译。内容内嵌，无外部片段依赖。包含工作、教育、研究方向、论文、学术服务、教程、教学和荣誉；暂不包含经费栏目。
- `Latex/英文简历/main.tex`：完整英文简历，参考 Yiwei Wang 与旧英文 CV 共用的 Trey Hunner resume 模板。姓名居中、栏目大写并加横线、日期右对齐、论文作者中的 Yujun Cai 加粗，保留论文评级标签。模板和内容内嵌，不依赖 `resume.cls` 或外部 `input` 文件，可单独在内置 LaTeX 编辑器编译。

英文 CV 使用资料中的 `_en` 字段。工作经历以单行展示，不显示学院与工作说明；`cv_title_en` 和 `cv_date_en` 可覆盖英文 CV 中的职位标题和日期（如 UQ 的 Lecturer / QS #42 标注、Meta 合并经历）。`department_en`、`description_en` 和 `previous_roles_en` 保留完整经历资料，当前英文 CV 不展示；研究方向也不在当前英文 CV 中展示。教育的 `note_en` 保存 GPA、排名等补充资料。学术服务的 `cv_title_en` 可简化 CV 栏目名称，服务与教学的 `cv_extra_items_en` 保存从旧 CV 合入、暂不显示在主页上的记录。旧 CV 与当前主页的教学日期冲突时采用当前主页资料。未确认到月份的任职结束时间先保留年份。

修改资料后运行 `.\generate.ps1` 自动生成并编译两种简历。直接修改 `main.tex` 后运行 `.\compile_cv.ps1 -Language en`（中文用 `zh`），或保持 `-Watch` 运行自动编译。重新生成仍会以资料源文件覆盖手动修改，因此长期保留的内容调整应同步到资料或模板生成器。

正式 PDF 默认输出为 `Latex/英文简历/main.pdf` 和 `Latex/中文简历/main.pdf`，实际当前文件由 `profile.json` 中的 `cv_link`、`cv_zh_link` 管理。若当前 PDF 被预览程序占用，编译器自动生成 `main-<内容摘要>.pdf` 并更新对应链接，保留正在打开的旧文件。主页的简历图标在新窗口打开对应 PDF，并带构建时间参数以更新浏览器缓存。`scripts/check_site.py` 验证图标链接、输出 PDF 的存在及其与当前编译版本一致。英文简历的工作和教育日期统一只显示年份，资料中的原始月份仍保留。

Windows 上 `.\compile_cv.ps1` 默认编译两种已保存的 `main.tex`，使用 Codex 自带的 Tectonic；`-Language en` / `zh` 选择一种，`-Force` 强制重编译。跨平台入口是 `python scripts/compile_cv.py`，Linux/macOS 使用 PATH 中的 Tectonic，也可通过 `TECTONIC_BIN` 指定路径。首次运行下载字体和宏包到 `.venv/tectonic-cache`，以后复用；源码和 PDF 均未变化时跳过编译。PDF 只在编译成功且没有缺字后原子替换，失败保留上一版。

内置实时预览若遇到 `Unable to find standard directories for platform`，继续使用上述独立入口；它不修改应用设置、不另装桌面 TeX。编译后打印 `.venv/cv-previews/` 下带内容摘要的预览快照路径：在 Codex 中打开该快照，保留源文件编辑器，避免桌面预览占用正式网页 PDF。每次内容变化后打开新的快照；同一快照不被覆盖。正式 PDF 被占用时自动切换到新的独立版本，其他编译/文件更新失败仍保留上一版并报错。旧的 `cai_yujun_cv_*.pdf` 和根目录 `yujun_cv.pdf` 保留作历史文件，图标已使用当前输出。

生成文件可随资料一起提交，避免普通本地 Jekyll 预览缺少片段；GitHub Actions 仍会在发布前重新生成，以资料源文件为准。

## 自动构建与发布

`.github/workflows/site.yml` 已替换旧的引文抓取工作流。

PR：校验资料和生成器/编译/监听测试 → 生成所有内容 → 编译中英文 PDF → Jekyll 构建 → 检查导航、校标、首页头像、中英文简历图标链接及 PDF 内容。

Actions 下载官方 [Tectonic 0.17.0](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0) Linux 可执行文件并校验公布的 SHA-256，缓存宏包和字体。编译使用官方支持的 [独立文档命令](https://tectonic-typesetting.github.io/book/latest/v2cli/compile.html)。云端编译失败会阻止发布，避免网页链接指向旧简历。

主分支 `main` 提交：完成相同检查 → 上传 `_site` → 发布到 GitHub Pages。联网论文查询和下载不会在发布时运行。

首次启用需要在 GitHub 仓库 `Settings → Pages → Build and deployment → Source` 中选择 `GitHub Actions`，然后提交并推送这些文件。当前只完成本地配置，尚未修改远程设置或推送。

官方参考：[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 验证命令

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
.\generate.ps1 -Build
.venv\Scripts\python.exe scripts/check_site.py
```

王艺炜参考仓库 `reference-wangywust/` 及其中的两份 Yiwei CV 已删除。本站的内容生成、简历编译和网站构建不依赖这些参考资料。
