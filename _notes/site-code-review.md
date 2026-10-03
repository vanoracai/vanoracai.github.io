# 主页代码与自动化整改建议

检查日期：2026-10-02。

本次检查使用的参考仓库提交为 `1478218b2531a4fdc0985436ff4a352ff7e4cfbd`，曾保存在 `reference-wangywust/`。本地参考副本已于 2026-10-03 删除，下文保留检查时的路径与结论。

## 参考仓库的自动化实际上做什么

1. `MetaData/data/*.json` 存放中英文工作经历、教育、教学、学术服务、荣誉和经费等资料；`MetaData/data/citations.bib` 是论文源文件。
2. `MetaData/scripts/generate_all.py` 是统一生成入口：调用论文生成器、个人信息生成器和各栏目生成器，再汇总中英文 Markdown 简历。
3. `MetaData/scripts/site_pages.py` 生成学术服务、教学、荣誉、经费页面，并用 `METADATA` 标记更新首页的工作和教育经历区块，保留其他手写内容。
4. `MetaData/generated/` 保存网站论文列表、Markdown 简历和 LaTeX 片段；中英文简历的 `main.tex` 通过 `input` 引用这些片段。
5. `fill_bib_url.py`、`python/paper_pdf_from_bib.py` 等是查找论文链接和下载 PDF 的辅助脚本；前者不在 `generate_all.py` 的默认流程中。
6. Google Drive 和 Hugging Face 上传／下载脚本是其他辅助工具，没有发现它们被站点生成入口调用。

下载的当前提交没有 `.github/workflows`。因此能确认的是内容生成自动化，不能仅凭这些代码确认 GitHub 仓库的 Pages 发布设置。`generate_all.py` 也没有编译简历 PDF；生成 LaTeX 片段与编译 PDF 是两个步骤。

参考文件：

- `reference-wangywust/MetaData/README.md`
- `reference-wangywust/MetaData/scripts/generate_all.py`
- `reference-wangywust/MetaData/scripts/site_pages.py`
- `reference-wangywust/MetaData/scripts/common.py`
- `reference-wangywust/Latex/中文简历/main.tex`
- `reference-wangywust/Latex/英文简历/main.tex`

## 值得借鉴的部分

网页和简历共用结构化资料，更新一次即可生成多种输出；论文共用一份 BibTeX；手写首页内容与生成区块有明确边界。

## 需要改进的部分

参考仓库的学校资料仍有多份来源：JSON 中的工作经历写东南大学，`_config.yml` 仍写 UC Merced，首页模板又直接写东南大学。这说明资料与模板还没有完全统一。

它的三个布局文件分别约 40KB、40KB、64KB，大量样式直接写在 HTML 中；复制这些模板会把重复维护的问题一起带进来。我们适合参考页面结构和字体，保留独立的 CSS 与共享导航组件。

## 整改前的情况

- 已有 `citations.bib` → `bib_to_html.py` → `_includes/output_file.md`，以及 `bib_to_latex.py` → `paper_list.tex`，但还没有统一生成入口。
- 两个论文转换脚本分别维护会议简称等规则，适合抽成共享模块。
- `.github/workflows/google_scholar_crawler.yaml` 在页面构建和每天定时触发时运行，但其中引用的 `google_scholar_crawler/` 目录不存在。按当前仓库文件执行会失败；这也不是网站部署工作流。
- 首页、教学和服务主要手工维护，更新网站与 CV 时可能需要重复修改。
- 页脚 `document.lastModified` 显示的是浏览器得到的文件修改时间，未必是该栏目内容最后修改时间。

## 整改方案

1. **统一模板与资料。** 导航已经集中到 `_data/navigation.yml` 与 `_includes/site-navigation.html`，头像由首页的 `show_profile` 控制。后续将个人信息、学生、教学、服务、荣誉、经费整理为 `_data/` 下的 YAML 或 JSON，普通栏目由 Jekyll 直接读取。
2. **统一生成入口。** 保留 `citations.bib` 为论文唯一来源，共用会议与作者格式规则；提供一个 `scripts/generate_all.py`，生成网页论文列表和中英文 CV 片段。明确哪些文件可手工修改、哪些输出必须重新生成。
3. **独立处理联网工具。** DOI 查询、论文下载和引文抓取单独运行，不放进每次网站发布的必经流程。现有失配的抓取工作流应删除或补齐实现，并独立验证。
4. **再接入构建与部署。** PR 运行数据校验、论文生成和 Jekyll 构建；主分支通过检查后上传 `_site` 并部署到 GitHub Pages。生成内容直接进入构建产物，可避免机器人为每次生成结果额外回写提交。
5. **按需要加入 CV PDF 编译。** 初期生成简历源片段即可；等中英文 CV 模板稳定后，再加入明确的 PDF 编译步骤和字体依赖。

推荐流程：提交资料 → 数据校验 → 生成论文／简历片段 → Jekyll 构建 → GitHub Pages 发布。

GitHub 官方参考：[Using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。切换部署方式时还需要核对仓库 Settings → Pages 中的发布来源。

## 已完成的整改

资料已迁移到 `_data/*.json`，论文继续以 `citations.bib` 为唯一来源，会议简称与排序统一放在 `_data/venues.json`。`generate.ps1` 与 `scripts/generate_all.py` 是统一入口，会校验资料并生成各栏目、论文列表和中英文简历源文件。

旧的失配引文抓取工作流已替换为 `.github/workflows/site.yml`，执行测试、内容生成、Jekyll 构建、页面检查和 Pages 部署。当前已完成本地配置；远程启用需选择 GitHub Actions 发布来源并推送。简历 PDF 编译尚未加入。

具体资料位置和使用命令以根目录 `README.md` 为准。页脚修改时间仍使用原有 `document.lastModified`。
