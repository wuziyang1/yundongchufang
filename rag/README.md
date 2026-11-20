# 运动处方知识库（RAG）

把权威指南、标准、教材等 PDF 解析、切片、向量化后存入 Milvus，供运动处方生成时检索引用。
这里负责**离线入库**；在线检索在 `backend/app/services/literature.py`：处方生成时按学生情况检索相关切片，
以「权威资料摘录」注入处方 prompt，每条带出处（《文档标题》章节 第 N 页），作为知识卡片的补充。
Milvus 未启动时后端自动跳过检索，处方照常生成。检索自检：`cd backend && uv run python scripts/check_kb.py`。

## 目录结构

```
rag/
├── knowledge/              知识库 PDF 放这里（支持子目录；PDF 不入 Git，需自行放入）
├── milvus/                 Milvus 向量库部署：单容器，不需要 MinIO
│   ├── docker-compose.yml
│   ├── embedEtcd.yaml
│   └── volumes/            数据卷（不入 Git）
├── indexer/                解析 → 切片 → 向量化 → 入库
│   ├── main.py             命令行入口
│   ├── parser/             PDF 解析：文字层、本地 OCR、表格、分栏、页眉页脚、清洗规则
│   ├── chunker.py          切片
│   ├── embedding.py        向量化（默认智谱 embedding-3，1024 维）
│   └── milvus_api.py       Milvus 建库、建表、读写
├── doc_rules.toml          按文档定制的清洗规则（跳过页、补充标题写法、丢弃的章节、字形修正）
├── scripts/
│   ├── milvus_up.sh        启动 Milvus 并等待就绪
│   ├── milvus_down.sh      停止 Milvus（数据保留）
│   └── run_index.sh        运行入库程序
├── cache/                  OCR 结果缓存与切片预览（不入 Git）
├── pyproject.toml / uv.lock
└── .env.example
```

## 快速开始

在项目根目录执行：

```bash
bash rag/scripts/milvus_up.sh
```

```bash
cd rag && uv sync
```

```bash
bash rag/scripts/run_index.sh
```

- 嵌入密钥默认复用 `backend/.env` 里的 `ZHIPU_API_KEY`，无需额外配置；要单独配置时 `cp rag/.env.example rag/.env` 后修改。
- `uv sync` 从 PyPI 下载较慢时，可以按锁定的版本改从清华镜像安装：

```bash
cd rag && uv venv && uv export --frozen --no-hashes -o /tmp/rag-requirements.txt && uv pip install -r /tmp/rag-requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

## 入库命令

`run_index.sh` 的参数原样传给 `indexer.main`：

| 命令 | 作用 |
|------|------|
| `bash rag/scripts/run_index.sh` | 默认 backfill：只处理新增或内容有变化（MD5 不同）的文件；已从 knowledge 移走的文件，其向量一并删除 |
| `… --mode full-refresh` | 全部文件重新解析、重新入库 |
| `… --mode rebuild` | 清空集合后全量入库；更换嵌入模型或维度后必须用它 |
| `… --mode dry-run` | 只解析和切片，不调嵌入接口、不写库，每份文档的切片预览写到 `rag/cache/preview/` |
| `… --mode stats` | 查看库中每份文档的切片数 |
| `… --mode query --text "青少年每天应该运动多久"` | 入库后自检：向量检索最相近的切片 |
| `… --file 某文件.pdf` | 只处理指定文件，可重复指定 |

文档在库里的唯一标识是「相对 `knowledge/` 的路径」，改名或移动目录等同于删掉旧文档、新增一份。

## 处理流程

1. **判断页面类型**：文字层不足 50 字、且图片覆盖 30% 以上的页按扫描页处理，用本地 RapidOCR（PP-OCR 模型，onnxruntime 推理）识别，结果缓存在 `cache/ocr/`，重复入库不再识别；其余页直接读取 PDF 文字层。
2. **还原阅读顺序**：按单词取字后按「视觉行」聚合，上标引用（如「［1］」）、错位的标点回到原位；自动找出分栏之间的空白带，双栏、三栏、左窄右宽的期刊版式都按栏顺序读；小字号的脚注、作者单位排到本页正文之后。
3. **清理**：去掉在 3 页以上重复出现的页眉页脚、页码、目录条目，丢弃参考文献列表；另有以下规则（在解析阶段生效，重新入库时脏数据不会再回来）：
   - **前置页**：书籍开头 12 页内，出现两种以上出版信息（ISBN、CIP、定价、印张、责任编辑、编委会等）且几乎没有成句正文的页，以及开头 3 页里字很少、没有句号的封面页，整页跳过；
   - **期刊首页**：页面上有 DOI、通讯作者、Address correspondence 等论文标志时，「摘要 / ABSTRACT / 前言」以上的标题、作者、单位区丢弃；页面下部孤立短横线以下的小字号脚注，含基金项目、作者单位、邮箱等作者信息时丢弃；
   - **噪声行**：扫描页上不含汉字的短行（插图里识别出的「TEEEE」「A→ D → C」「20m」）、课程标准奇数页的「│ 前言 │」页眉；
   - **按文档的规则**：写在 `doc_rules.toml`，如指南的「指南整合过程 / 利益冲突 / 附表」、课程标准附录的空白评价表单与理论测试题整章丢弃，NSCA 文献 PDF 的字形映射错误（「×」被抽成「3」、「≥」被抽成「$」）修正。
4. **表格**：用 PyMuPDF 定位表格，合并单元格按「横向取左、纵向取上」补全，逐行展开为「表头：值；表头：值」，每个切片都带表题（没有表题时用前一行短标题或所在章节标题），例如：
   `表1-1 男生体重指数（BMI）单项评分表 / 等级：超重；单项得分：80；一年级：18.2~20.3；…`
   PyMuPDF 认不出的无竖线表格（三线表、期刊表格）按「同宽的一组横线」定位：表头文字给出列位置，行按内部横线或视觉行切分；表头左右重复的双栏表（身体活动内容 | MET | 身体活动内容 | MET）拆成一列。
5. **切片**：识别「第一篇 / 一、 / （一） / 问题N / 练习N： / 附录N / 前言」等标题，按章节切分，一个切片只属于一个章节；正文按段落累积到 600 折算字数，相邻切片重叠末尾约 80 字的整句；每个切片记录章节路径和起止页码。不足 50 折算字数的小切片：表注（「注：」「*」开头）并入表格，章节开头的一小段并入后面同章或下级章节的正文，孤立的一行短标题丢弃，其余带完整句子的（如器材章节的「功能：发展速度、灵敏、协调能力。」）保留。
6. **向量化与写入**：向量化时在正文前加上「《文档标题》章节路径」，切片脱离上下文也能被准确召回；全部向量算好后再替换该文档的旧数据，写入中途失败会回滚，不会留下半份文档。

## Milvus

- 镜像 `milvusdb/milvus:v2.5.16` 直接从 Docker Hub 拉取；单容器 standalone 模式，内嵌 etcd，`COMMON_STORAGETYPE=local` 让向量数据直接写本地磁盘，因此**不需要 MinIO 容器，也不监听 MinIO**。
- 端口：`19530`（gRPC / SDK）、`9091`（健康检查 `http://127.0.0.1:9091/healthz`）。
- 数据持久化在 `rag/milvus/volumes/`，停止容器不丢数据；彻底清空可先 `milvus_down.sh` 再删除该目录。
- 数据库 `yundongchufang`，集合 `knowledge_chunks`，向量索引 HNSW + COSINE。字段：

| 字段 | 说明 |
|------|------|
| `doc_name` / `doc_title` / `doc_path_name` | 文件名 / 文档标题 / 相对 knowledge 的路径（文档唯一标识） |
| `doc_type` / `doc_md5` / `doc_length` | 文件类型 / 内容 MD5（判断是否需要重新入库）/ 字节数 |
| `chunk_index` / `chunk_type` | 切片序号 / `text` 正文或 `table` 表格 |
| `section` | 章节路径，如「第一篇 体能素质篇 > 一、速度素质 > 练习3：…」 |
| `page_start` / `page_end` | 起止页码 |
| `parse_method` | `text` 文字层 / `ocr` 扫描识别 / `mixed` |
| `content` | 切片正文 |
| `content_vector` | 1024 维向量 |
| `embedding_model` | 生成向量用的模型 |

在线检索接入时，查询文本必须用同一个嵌入模型和维度（默认智谱 `embedding-3`、1024 维）。
