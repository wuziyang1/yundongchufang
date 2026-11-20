# 运动处方知识库（RAG）

把权威指南、标准、教材等 PDF 解析、切片、向量化后存入 Milvus，供运动处方生成时检索引用。
目前完成的是**离线入库**部分；在线检索尚未接入 `backend`（后端现有的 `app/services/rag.py` 仍按原方式工作，不受影响）。

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
│   ├── parser/             PDF 解析：文字层、本地 OCR、表格、分栏、页眉页脚
│   ├── chunker.py          切片
│   ├── embedding.py        向量化（默认智谱 embedding-3，1024 维）
│   └── milvus_api.py       Milvus 建库、建表、读写
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
3. **清理**：去掉在 3 页以上重复出现的页眉页脚、页码、目录条目，丢弃参考文献列表。
4. **表格**：用 PyMuPDF 定位表格，合并单元格按「横向取左、纵向取上」补全，逐行展开为「表头：值；表头：值」，每个切片都带表题，例如：
   `表1-1 男生体重指数（BMI）单项评分表 / 等级：超重；单项得分：80；一年级：18.2~20.3；…`
5. **切片**：识别「第一篇 / 一、 / （一） / 问题N / 练习N：」等标题，按章节切分，一个切片只属于一个章节；正文按段落累积到 600 折算字数，相邻切片重叠末尾约 80 字的整句；每个切片记录章节路径和起止页码。
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
