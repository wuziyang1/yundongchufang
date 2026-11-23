# 运动处方知识库（RAG）

把权威指南、标准、教材等 PDF 解析、切片、向量化后存入 Milvus。**它是运动处方专业依据的唯一来源**：
处方生成时，体测成绩的评分、频率强度时间等决定所依据的资料、可以选用的动作，全部从这里取，
不再使用后端早期由 AI 整理的知识卡片和动作库。

这里负责**离线入库**；在线检索在 `backend/app/services/kb/`。知识库不可用时后端不生成处方。
检索自检：`cd backend && uv run python -u scripts/check_kb.py`。

### 入库的不只是文字

只把 PDF 切成段落存进去是不够的：处方要用的三种东西取法完全不同，入库时就要分开。

| 切片类型 | 内容 | 在线怎么用 |
|------|------|------|
| `text` / `table` | 正文段落、普通表格 | 按主题检索（向量 + 关键词两路召回，重排模型把关），作为「权威资料摘录」 |
| `lookup` | 查表数据：体质标准的评分表 | 整张取出，代码按性别、年级、成绩查表。不做相似度检索 —— 否则「怎么提高跳绳成绩」召回的全是评分表，模型自己读表还会读错行 |
| `exercise` | 动作条目：训练手册里的「练习N：…」，一个练习一条 | 带编号、名称、素质类别、器材、是否需要同伴，代码按学段和器材筛出候选，处方只能从中选 |

每份文档还要在 `doc_rules.toml` 登记**类别**与**适用学段**，随切片写入 Milvus，在线检索只取适用于该学生学段的资料
（例如高中课程标准不会用到小学生身上）。

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
│   ├── chunker.py          切片（含动作条目的组装、查表数据的标记）
│   ├── exercise.py         动作条目：从「练习N：…」抽取类别、器材、各段文字
│   ├── embedding.py        向量化（默认智谱 embedding-3，1024 维）
│   └── milvus_api.py       Milvus 建库、建表（向量索引 + BM25 关键词索引）、读写
├── doc_rules.toml          每份文档的身份（类别、适用学段、查表数据、动作条目）与清洗规则
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
- **从旧结构升级**（集合里没有 `stages`、`meta`、`content_sparse` 等字段）必须重建一次：`bash rag/scripts/run_index.sh --mode rebuild`。
  后端检测到旧结构会直接报「知识库不可用」并给出这条命令。
- `uv sync` 从 PyPI 下载较慢时，可以按锁定的版本改从清华镜像安装：

```bash
cd rag && uv venv && uv export --frozen --no-hashes -o /tmp/rag-requirements.txt && uv pip install -r /tmp/rag-requirements.txt --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

## 新增一份资料

1. 把 PDF 放进 `rag/knowledge/`。
2. 在 `rag/doc_rules.toml` 加一条 `[[doc]]`，至少写 `match` 和 `kind`；只适用于部分学段的写 `stages`：

   ```toml
   [[doc]]
   match = "义务教育体育与健康课程标准*.pdf"
   kind = "课程标准"
   stages = ["小学低年级", "小学高年级", "初中"]
   ```

   | 键 | 含义 |
   |------|------|
   | `kind` | 文档类别：标准 / 指南 / 课程标准 / 训练手册 / 立场声明 / 教材 / 文献 |
   | `stages` | 适用学段，取值「小学低年级」「小学高年级」「初中」「高中」，不写表示四个学段都适用 |
   | `lookup_tables` | 表题正则。命中的表格是查表数据（评分表），不参与相似度检索 |
   | `exercise_prefix` | 1 至 4 个大写字母。写了表示「练习N：…」按动作条目入库，动作编号 = 前缀 + 序号（如 `JK078`） |

3. `bash rag/scripts/run_index.sh --mode dry-run` 看切片预览（`rag/cache/preview/`），确认章节、表格、动作条目切得对。
4. `bash rag/scripts/run_index.sh` 入库。
5. `cd backend && uv run python -u scripts/check_kb.py --all`，看原先「未覆盖」的问题是否有了资料。

后端不认识具体的文件名：评分表靠 `lookup` 类型找，动作靠 `exercise` 类型找，学段靠 `stages` 过滤，所以新增资料不用改后端代码。
知识库指纹（各文档内容 MD5 的摘要）随每份处方留档，增删或替换任何一份资料指纹都会变。

## 入库命令

`run_index.sh` 的参数原样传给 `indexer.main`：

| 命令 | 作用 |
|------|------|
| `bash rag/scripts/run_index.sh` | 默认 backfill：只处理新增或内容有变化（MD5 不同）的文件；已从 knowledge 移走的文件，其向量一并删除 |
| `… --mode full-refresh` | 全部文件重新解析、重新入库（改了 `doc_rules.toml` 或解析、切片代码后用） |
| `… --mode rebuild` | 清空集合后全量入库；更换嵌入模型或维度、集合结构有变化后必须用它 |
| `… --mode dry-run` | 只解析和切片，不调嵌入接口、不写库，每份文档的切片预览写到 `rag/cache/preview/` |
| `… --mode stats` | 查看库中每份文档的类别、适用学段与各类切片数 |
| `… --mode query --text "青少年每天应该运动多久"` | 入库后自检：分别列出向量相似度与关键词（BM25）两路的检索结果 |
| `… --file 某文件.pdf` | 只处理指定文件，可重复指定 |

文档在库里的唯一标识是「相对 `knowledge/` 的路径」，改名或移动目录等同于删掉旧文档、新增一份。
注意 backfill 只看文件内容是否变化：只改了 `doc_rules.toml` 而 PDF 没变时，要用 `full-refresh` 才会生效。

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
6. **动作条目**（文档配了 `exercise_prefix`）：一个「练习N：名称」整理成一条，不与相邻小块合并；按原书小标题拆出功能、器材、起始姿势、练习步骤、指导提示、安全事项、拓展练习，并得出：
   - 素质类别：优先看所在章节（「一、力量练习」「（二）增加心肺耐力的练习」），章节没写明的看「练习功能 / 目标」里最先提到的素质；
   - 器材：「场地器材 / 练习器材」逐项取出并去掉数量；器械使用篇用小节名（「一、敏捷梯」）；「场地布置」是整句描述，只在其中找已出现过的器材名；
   - 是否需要同伴：双人、团队章节，或步骤里写了「2人」「辅助练习者」「面对面」等。「所有人围成一个圆圈」是课堂队形，不算。
   全部从原文抽取，不改写；扫描件识别丢的冒号、混进标题行的正文会修正回来。
7. **查表数据**（`lookup_tables`）：表题命中的表格类型记为 `lookup`。
8. **向量化与写入**：向量化文本是「《文档标题》章节路径 + 正文」，切片脱离上下文也能被准确召回；同一段文本同时交给 Milvus 内置的 BM25 建关键词索引（jieba 分词，英文转小写并取词干）。全部向量算好后再替换该文档的旧数据，写入中途失败会回滚，不会留下半份文档。

## Milvus

- 镜像 `milvusdb/milvus:v2.5.16` 直接从 Docker Hub 拉取；单容器 standalone 模式，内嵌 etcd，`COMMON_STORAGETYPE=local` 让向量数据直接写本地磁盘，因此**不需要 MinIO 容器，也不监听 MinIO**。
- 端口：`19530`（gRPC / SDK）、`9091`（健康检查 `http://127.0.0.1:9091/healthz`）。
- 数据持久化在 `rag/milvus/volumes/`，停止容器不丢数据；彻底清空可先 `milvus_down.sh` 再删除该目录。
- 数据库 `yundongchufang`，集合 `knowledge_chunks`，向量索引 HNSW + COSINE，关键词索引 BM25。字段：

| 字段 | 说明 |
|------|------|
| `doc_name` / `doc_title` / `doc_path_name` | 文件名 / 文档标题 / 相对 knowledge 的路径（文档唯一标识） |
| `doc_type` / `doc_md5` / `doc_length` | 文件类型 / 内容 MD5（判断是否需要重新入库，也是知识库指纹的来源）/ 字节数 |
| `doc_kind` / `stages` | 文档类别 / 适用学段（数组），来自 `doc_rules.toml` |
| `chunk_index` / `chunk_type` | 切片序号 / `text` 正文、`table` 表格、`lookup` 查表数据、`exercise` 动作条目 |
| `section` | 章节路径，如「第一篇体能素质篇 > 一、力量练习 > 练习2：平板支撑」 |
| `page_start` / `page_end` | 起止页码 |
| `parse_method` | `text` 文字层 / `ocr` 扫描识别 / `mixed` |
| `content` | 切片正文 |
| `meta` | 动作条目的结构化信息（JSON）：`code` 编号、`name`、`category` 素质类别、`group` 单人/多人、`equipment`、`fields` 各段文字；其余切片为空对象 |
| `search_text` | 「《文档标题》章节路径 + 正文」，向量化与关键词检索共用 |
| `content_vector` | 1024 维向量 |
| `content_sparse` | BM25 稀疏向量，由 Milvus 根据 `search_text` 自动生成 |
| `embedding_model` | 生成向量用的模型 |

在线检索时，查询文本必须用同一个嵌入模型和维度（默认智谱 `embedding-3`、1024 维）。
