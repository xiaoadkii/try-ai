"""
RAG 文档解析与切片管道 → pgvector 入库 → 向量检索
==================================================
独立脚本 — 不依赖 LangChain / tiktoken / Chroma / 任何 RAG 框架。
纯 Python 原生字符串操作实现工业级 sliding-window chunking，
原生 SQL 操作 pgvector 完成向量入库与相似度检索。

依赖安装：
    uv pip install psycopg[binary] openai python-dotenv
    （psycopg 是 psycopg3，[binary] 免编译直接装预编译包）

.env 配置（新增项）：
    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
    EMBEDDING_MODEL=text-embedding-3-small   # DeepSeek 无 Embedding API，用 OpenAI

CLI 用法：
    python rag_pipeline.py                  # 默认：切片 QA 质检
    python rag_pipeline.py --ingest         # 切片 → Embedding → 写入 pgvector
    python rag_pipeline.py --search "练背注意什么"  # 向量相似度检索

对照 Java 心智模型：
   本脚本 ≈ Spring Batch 的一个完整 ItemProcessor/Writer Step：
      Reader     → MockDocumentReader（本文的长文本）
      Processor  → manual_chunk_text（切片算法）+ Embedding API
      Writer     → pgvector 批量 INSERT（≈ Spring JdbcTemplate.batchUpdate）
      Searcher   → 余弦相似度检索（≈ MyBatis + 自定义 SQL）
"""

import sys
import os
from typing import List, Tuple

# 可选依赖：仅在 --ingest / --search 时需要
try:
    from dotenv import load_dotenv
    from openai import OpenAI
    import psycopg
    HAS_PGVECTOR_DEPS = True
except ImportError:
    HAS_PGVECTOR_DEPS = False

# ============================================================================
# 1) 模拟长文本：《健身房私教专业减脂饮食与背部/核心训练规范》
# ============================================================================
# 约 1800 字，涵盖动作要领、饮食计算、训练周期、名词解释。
# 刻意包含数字、英文简称、公式，模拟真实文档的复杂性。

MOCK_DOCUMENT = """
健身房私教专业减脂饮食与背部核心训练规范 V3.2

第一章 总则与适用范围

本规范适用于体脂率 18%-30% 的男性学员及 22%-35% 的女性学员。
训练周期标准设定为 12 周，分为三个阶段：基础适应期（第 1-4 周）、
强化提升期（第 5-8 周）、冲刺塑形期（第 9-12 周）。
执行本规范前，学员需完成 InBody 体测并记录以下基础数据：
体重（kg）、骨骼肌量（kg）、体脂率（%）、基础代谢率 BMR（kcal/day）。

第二章 减脂饮食热量与宏量营养素计算公式

每位学员的每日总热量消耗 TDEE 由下式得出：
TDEE = BMR × 活动系数。
其中活动系数取值：久坐 1.2、轻度运动（每周 1-3 天）1.375、
中度运动（每周 3-5 天）1.55、重度运动（每周 6-7 天）1.725、
运动员级别 1.9。减脂期热量缺口设定为 300-500 kcal/day，
即每日目标摄入量 = TDEE - 缺口值。

宏量营养素分配标准如下：
蛋白质摄入量定为每公斤体重 1.6g-2.2g。
例如体重 75kg 学员，每日蛋白质目标 = 75 × 2.0 = 150g。
脂肪摄入不低于每公斤体重 0.8g，以防内分泌紊乱。
剩余热量由碳水化合物补足，优先选择低 GI 来源如燕麦、红薯、糙米。

关键提醒：训练前 90 分钟应摄入 30-40g 慢碳 + 15-20g 蛋白质，
训练后 60 分钟内补充快碳 40-50g + 乳清蛋白 25-30g（WPI 分离乳清蛋白吸收最快）。
全天饮水不低于每公斤体重 35ml，单次训练中每 15 分钟补水 150-200ml。

第三章 背部训练动作规范

第一节 坐姿划船（Seated Cable Row）

目标肌群：背阔肌、菱形肌、中下斜方肌、肱二头肌。

动作设置：使用低滑轮缆绳训练器，配装 V 形手柄。双脚踏于踏板，
膝微屈保持身体稳定。起始位置时躯干与地面呈 90 度直角，
肩胛骨前引但不含胸。

动作执行：呼气阶段，驱动肘部向后移动，将手柄拉向腹部脐上两指位置。
关键细节——先启动肩胛骨后缩，再屈肘拉动。切忌用腰部猛甩借力。
顶峰收缩保持 1.5 秒，充分挤压背阔肌外缘。
吸气阶段，缓慢放回手柄，让肩胛骨完全前引感受背部拉伸。
推荐负荷：12RM × 4 组，组间休息 60-75 秒。

常见错误纠正：错误一，躯干过度后仰超过 15 度，这会将拉力转移到竖脊肌
而非背阔肌。错误二，拉向胸口而非腹部，导致上斜方肌过度参与。
错误三，离心阶段速度过快（应控制在 2-3 秒），丧失肌肉微撕裂的增肌收益。

第二节 硬拉（Deadlift）—— 力量训练之王

目标肌群：竖脊肌、臀大肌、腘绳肌、斜方肌，次要激活背阔肌与核心稳定肌群。

动作设置：双脚与髋同宽站立于杠铃杆正上方，杠铃杆距胫骨约一拳距离。
采用正反手握法或正手握法加助力带，握距略宽于肩。

动作执行五步法：（1）脚掌扎根地面，胫骨轻触杠铃杆；
（2）臀部后推、躯干前倾，保持脊柱中立——想象从尾骨到颈椎是一条直线；
（3）深吸气，收紧腹内压如同腹部挨一拳的准备状态，俗称瓦式呼吸（Valsalva Maneuver）；
（4）双腿蹬地启动，将杠铃杆沿胫骨前侧垂直向上拉至髋关节完全伸展，
顶峰时臀部收紧但不过度后仰；（5）下放时先髋关节后移再膝关节弯曲，
杠铃杆紧贴身体下落。推荐负荷：8RM × 3 组，组间休息 90-120 秒。

安全红线：任何时候腰椎不得弯曲！如出现腰酸而非臀酸，说明竖脊肌代偿过度，
应立即降重并重新学习罗马尼亚硬拉作为退阶动作。

第四章 核心抗旋转与抗伸展训练

核心训练不是仰卧起坐。真正的核心功能是抗旋转（Anti-Rotation）
和抗伸展（Anti-Extension）。请掌握以下黄金动作：

帕罗夫推（Pallof Press）：将弹力带固定于身体侧面，双手握手柄置于胸口正前方，
双腿站距略宽于肩，核心收紧后双手将手柄直线推离胸口并保持 3 秒后回收。
此动作训练核心抗旋转能力，是预防腰间盘突出的第一功臣。

死虫式（Dead Bug）：仰卧，双手垂直上举，双膝屈曲 90 度抬至髋部正上方。
对侧手脚同时缓慢下放至离地 5cm 处，全程下背必须紧贴地面。
此动作建立腹横肌的分离控制能力，纠正骨盆前倾。

活虫进阶（Bird Dog）：四足跪姿，对侧手臂和腿同时伸展至与躯干平行，
保持脊柱不产生任何旋转或塌陷 3 秒后换边。进阶后可手握 2kg 哑铃增加负重。

第五章 周期化训练模板与监测指标

每周训练频率：背部训练每周 2 次（间隔至少 72 小时），核心训练每周 3 次，
可安排在任一训练日末尾或独立日进行。

进度追踪指标包括：每周体脂率变化（目标 -0.5%/周）、硬拉 1RM 增幅
（每 4 周测试一次，目标 +5kg/月）、坐姿划船 8RM 增幅（每 4 周测试一次，
目标 +2.5kg/月）。如连续两周体脂未下降，需核查饮食日志并调整热量缺口至上限。

附录 术语表

BMR：Basal Metabolic Rate，基础代谢率——人体维持生命所需最低热量。
TDEE：Total Daily Energy Expenditure，每日总热量消耗。
RM：Repetition Maximum，最大重复次数——8RM 即只能完成 8 次的最大重量。
WPI：Whey Protein Isolate，分离乳清蛋白——蛋白质含量 ≥90% 的乳清蛋白形态。
DOMS：Delayed Onset Muscle Soreness，延迟性肌肉酸痛——训练后 24-72 小时出现。
瓦式呼吸：Valsalva Maneuver——闭气收紧腹腔增加腹内压以保护脊柱的呼吸技术。

免责声明：本规范仅供认证私人教练参考使用。执行前应获取学员医疗许可。
如学员在训练中出现头晕、胸痛、关节锐痛等警示症状，应立即停止并就医评估。
"""


# ============================================================================
# 2) 纯手工切片函数 —— RAG Pipeline 的核心引擎
# ============================================================================
def manual_chunk_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    separator: str = "",
) -> List[str]:
    """
    工业级 Sliding-Window 文本切片算法。

    ★ 严格限制：
        - 零第三方依赖，只使用 Python 原生 str 操作
        - 不含 LangChain / tiktoken / nltk / spaCy
        - 不含任何正则（re 模块也没有用）

    ★ 算法原理（对照 Java）：
        这个函数的逻辑 ≈ Apache Flink 的 SlidingWindow 算子：
            window size  = chunk_size（窗口大小）
            slide step   = chunk_size - chunk_overlap（步长）
        每个窗口与前一窗口重叠 chunk_overlap 个字符。

        用指针语言描述：
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]     ← Python 切片，O(chunk_size) 拷贝
                yield chunk
                start = end - chunk_overlap ← ★ 回退 overlap 距离，不是向前跳全量

    ★ 临界情况处理：
        1) 最后一个 chunk 不足 chunk_size → 保留（不填充，不丢弃）
        2) 如果文本总长 < chunk_size → 返回单 chunk 列表
        3) text 为空字符串 → 返回空列表
        4) chunk_overlap >= chunk_size → 抛异常（逻辑矛盾）

    Args:
        text:          待切片的原始文本字符串
        chunk_size:    每个切片的最大字符数，默认 300
        chunk_overlap: 相邻切片的重叠字符数，默认 50
        separator:     保留参数（本次未使用），预留按分隔符切分的能力

    Returns:
        切片后的字符串列表，保证相邻元素存在 chunk_overlap 字符的重叠。
    """

    # ---- 参数校验 ----
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})，"
            f"否则指针不会前进，造成死循环"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须 > 0，当前为 {chunk_size}")

    text_length = len(text)

    # ---- 空文本 / 超短文本处理 ----
    if not text:
        return []
    if text_length <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0  # 当前窗口起始位置（字符索引）

    # ---- 滑动窗口主循环 ----
    # ★ 对照 Java：
    #    for (int start = 0; start < text.length(); start += chunkSize - chunkOverlap) {
    #        int end = Math.min(start + chunkSize, text.length());
    #        chunks.add(text.substring(start, end));
    #    }
    # 但 Python 的 while 循环更灵活，可以在循环体内做末尾判断。
    while start < text_length:
        # 计算当前窗口结束位置
        end = start + chunk_size

        # 从原始文本切出当前 chunk
        # Python 切片超出索引不会报错，自动截断到 len(text)
        chunk = text[start:end]
        chunks.append(chunk)

        # 如果已经切到文本末尾，退出循环
        if end >= text_length:
            break

        # ★ 关键：下一轮的起点 = 当前终点 - 重叠量
        # 假设 chunk_size=300, overlap=50，第一轮切 [0:300]，
        # 下一轮起点 = 300 - 50 = 250，即 [250:550]
        # 这样 [250:300] 这 50 个字符在第一块和第二块中都有！
        start = end - chunk_overlap

    return chunks


# ============================================================================
# 3) 质检可视化 —— 终端打印 + 重叠验证
# ============================================================================
def visualize_chunks(text: str, chunks: List[str], overlap: int):
    """
    格式化打印切片结果，并高亮验证重叠区。

    对于 Chunk N（N >= 2）：
        - 打印 Chunk N-1 的末尾 overlap 个字符
        - 打印 Chunk N 的开头 overlap 个字符
        - 肉眼对比二者是否完全一致
    """
    width = 70  # 终端打印宽度

    print("=" * width)
    print("  RAG Chunking Pipeline — 质检报告")
    print("=" * width)
    print(f"  原始文本总字数 : {len(text)} 字")
    print(f"  切片总数       : {len(chunks)} 个")
    print(f"  理论步长       : {300 - overlap} 字/步")
    print(f"  重叠量         : {overlap} 字")
    print("=" * width)

    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        size = len(chunk)

        # ---- 分割线 ----
        print(f"\n{'─' * width}")
        print(f"  📄 Chunk {chunk_num:02d}  (Size: {size} 字)")
        print(f"{'─' * width}")

        # ---- 打印当前 chunk 全文 ----
        print(chunk)

        # ============================================================
        # ★ 高级质检：Chunk N 与 Chunk N-1 的重叠区验证 (N >= 2)
        # ============================================================
        if i >= 1:
            prev_chunk = chunks[i - 1]

            # Chunk N-1 的末尾 overlap 字
            prev_tail = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk

            # Chunk N 的开头 overlap 字
            curr_head = chunk[:overlap] if len(chunk) >= overlap else chunk

            # 判断是否一致
            match = (prev_tail == curr_head)

            print(f"\n  {'🔍' if match else '❌'} 重叠区验证 (与 Chunk {chunk_num - 1:02d})：")
            print(f"  ┌─ Chunk {chunk_num - 1:02d} 末尾 {overlap} 字: \"{prev_tail}\"")
            print(f"  ├─ Chunk {chunk_num:02d} 开头 {overlap} 字: \"{curr_head}\"")
            print(f"  └─ 重叠一致性: {'✅ 通过' if match else '❌ 失败 — 算法有 Bug！'}")

    # ---- 全局统计 ----
    print(f"\n{'=' * width}")
    print(f"  质检完成。")
    # 计算总字符覆盖率：每个 chunk 贡献 (chunk_size - overlap) 净新增
    # 第 1 个贡献 chunk_size，后续每个贡献 (chunk_size - overlap)
    theoretical_new = 300 + (len(chunks) - 1) * (300 - overlap)
    print(f"  理论覆盖字数   : {theoretical_new}")
    print(f"  实际原始字数   : {len(text)}")
    print(f"{'=' * width}")


# ============================================================================
# 5) pgvector 入库管道 — Embedding + Bulk Insert
# ============================================================================

# ---- 全局配置（懒加载，仅在 --ingest / --search 时初始化）----

def _ensure_deps():
    """确保 pgvector 相关依赖已安装，否则给出友好提示"""
    if not HAS_PGVECTOR_DEPS:
        print("❌ 缺少 pgvector 相关依赖，请先安装：")
        print("   uv pip install psycopg[binary] openai python-dotenv")
        sys.exit(1)

def _init_pgvector_env():
    """初始化环境变量 + 数据库连接 + OpenAI 客户端"""
    _ensure_deps()
    load_dotenv()

    DATABASE_URL = os.getenv("DATABASE_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not DATABASE_URL:
        print("❌ 环境变量 DATABASE_URL 未设置，请在 .env 中配置")
        print("   示例: DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb")
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("❌ 环境变量 OPENAI_API_KEY 未设置，请在 .env 中配置")
        sys.exit(1)

    # PostgreSQL 连接（同步连接 — 脚本场景足够）
    # psycopg3 的 connect() 同时支持同步和异步，这里用同步
    conn = psycopg.connect(DATABASE_URL)

    # OpenAI 客户端（同步客户端即可，embedding 调用是同步的）
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return conn, client, EMBEDDING_MODEL


def create_table_if_not_exists(conn, vector_dim: int) -> None:
    """
    创建 fitness_knowledge 表 + HNSW 向量索引。

    ★ vector_dim 由首次 Embedding 调用后自动探测，不硬编码。
      text-embedding-3-small → 1536
      BAAI/bge-m3            → 1024
      text-embedding-3-large → 3072

    ★ 对照 Java：
       这里相当于一个 Flyway/Liquibase migration script。
       但 pgvector 的 DDL 包含自定义类型 vector({dim})，
       Java 里用 JDBC Statement.execute() 一样能跑，
       因为 pgvector 的 vector 类型是 PostgreSQL 原生扩展，
       对 JDBC 驱动透明 —— JDBC 看到的是 PGobject，不是标准 java.sql.Types。
    """
    with conn.cursor() as cur:
        # 1) 启用 pgvector 扩展（只需执行一次，幂等）
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # 2) 先检查表是否已存在，如果存在且维度不匹配 → 删掉重建
        #    生产环境应该用 migration（ALTER TABLE ... ALTER COLUMN TYPE），
        #    这里为了 demo 清晰直接 DROP
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'fitness_knowledge' AND column_name = 'embedding'
        """)
        existing = cur.fetchone()
        if existing:
            # pgvector 的 data_type 显示为 "USER-DEFINED"，拿不到具体维度
            # 用 pg_column_size 或直接查实际存储维度来判断
            cur.execute("SELECT id FROM fitness_knowledge LIMIT 1")
            if cur.fetchone():
                cur.execute("SELECT array_length(embedding::real[], 1) FROM fitness_knowledge LIMIT 1")
                stored_dim = cur.fetchone()[0]
                if stored_dim and stored_dim != vector_dim:
                    print(f"⚠️ 表已存在但维度不匹配（表:{stored_dim}D, 模型:{vector_dim}D），删除旧表重建...")
                    cur.execute("DROP TABLE IF EXISTS fitness_knowledge CASCADE")
                    conn.commit()

        # 3) 建表（如果不存在）
        #    vector({dim}) = {dim} 维浮点向量，维度由实际使用的 Embedding 模型决定
        #    metadata JSONB 存来源信息（文档名/页码/时间戳等）
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS fitness_knowledge (
                id          SERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                metadata    JSONB DEFAULT '{{}}'::jsonb,
                embedding   vector({vector_dim})
            )
        """)

        # 4) 建 HNSW 索引（pgvector 0.5.0+ 原生支持）
        #    HNSW = Hierarchical Navigable Small World
        #    vector_cosine_ops: 按余弦距离排序
        #    m = 16: 每层每个节点最多 16 个邻居（默认值）
        #    ef_construction = 64: 构建时的候选队列长度，越大越准但建索引越慢
        #
        #    ★ HNSW vs IVFFlat：
        #      - HNSW：基于图，查询快（微秒级），增量写入友好，内存占用较大
        #      - IVFFlat：基于聚类，需要先建索引再查（须有数据样本），内存更省
        #      生产环境体量 < 100万 条用 HNSW 足够
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_fitness_embedding_hnsw
            ON fitness_knowledge
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        conn.commit()
    print(f"✅ 表 fitness_knowledge (vector({vector_dim})) + HNSW 余弦距离索引已就绪")


def generate_embeddings(texts: List[str], client, model: str) -> List[List[float]]:
    """
    调用大模型 Embedding API，将文本列表转为向量列表。

    ★ 不同模型的输出维度不同：
      text-embedding-3-small  → 1536 维
      BAAI/bge-m3            → 1024 维
      text-embedding-3-large → 3072 维
      本函数不硬编码维度，由 API 返回值自动确定。

    ★ 对照 Java：
       这里相当于 RestTemplate.postForObject(embeddingApiUrl, texts, float[][].class)
       返回 List<float[]>。
    """
    response = client.embeddings.create(
        model=model,
        input=texts,  # ★ API 支持一次传数组，比逐条调快 N 倍
    )
    # response.data[] 按 input 顺序返回，每个 data[i].embedding 是 list[float]
    return [d.embedding for d in response.data]


def bulk_insert_chunks(
    conn,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: dict = None,
) -> int:
    """
    原生 SQL 批量写入 pgvector。

    参数：
        conn:       psycopg3 连接
        chunks:     文本切片列表
        embeddings: 对应的向量列表（顺序对齐）
        metadata:   公共元数据（如 {"source": "fitness_guide_v3.2"}），
                   每条记录都会写入此值

    ★ 关键：psycopg3 原生支持 Python list[float] → pgvector::vector 的自动转换！
       你只需要把 list[float] 作为参数传给 INSERT，驱动会搞定序列化。
       不需要手动转成 '[1.2,3.4,...]' 字符串。
       （详见文件末尾的 Java 对照）

    ★ 对照 Java：
       这里相当于 Spring JdbcTemplate.batchUpdate(sql, batchArgs)
    """
    if metadata is None:
        metadata = {"source": "fitness_guide_v3.2"}

    records = [
        (chunk, psycopg.types.json.Json(metadata), emb)
        for chunk, emb in zip(chunks, embeddings)
    ]

    with conn.cursor() as cur:
        # executemany = 单条 SQL，多组参数，一次网络往返
        cur.executemany(
            """
            INSERT INTO fitness_knowledge (content, metadata, embedding)
            VALUES (%s, %s, %s)
            """,
            records,
        )
        conn.commit()

    print(f"✅ 批量写入 {len(chunks)} 条记录到 fitness_knowledge")
    return len(chunks)


# ============================================================================
# 6) 向量检索 — 余弦相似度 TOP-K 查询
# ============================================================================

def search_similar(
    query: str,
    client,
    conn,
    embed_model: str,
    top_k: int = 2,
) -> List[Tuple[int, str, dict, float]]:
    """
    向量相似度检索 — RAG 的 "Retrieval" 环节。

    流程：
        1. 用户问题 → Embedding API → 向量 q
        2. PostgreSQL 内执行: SELECT ... ORDER BY embedding <=> q LIMIT top_k
        3. 返回最相关的 top_k 条文本

    算子解释：
        <=>  余弦距离（Cosine Distance）= 1 - cosine_similarity
        <->  欧几里得距离（L2 Distance）
        <#>  内积（Inner Product / 负点积）

        我们选 <=>，因为语义搜索中余弦相似度是事实标准。
        打印时用 1 - distance 转回相似度分数（0=无关, 1=完全相同）。

    ★ 对照 Java：
        MyBatis 里写 @Select("SELECT ... ORDER BY embedding <=> #{q}::vector LIMIT #{k}")
        List<FitnessKnowledge> selectTopK(@Param("q") String vectorStr, @Param("k") int k);
        但 vector 类型需要自定义 TypeHandler 把 float[] 转成 pgvector 格式。
    """
    # 1) 问题 → 向量
    embeddings = generate_embeddings([query], client, embed_model)
    query_vector = embeddings[0]  # list[float]，维度由模型决定

    # 2) 余弦相似度检索
    #    embedding <=> %s::vector  计算余弦距离（越小越相似）
    #    1 - (embedding <=> %s::vector)  转回相似度（越大越相似）
    #    ::vector 是显式类型转换，确保参数被当 vector 类型处理
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM fitness_knowledge
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        )
        rows = cur.fetchall()

    return rows


def print_search_results(query: str, rows, width: int = 70):
    """格式化打印检索结果"""
    print(f"\n{'=' * width}")
    print(f"  🔎 向量检索结果")
    print(f"{'=' * width}")
    print(f"  查询: \"{query}\"")
    print(f"{'─' * width}")

    if not rows:
        print("  (无结果 — 数据库为空或相似度太低)")
        return

    for i, (id_, content, meta, sim) in enumerate(rows, 1):
        # 截取前 150 字作为预览
        preview = content[:150].replace("\n", " ") + ("..." if len(content) > 150 else "")
        meta_str = str(meta).replace("{", "").replace("}", "") if meta else ""
        print(f"\n  🏆 Top-{i}  (相似度: {sim:.4f})  [id={id_}]")
        print(f"     元数据: {meta_str}")
        print(f"     内容  : {preview}")

    print(f"{'─' * width}")


# ============================================================================
# 4) 主入口（重构为 CLI 模式）
# ============================================================================
if __name__ == "__main__":
    doc = MOCK_DOCUMENT.strip()
    arg1 = sys.argv[1] if len(sys.argv) > 1 else ""

    # ── 模式 1: 仅打印原始文档 ──
    if arg1 == "--raw":
        print(f"原始文本（{len(doc)} 字）：")
        print(doc)
        sys.exit(0)

    # ── 模式 2: pgvector 入库管道 ──
    if arg1 == "--ingest":
        CHUNK_SIZE = 300
        CHUNK_OVERLAP = 50

        # 2.1 切片
        chunks = manual_chunk_text(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"📦 切片完成：{len(chunks)} 个 chunk（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

        # 2.2 初始化 pgvector 环境
        conn, openai_client, embed_model = _init_pgvector_env()
        try:
            # 2.3 ★ 先生成 Embedding，探测向量维度（不同模型维度不同）
            #      BAAI/bge-m3 → 1024, text-embedding-3-small → 1536
            print(f"🧮 正在为 {len(chunks)} 个 chunk 生成 Embedding（模型: {embed_model}）...")
            embeddings = generate_embeddings(chunks, openai_client, embed_model)
            vector_dim = len(embeddings[0])
            print(f"   实际向量维度: {vector_dim}D（由模型 {embed_model} 决定）")

            # 2.4 用探测到的维度建表 + 索引（维度不匹配时自动重建）
            create_table_if_not_exists(conn, vector_dim=vector_dim)

            # 2.5 批量写入 pgvector
            bulk_insert_chunks(conn, chunks, embeddings)
            print("🎉 RAG 入库完成！现在可以用 --search \"问题\" 来检索。")
        finally:
            conn.close()

        sys.exit(0)

    # ── 模式 3: 向量相似度检索 ──
    if arg1 == "--search":
        query = sys.argv[2] if len(sys.argv) > 2 else "我今天练背应该注意什么？"
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 2

        conn, openai_client, embed_model = _init_pgvector_env()
        try:
            rows = search_similar(query, openai_client, conn, embed_model, top_k=top_k)
            print_search_results(query, rows)
        finally:
            conn.close()

        sys.exit(0)

    # ── 默认模式: 切片 QA 质检（原有逻辑）──
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    chunks = manual_chunk_text(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    visualize_chunks(doc, chunks, CHUNK_OVERLAP)

    # 边界条件自测
    print("\n\n  🧪 边界条件自测")
    print("-" * 70)

    assert manual_chunk_text("") == [], "空文本应返回空列表"
    print("  ✅ 空文本 → []")

    short = "Hello World"
    assert manual_chunk_text(short, chunk_size=300) == [short], "短文本应返回单元素列表"
    print(f"  ✅ 短文本（{len(short)}字）→ [{short!r}]")

    try:
        manual_chunk_text("test", chunk_size=10, chunk_overlap=10)
        assert False, "应抛出 ValueError"
    except ValueError:
        print("  ✅ overlap >= chunk_size → ValueError（符合预期）")

    boundary = "0123456789"
    result = manual_chunk_text(boundary, chunk_size=3, chunk_overlap=2)
    expected = ["012", "123", "234", "345", "456", "567", "678", "789"]
    assert result == expected, f"边界条件失败: {result} != {expected}"
    print(f"  ✅ 步长=1 边界条件 → {result}")

    print("\n  💡 提示: 用 --ingest 入库 pgvector，用 --search 检索")


# ============================================================================
# ★ 架构师思维对照：工业级 RAG 文档解析方案
# ============================================================================
#
# 问题：面对 PDF（尤其双栏排版、表格、图片的硬核 PDF），Python vs Java 如何选？
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Python 主流开源解析方案（按场景推荐）                                │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  方案 A: PyMuPDF (fitz) — 瑞士军刀                                  │
# │  ─────────────────────────────────                                  │
# │  底层：C 写的 MuPDF 引擎，Python 只是 binding                        │
# │  能力：文本提取 / 布局分析 / 图片导出 / 表格检测 / 批注读取          │
# │  双栏处理：`page.get_text("blocks")` 返回每个文本块的                │
# │            (x0,y0,x1,y1) 坐标，你按 x 坐标排序就能拆栏               │
# │  表格：`page.find_tables()` 直接返回 Table 对象，                    │
# │        每个 cell 的文本和坐标一目了然                                │
# │  速度：比 PDFBox 快 5-10 倍（纯 C vs JVM）                          │
# │  License：AGPL（商用需购买 license）                                │
# │                                                                     │
# │  方案 B: pdfplumber — 表格王者                                       │
# │  ─────────────────────────────                                       │
# │  底层：基于 pdfminer.six，纯 Python（无 C 依赖）                     │
# │  强项：表格提取是开源界最准的。对有线表格（有边框线）                 │
# │        和有隐表格（无边框线但字符对齐）都能识别。                    │
# │  双栏处理：`page.extract_text(layout=True)` 保留空间排布，           │
# │            但不如 PyMuPDF 精确                                      │
# │  劣势：慢。大 PDF（100+ 页）可能要几秒到几十秒                       │
# │  License：MIT                                                       │
# │                                                                     │
# │  方案 C: Unstructured.io — RAG 专用管道                             │
# │  ─────────────────────────────────────                               │
# │  定位：不是单一的 PDF 解析器，而是"文档 → RAG-ready chunks"         │
# │        的全自动管道。输入 PDF/Word/HTML/PPT/Markdown，               │
# │        输出已经切好的 chunk 列表。                                   │
# │  内部：它底层仍然调用 PyMuPDF 或 pdfplumber（可配置），              │
# │        但它加了一层"版面理解"（Layout Parsing）——                   │
# │        用目标检测模型（YOLOX 微调版 Detectron2）自动找到：          │
# │        标题 / 正文 / 表格 / 图片 / 页眉页脚 / 列表。                │
# │  优势：表格输出为 HTML 字符串便于 LLM 理解，                         │
# │        图片自动调用 OCR，真正实现 "文档 → 文本" 一键完成             │
# │  劣势：重。依赖 PyTorch + 模型权重文件，首次加载需要下载模型         │
# │  License：Apache 2.0                                                │
# │                                                                     │
# │  方案 D: Marker (SOTA at Jul 2024) — 高质量 PDF → Markdown          │
# │  ───────────────────────────────────────────────────────             │
# │  定位：专门把 PDF 转成结构化 Markdown。                              │
# │  亮点：公式转 LaTeX、表格转 Markdown 表格、标题层级保留。           │
# │  底层：自己训练的深度学习版面分析模型 + 专门的公式检测 OCR。        │
# │  License：GPL（商用不友好）                                         │
# │                                                                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │  实战组合推荐（按你的 RAG 场景）：                                   │
# │                                                                     │
# │  场景 1 — 纯文本 PDF（论文、合同）:                                 │
# │    PyMuPDF 提取文本 → manual_chunk_text（本文函数）→ Embedding       │
# │                                                                     │
# │  场景 2 — 含表格的 PDF（财报、实验报告）:                            │
# │    pdfplumber 提取表格 → 表格转 Markdown/HTML → 拼入文本 → 切片     │
# │                                                                     │
# │  场景 3 — 复杂排版 PDF（杂志、教材、双栏论文）:                      │
# │    Unstructured.io 或 Marker → 自动版面分析 → 结构化 Markdown        │
# │    → 按标题层级做语义切片（semantic chunking，比固定字数的           │
# │       sliding window 更智能）                                       │
# │                                                                     │
# │  场景 4 — 图片扫描件（无文本层的 PDF）:                              │
# │    PyMuPDF 导出图片 → PaddleOCR / Tesseract OCR → 文本 → 切片        │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  为什么 Java 的 Apache PDFBox / POI 在 AI 时代不够好用了？           │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  原因 1 — 设计哲学落后一个时代                                       │
# │  ────────────────────────────                                        │
# │  PDFBox 和 POI 诞生于 2000 年代，它们的设计目标是：                  │
# │  "让 Java 程序能读写 Office / PDF 文件"。                            │
# │  注意：是"读写文件格式"，不是"理解文档内容"。                        │
# │                                                                     │
# │  而 AI/RAG 时代需要的是"版面理解"（Layout Understanding）——          │
# │  不是把 PDF 的每个字符按物理坐标吐给你就完了，                       │
# │  而是告诉你"这块是正文、这块是表格、这个表格的 headers 是什么、      │
# │  这两栏文字的阅读顺序是 A→B 而不是 A→C"。                           │
# │                                                                     │
# │  类比：PDFBox 像是 JDBC —— 给你原始的 ResultSet，                    │
# │        你自己写代码把行和列拼成业务对象。                             │
# │        PyMuPDF + Unstructured 像是 JPA/Hibernate ——                   │
# │        直接给你映射好的 Entity 对象（Title / Table / Image / Text）。 │
# │                                                                     │
# │  原因 2 — 双栏排版是 PDFBox 的阿克琉斯之踵                           │
# │  ────────────────────────────────────────────                         │
# │  PDFBox 的 TextStripper 按物理坐标顺序吐文本：                        │
# │    左栏第一行 → 右栏第一行 → 左栏第二行 → 右栏第二行...              │
# │  结果："The cat sat on the the mat Meanwhile in another part..."      │
# │  两栏内容交叉输出，彻底破坏阅读顺序。                                 │
# │                                                                     │
# │  要解决这个问题，你需要自己在 PDFBox 之上写 300+ 行的                │
# │  LayoutAnalyzer：获取每个字符的坐标 → 聚类成行 → 行聚类成栏          │
# │  → 按阅读顺序排序。而 PyMuPDF 的 `get_text("blocks")` 一行搞定。     │
# │                                                                     │
# │  原因 3 — 表格提取是后添加的补丁，不是原生能力                        │
# │  ────────────────────────────────────────────────────                  │
# │  PDFBox 没有内置的表格检测。你需要：                                  │
# │    1) 用 PDFBox 获取所有线条的坐标                                    │
# │    2) 自己写算法判断哪些矩形构成一个表格                               │
# │    3) 把每个 cell 的文本映射进去                                      │
# │    4) 处理合并单元格（colspan/rowspan）                               │
# │  这是 500 行起跳的工作量。                                            │
# │                                                                     │
# │  而 pdfplumber 的 `extract_table()` 一个方法调完。                    │
# │  PyMuPDF 的 `find_tables()` 一行调完。                               │
# │                                                                     │
# │  原因 4 — Python 生态的 AI 先发优势                                   │
# │  ─────────────────────────────────────                                │
# │  整个 AI 工具链（PyTorch / Transformers / LangChain /                 │
# │  LlamaIndex / Unstructured）全都在 Python 生态里。                    │
# │  你用 PDFBox 解析完，还是要把数据传给 Python 做 Embedding。           │
# │  中间多一层跨进程/跨语言的数据搬运（Java→Python），                   │
# │  增加延迟、序列化开销、运维复杂度。                                   │
# │                                                                     │
# │  Java 方案在 AI 时代的正确姿势：                                      │
# │    不要试图用 Java 解析 PDF。把 PDF 解析交给 Python 微服务           │
# │    （FastAPI + PyMuPDF），Java 只负责业务编排 ——                    │
# │    用户上传 PDF → 消息队列 → Python 解析服务 → 返回结构化 JSON       │
# │    → Java 存入 MySQL / 发 ES 索引 → 后续 RAG 检索。                  │
# │                                                                     │
# │  一句话总结：                                                        │
# │    PDFBox 是"文件格式 SDK"，PyMuPDF 是"文档理解引擎"。              │
# │    RAG 需要的是后者。                                                │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Java 思维对照：sliding-window chunking 的分布式化                   │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  你写的这个 manual_chunk_text 函数在单体中跑没问题。                 │
# │  但如果公司有一百万份 PDF 要做 RAG 切片：                             │
# │                                                                     │
# │    1) 文档解析层 — Python 微服务集群                                 │
# │       用 FastAPI + PyMuPDF/pdfplumber 部署成无状态服务，             │
# │       Kafka 消费上传事件，解析完把 Markdown 写回 OSS/MinIO           │
# │                                                                     │
# │    2) 切片层 — 可复用你的 manual_chunk_text 逻辑                     │
# │       但如果数据量大，用 Apache Spark 做分布式切片：                 │
# │         df = spark.read.text("oss://docs/*.md")                      │
# │         df.select(sliding_window_udf("text", 300, 50))               │
# │       Spark 的滑动窗口比你手动 for 循环更适合 100 万份文档           │
# │                                                                     │
# │    3) Embedding 层 — 独立扩容                                        │
# │       文本切片 → Embedding API（OpenAI/本地模型）→ 向量写入           │
# │       Milvus / Qdrant / Elasticsearch                                │
# │                                                                     │
# │    4) Java 业务层 — 你的主战场                                        │
# │       Spring Cloud Gateway → Controller → Service                    │
# │       → 调用向量数据库做相似检索 → 拼 prompt → 调 LLM                 │
# │       → 返回给前端                                                   │
# │                                                                     │
# │  此时 manual_chunk_text 的价值：                                      │
# │    它是你理解 RAG 管道核心算法的起点。                                │
# │    理解了它，你就理解了 LangChain RecursiveCharacterTextSplitter     │
# │    源码里 80% 在做的事情 —— 剩下 20% 是分词边界处理（               │
# │    按句号/换行符断句而非按固定字数截断，以保护语义完整性）。          │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Java 思维对照：Python list[float] vs JDBC 的 vector 类型传输差异    │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  这是你在生产环境中一定会遇到的坑。                                   │
# │                                                                     │
# │  ▎Python psycopg3 的做法（本文代码）                                 │
# │                                                                     │
# │    embedding = [0.0123, -0.0456, 0.0789, ...]  ← Python list[float] │
# │    cur.execute(                                                     │
# │        "INSERT INTO t (embedding) VALUES (%s)",                      │
# │        (embedding,)    ← ★ 直接把 list[float] 传进去！               │
# │    )                                                                │
# │                                                                     │
# │  psycopg3 的协议层做了什么：                                         │
# │    1) 检测到参数类型是 list                                          │
# │    2) 自动序列化为 PostgreSQL vector 的字符串格式：                   │
# │       "[0.0123, -0.0456, 0.0789, ...]"                              │
# │    3) 通过 binary protocol 发送（非 text protocol，性能更好）         │
# │    4) PostgreSQL 的 pgvector 扩展收到后解析为 vector 内部格式         │
# │                                                                     │
# │  整个过程对开发者透明 —— 你写 Python list，驱动帮你翻译。             │
# │                                                                     │
# │  ▎Java JDBC 的做法（对比）                                           │
# │                                                                     │
# │  JDBC 是 1997 年的标准，java.sql.Types 里没有 VECTOR 类型。          │
# │  所以 JDBC 驱动拿到 float[] 会懵掉：                                  │
# │                                                                     │
# │    float[] embedding = new float[1536];                              │
# │    PreparedStatement ps = conn.prepareStatement(                     │
# │        "INSERT INTO t (embedding) VALUES (?)"                        │
# │    );                                                               │
# │    ps.setObject(1, embedding);  // ❌ JDBC 不知道这是什么！           │
# │                                                                     │
# │  常见报错：                                                          │
# │    org.postgresql.util.PSQLException:                                │
# │      Can't infer the SQL type to use for an instance of float[]      │
# │                                                                     │
# │  三种解法：                                                          │
# │                                                                     │
# │  解法 1 — 手写字符串（最原始但最透明）：                              │
# │    String str = Arrays.toString(embedding); // [0.01, -0.04, ...]    │
# │    ps.setString(1, str);                                             │
# │    // SQL 里需要显式转换: ?::vector                                  │
# │    优点：不依赖任何库；缺点：快 10 万条时字符串拼接开销很大            │
# │                                                                     │
# │  解法 2 — pgvector-java 库（推荐）：                                  │
# │    // Maven: com.pgvector:pgvector                                   │
# │    PGvector vector = new PGvector(embedding);  // ★ 包装类            │
# │    ps.setObject(1, vector);                                         │
# │    pgvector-java 实现了 JDBC 的 SQLData 接口，                        │
# │    负责把 float[] ↔ pgvector 的二进制格式互转。                       │
# │    这是 psycopg3 自动做的事情，但 Java 需要显式引入第三方库。          │
# │                                                                     │
# │  解法 3 — Spring Data JDBC / JPA 自定义 TypeHandler：                │
# │    @Column(columnDefinition = "vector(1536)")                        │
# │    @Type(PGVectorType.class)  // 自定义类型处理器                     │
# │    private float[] embedding;                                        │
# │    JPA 在 flush 时自动调用 TypeHandler 序列化。                       │
# │    对业务层透明，但调试时容易摸不到底层发生了什么。                    │
# │                                                                     │
# │  ▎核心差异总结                                                       │
# │                                                                     │
# │  | 维度           | Python (psycopg3)     | Java (JDBC)              │
# │  |----------------|-----------------------|--------------------------|
# │  | 原生支持       | ✅ list[float] 自动转  | ❌ float[] 不被识别       │
# │  | 第三方库       | 不需要                 | 需要 pgvector-java       │
# │  | 协议           | PostgreSQL binary      | text (或 binary + lib)  │
# │  | 开发者认知负担 | 低 — 你写 Python 就行  | 中 — 必须理解为什么      |
# │  |                |                       |   JDBC 不认识你的数组    │
# │  | 批量写入性能   | executemany + binary   | batchUpdate + pgvector  │
# │  |                | 一次网络往返           | jar 一次网络往返         │
# │  | 调试透明度     | 中（驱动层面黑盒）     | 解法1最高（全是明文）    │
# │                                                                     │
# │  ▎为什么 Python 更适合做 Embedding 入库的胶水层                       │
# │                                                                     │
# │    1) Python 的 list[float] ↔ vector 零阻抗匹配。                    │
# │       OpenAI SDK 返回的就是 list[float]，psycopg3 吃的也是            │
# │       list[float] — 同一个数据结构从头传到尾，无序列化步骤。           │
# │                                                                     │
# │    2) Java 链路：float[] → JSON String → JDBC String →               │
# │       pgvector binary → PostgreSQL vector。                          │
# │       每一步都是转换开销 + 潜在精度损失（float → string → float）。    │
# │                                                                     │
# │    3) 实战建议：Embedding 入库的 Writer 用 Python 写，                │
# │       业务查询和 RAG 检索用 Java 调 PostgreSQL（读到的是              │
# │       similarity score 和 text，不需要解 vector）。                   │
# └─────────────────────────────────────────────────────────────────────┘
""" 文件尾注释结束 """
