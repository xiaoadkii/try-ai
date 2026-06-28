"""
最简 AI Web 后端服务 —— 给 Java 后端工程师的入门 Demo
=========================================================
对照 Spring Boot 帮你建立心智模型：
  - FastAPI 本身  ≈ Spring Boot (Web 部分)
  - uvicorn      ≈ 内嵌的 Tomcat / Netty (ASGI 服务器)
  - 装饰器 @app.post ≈ @RestController + @PostMapping
  - Pydantic 模型 ≈ Java 的 DTO + Bean Validation (JSR-303)
  - async def    ≈ Spring WebFlux 里的响应式方法 (Mono/Flux)，但写法是同步风格

新增功能：
  - /v1/chat       非流式一问一答（保留）
  - /v1/chat/stream 流式输出 + 内存多轮对话记忆（无框架，纯原生 Python）
"""

# ---------------------------------------------------------------------------
# 1) 导入依赖
# ---------------------------------------------------------------------------
import os
import json
import uuid
import asyncio
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# pgvector 驱动 — 仅在 RAG 端点被调用时才需要
try:
    import psycopg
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

# ---------------------------------------------------------------------------
# 2) 加载配置
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("FLASH_NAME", "gpt-4o-mini")

# pgvector / RAG 配置
DATABASE_URL = os.getenv("DATABASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

if not OPENAI_API_KEY:
    raise RuntimeError("环境变量 OPENAI_API_KEY 未设置，请在 .env 中配置")

# ---------------------------------------------------------------------------
# 3) 全局对象
# ---------------------------------------------------------------------------
app = FastAPI(title="My First AI Backend", version="0.2.0")

# ★ CORS 中间件 — 允许前端页面跨域调用 API
# 对照 Java：相当于 Spring Security 的 CorsConfiguration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 开发阶段放行所有来源
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局复用一个异步客户端（线程安全 + 连接池复用，勿每次请求 new）
client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ============================================================================
# ★ 本地业务函数（模拟异步数据库查询）
# ============================================================================
# 在真实微服务架构中，这些函数存在于独立的 Service 层或远程 RPC 服务中。
# 大模型不直接调用它们 — 大模型只输出"我想调用哪个函数 + 什么参数"，
# 由我们的胶水层（本文件中的 handshake 逻辑）负责解析并真正执行。
#
# 对照 Java：
#   这个函数 ≈ Spring @Service 中的一个方法，
#   比如 UserFitnessService.getUserFitnessStats(userId, metric)
#   它背后可能是 MyBatis 查 MySQL，或是 Feign 调用另一个微服务。
async def get_user_fitness_stats(user_id: str, metric: str) -> dict:
    """
    模拟从 MySQL 数据库查询用户的健身数据。

    真实场景中这里会是：
      async def get_user_fitness_stats(user_id: str, metric: str) -> dict:
          async with db_pool.acquire() as conn:
              row = await conn.fetchrow(
                  "SELECT * FROM fitness_stats WHERE user_id=$1 AND metric=$2",
                  user_id, metric
              )
              return dict(row)

    Args:
        user_id: 用户唯一标识
        metric:  指标类型 — "protein"（蛋白质摄入）或 "workout_days"（锻炼天数）

    Returns:
        查询结果字典，包含 user_id / metric / value / status
    """
    # 模拟数据库网络延迟（50~150ms）
    await asyncio.sleep(0.1)

    if metric == "protein":
        return {
            "user_id": user_id,
            "metric": "protein",
            "value": "60g",
            "status": "今日还差 20g 达标",
        }
    elif metric == "workout_days":
        return {
            "user_id": user_id,
            "metric": "workout_days",
            "value": "3天",
            "status": "本周已练 3 天，完成度 60%",
        }
    return {"error": f"未找到指标 '{metric}'，支持: protein, workout_days"}


# ============================================================================
# ★ 工具声明（Tools Definition — JSON Schema 格式）
# ============================================================================
# 这是 OpenAI / DeepSeek Function Calling 协议的核心"接口描述语言"。
# 大模型通过阅读这段 JSON Schema 来理解：
#   1. 有哪些函数可用（name / description）
#   2. 什么时候该调哪个函数（description 的语义引导）
#   3. 调用时需要传什么参数（parameters — 标准 JSON Schema 格式）
#
# 关键规则：
#   - name: 必须与本地函数名完全一致（胶水层用这个名字做 dispatch）
#   - description: 越清晰越好！大模型靠它判断"用户这句话该不该调这个函数"
#   - parameters: 严格 JSON Schema（type / properties / required / enum）
#   - enum 约束：限制大模型只能传合法值，防止它脑补出 "weight_loss" 之类的幻觉
#
# 对照 Java：
#   这个 tools 列表 ≈ 你写 Feign 接口时定义的 @FeignClient + 方法签名
#   大模型就是通过这个 Schema 来"反射"调用你的服务
TOOLS = [
    {
        "type": "function",  # 固定值，OpenAI 协议要求
        "function": {
            "name": "get_user_fitness_stats",  # ★ 与本地函数名严格一致
            "description": (
                "查询指定用户的健身相关数据。"
                "当用户询问以下问题时，应调用此函数："
                "- 蛋白质摄入情况（吃了多少蛋白质、是否达标）"
                "- 锻炼天数（这周/最近练了几天）"
                "- 健身数据、运动统计"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户的唯一标识 ID",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["protein", "workout_days"],  # ★ 枚举约束，防幻觉
                        "description": (
                            "要查询的指标类型。"
                            "protein: 蛋白质摄入量；"
                            "workout_days: 锻炼天数"
                        ),
                    },
                },
                "required": ["user_id", "metric"],  # ★ 必填字段
            },
        },
    }
]

# ============================================================================
# ★ 工具调度表（Tool Dispatch Table）
# ============================================================================
# 用字典做函数名 → 异步函数的映射，避免写 if-elif 链。
# 新增工具只需：1) 写函数 2) 在 TOOLS 中声明 3) 在此注册
#
# 对照 Java：
#   这个 dispatch 表 ≈ Spring 的 ApplicationContext.getBean(name) 动态获取 Bean，
#   或者一个 Map<String, Function> 的函数注册中心。
#   如果是远程调用，这里就是 FeignClient 的动态路由表。
TOOL_DISPATCH: Dict[str, callable] = {
    "get_user_fitness_stats": get_user_fitness_stats,
}


# ============================================================================
# ★ RAG 向量检索层 — pgvector 对接
# ============================================================================
# 从 rag_pipeline.py 的核心逻辑提取，封装为可复用的函数。
# 生产环境建议：
#   - pgvector 连接用 psycopg_pool.AsyncConnectionPool（连接池）
#   - Embedding 调用用 asyncio.Semaphore 限流（API 有 RPM 上限）
#   - 检索结果用 lru_cache 缓存（高频问题避免重复查向量库）

def _check_pgvector():
    """确保 pgvector 驱动已安装"""
    if not HAS_PGVECTOR:
        raise HTTPException(
            status_code=500,
            detail="pgvector 驱动未安装，请执行: uv pip install psycopg[binary]"
        )
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="环境变量 DATABASE_URL 未设置，请在 .env 中配置 pgvector 连接"
        )


def _get_pg_connection():
    """
    获取 pgvector 同步连接（每次请求新建，用完关闭）。

    ★ 对照 Java：
       这里相当于 HikariCP 的 dataSource.getConnection()。
       但 Python 没有内置连接池（psycopg3 的连接池在 psycopg_pool 扩展包中），
       生产环境应改用 psycopg_pool.AsyncConnectionPool 避免频繁建连/断连开销。
    """
    _check_pgvector()
    return psycopg.connect(DATABASE_URL)


async def rag_retrieve(
    query: str,
    top_k: int = 2,
) -> Tuple[List[List[float]], List[dict]]:
    """
    RAG 检索核心 — 用户问题 → Embedding → pgvector 余弦相似度 TOP-K。

    ★ async 的原因：
       1) client.embeddings.create() 是异步的（网络 I/O）
       2) pgvector 查询目前用同步 psycopg，包在 asyncio.to_thread() 里，
          不阻塞事件循环。真正的异步方案用 psycopg.AsyncConnection。

    返回:
        (query_vector, rows)
        rows 每条: {"id": int, "content": str, "metadata": dict, "similarity": float}

    ★ 对照 Java（Spring WebFlux）：
       这里相当于：
         Mono<float[]> embedMono = webClient.post(embedApi, query).bodyToMono(float[].class);
         Mono<List<DocChunk>> dbMono = r2dbcClient.sql("SELECT ... <=> ...").fetch().all();
         return embedMono.flatMap(vec -> dbMono);
       但 Python 的 await 写起来更像同步代码，没有 Mono/flatMap 的嵌套地狱。
    """
    # Step 1: 问题 → 向量（异步，网络 I/O）
    embed_response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_vector: List[float] = embed_response.data[0].embedding

    # Step 2: pgvector 余弦相似度检索（同步 DB 查询，扔进线程池避免阻塞事件循环）
    #    <=> = 余弦距离算子，值越小越相似
    #    1 - distance = 相似度分数（1.0 = 完全相同，0.0 = 完全无关）
    def _db_search():
        conn = _get_pg_connection()
        try:
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
        finally:
            conn.close()

    rows = await asyncio.to_thread(_db_search)

    # 组装结果
    results = [
        {
            "id": row[0],
            "content": row[1],
            "metadata": row[2] if isinstance(row[2], dict) else {},
            "similarity": round(float(row[3]), 4),
        }
        for row in rows
    ]
    return query_vector, results


def build_rag_messages(
    user_prompt: str,
    retrieved_chunks: List[dict],
    history: List[dict],
) -> List[dict]:
    """
    ★ 核心拼装：将检索到的参考资料作为 system 指令注入 messages。

    策略：在 system prompt 最前端插入参考资料上下文，强制大模型"看着小抄回答"。

    生成的 messages 结构：
        [
            {role: "system", content: "严格基于以下参考资料回答..."},
            {role: "system", content: "<原始 system prompt>"},   ← 如果有
            ...历史消息...,
            {role: "user", content: "<用户当前问题>"},
        ]

    ★ 为什么用 system 角色而不是 user？
       system 的优先级最高，大模型会把 system 内容当"铁律"执行。
       如果放在 user 消息里，大模型可能选择性忽略。

    ★ 对照 Java（Prompt 模板引擎）：
       Spring AI 里用 StringTemplate 或 Mustache 做类似的事：
         String prompt = templateEngine.render("rag-prompt.txt", Map.of(
             "context", joinChunks(chunks),
             "query", userPrompt
         ));
       但手写 f-string 更直观，适合小规模模板。
    """
    if not retrieved_chunks:
        # 无检索结果 → 正常对话，不做 RAG 增强
        return history + [{"role": "user", "content": user_prompt}]

    # 拼接参考资料
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        sim = chunk["similarity"]
        content = chunk["content"].strip()
        context_parts.append(f"[参考资料 {i}]（相似度: {sim:.2%}）\n{content}")

    context_block = "\n\n---\n\n".join(context_parts)

    # 构建 RAG system 指令
    rag_system_prompt = (
        "你是一个严格基于参考资料的 AI 助手。请遵守以下铁律：\n"
        "1. 优先使用下方【参考资料】中的信息回答用户问题。\n"
        "2. 如果参考资料中明确包含了答案，请直接引用并注明来源（如「根据参考资料1」）。\n"
        "3. 如果参考资料未提及用户所问的内容，请明确回答："
        "'抱歉，当前知识库中未找到相关信息，我无法编造答案。'\n"
        "4. 严禁在参考资料之外自行编造事实、数据或建议。\n\n"
        f"【参考资料】\n{context_block}\n\n"
        f"【用户问题】{user_prompt}"
    )

    # 组装 messages：RAG system 指令放在最前面
    messages = [{"role": "system", "content": rag_system_prompt}]

    # 如果历史中有原始 system prompt，追加到后面（次优先级）
    for msg in history:
        if msg.get("role") == "system":
            messages.append(msg)
            break  # 只保留第一个原始 system prompt

    # 追加历史（跳过 system，已处理）
    for msg in history:
        if msg.get("role") != "system":
            messages.append(msg)

    # 追加当前用户消息
    messages.append({"role": "user", "content": user_prompt})

    return messages


# ============================================================================
# ★ 内存多轮对话历史存储（纯 dict，无任何框架依赖）
# ============================================================================
# 类比 Java 的 ConcurrentHashMap<String, List<ChatMessage>>。
# Python 的 dict 在 CPython 中受 GIL 保护，单个操作是原子的，
# 但并发读写同一个 session 时仍建议加 asyncio.Lock（见下方生成器内部）。
#
# 存储结构示例（messages 与 OpenAI Chat Completions API 格式一致）：
# {
#   "session-abc": [
#       {"role": "system", "content": "你是一个有帮助的助手"},
#       {"role": "user", "content": "我叫张三"},
#       {"role": "assistant", "content": "你好张三！"},
#       {"role": "user", "content": "我叫什么？"},  ← 新请求追加在这里
#       # ... 大模型回复在流式结束后追加
#   ],
#   "session-xyz": [...],
# }
conversations: Dict[str, List[dict]] = {}

# 系统提示词（所有会话共享），可为空 —— 空列表即不设 system prompt
DEFAULT_SYSTEM_PROMPT = "你是一个有帮助的AI助手，请用中文回复。"


# ---------------------------------------------------------------------------
# 4) 请求 / 响应 DTO
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """
    非流式请求体（保留兼容）
    """
    prompt: str = Field(..., min_length=1, description="用户输入的提示词")


class ChatResponse(BaseModel):
    reply: str
    model: str


class StreamChatRequest(BaseModel):
    """
    流式 + 多轮对话请求体

    session_id 可选：不传则自动生成一个，前端收到后应缓存起来，
    后续请求带上同一个 session_id 就能延续对话。
    """
    prompt: str = Field(..., min_length=1, description="用户输入的提示词")
    session_id: str = Field(
        default="",
        description="会话 ID，不传则自动创建新会话。首次调用后请保存返回的 session_id",
    )


# ---------------------------------------------------------------------------
# 5) 路由：POST /v1/chat（非流式，保留作对比）
# ---------------------------------------------------------------------------
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """一问一答，无历史，非流式。"""
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": req.prompt}],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用大模型失败: {e}")

    reply_text = completion.choices[0].message.content or ""
    return ChatResponse(reply=reply_text, model=MODEL_NAME)


# ---------------------------------------------------------------------------
# 6) 路由：POST /v1/chat/stream（流式 + 多轮记忆）
# ---------------------------------------------------------------------------
@app.post("/v1/chat/stream")
async def chat_stream(req: StreamChatRequest):
    """
    流式对话接口 —— 打字机效果 + 内存多轮记忆。

    ★ 核心流程（5 步）：
    1. 解析或创建 session_id
    2. 从内存取出历史 → 追加当前 user 消息
    3. 调用 SDK stream=True → 用 async generator + yield 逐块推给客户端（SSE 协议）
    4. 在流式过程中，本地拼接 assistant 完整回复
    5. 流结束后，把完整回复追加回历史

    ★ 对照 Java：
      这个 async generator ≈ Java 里返回 Flux<ServerSentEvent<String>> 的 Controller。
      但 Python 的 yield 比 Java 的 Iterator/Stream 更底层，详见下方注释。
    """
    # ------ 6.1 解析 session_id ------
    # 前端没传或传空字符串 → 自动建新会话
    session_id = req.session_id or str(uuid.uuid4())

    # 取历史，没有就初始化一个带 system prompt 的空列表
    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]

    # ------ 6.2 取出历史（引用！不是拷贝）------
    # 这里拿到的是 conversations[session_id] 的引用，
    # 后续对流结束后直接 append 到同一个列表，无需回写
    history: List[dict] = conversations[session_id]

    # 把当前用户消息追加到历史
    history.append({"role": "user", "content": req.prompt})

    # ------ 6.3 定义异步生成器（核心中的核心）------
    # 这是 Python 最强大的特性之一：async generator。
    #
    # 写法：async def + yield
    # 作用：每产出 (yield) 一个值，调用方就能立刻拿到，不必等函数执行完毕。
    #       产生值的间隙，事件循环可以处理其他请求 → 高并发。
    #
    # ★ 对照 Java（重要！）：
    #
    #   Python `yield`           Java `Iterator<T>` / `Stream<T>`
    #   ─────────────────────    ─────────────────────────────────
    #   本质：协程挂起点          本质：接口 / 管道操作符
    #   生成器函数体暂停在 yield    Iterator 的 next() 是普通方法调用，
    #   处，调用方取走值后，函数   执行完就返回；不保留调用栈。
    #   从 yield 下一行继续执行。
    #
    #   并发场景下的核心差异：
    #   - Python async generator：yield 时释放事件循环给其他协程，
    #     不阻塞线程。单线程内通过 "协作式多任务" 实现并发。
    #     → 类比 Java 虚拟线程 (Loom) 的 yield，或 Kotlin 的 suspend
    #
    #   - Java Iterator：next() 是同步阻塞调用，调用线程被占用。
    #     要并发必须开多个线程（或线程池），存在上下文切换开销。
    #
    #   - Java Stream：链式惰性操作，但背后仍是 Iterator，
    #     不解决并发问题。parallel() 用 ForkJoinPool 多线程。
    #
    #   - Reactor Flux：真正对标 Python async generator！
    #     Flux.create(sink -> { sink.next(data); }) 也是事件循环 + 背压，
    #     但写起来比 yield 啰嗦得多。
    #
    #   一句话总结：
    #     yield = "我暂停一下，你先用这个值，我一会再接着跑"
    #     Iterator = "你调用我，我给一个值，给完我就没上下文了"
    #     Java Stream = "我有一堆数据，你把操作套在我身上，我帮你遍历"
    #     并发友好度：yield ≈ Flux > Stream ≈ Iterator

    async def generate():
        """
        异步生成器 — StreamingResponse 会把它当作事件源逐块消费。

        SSE (Server-Sent Events) 协议格式：
          每行以 "data: " 开头，以 "\n\n" 结尾。
          空行是事件分隔符，浏览器 EventSource API 靠它切分事件。
        """
        try:
            # ---- 调用 SDK，开启流式模式 ----
            # stream=True 后，.create() 返回的不是普通 response 对象，
            # 而是一个 async iterator（AsyncStream），必须 async for 消费。
            #
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,        # ★ 把全部历史消息发给大模型
                stream=True,             # ★ 关键开关：开启流式
                # 流式模式下 max_tokens 建议显式设，防止无限输出
                max_tokens=2048,
            )

            # ---- 逐块消费流 ----
            # Python 的 async for ≈ Java 的 for-each，但每次迭代会 await
            # 异步等待下一个 chunk 到达（网络 I/O）而不阻塞事件循环。
            #
            # chunk 结构（OpenAI 兼容协议）：
            #   chunk.choices[0].delta.content → 本次增量文本（可能为 None/空）
            #   chunk.choices[0].finish_reason → 结束原因（None=继续, "stop"=正常结束）
            full_reply: str = ""  # ★ 拼接完整回复，流结束后存入历史

            async for chunk in stream:
                # delta 是这次推送的增量（非累积），类似键盘每敲一个字推送那个字
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # content 是增量文本片段，可能为 None（只有 role 信息等）
                content = delta.content
                if content:
                    full_reply += content  # ★ 拼回完整回复，用于存历史

                    # ---- yield 输出（SSE 格式）----
                    # ★ yield 是 Python 生成器的灵魂：
                    #   1. 把 content 包装成 SSE 字符串推给客户端
                    #   2. 函数在此 "挂起"，事件循环去处理其他协程
                    #   3. 等 StreamingResponse 取走这个值后，从下一行恢复执行
                    #
                    # 用 json.dumps 保证特殊字符（换行、引号）不出问题
                    sse_data = json.dumps(
                        {"delta": content, "session_id": session_id},
                        ensure_ascii=False,
                    )
                    yield f"data: {sse_data}\n\n"

                # 检查是否结束（finish_reason 在 choice 层级，不在 delta 上）
                # choice.finish_reason：None=进行中 / "stop"=正常结束 / "length"=达到 max_tokens
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                if finish_reason:
                    break

            # ---- 流结束，处理拼接好的完整回复 ----
            # ★ 关键步骤：把 assistant 回复追加到历史记录
            # 因为 history 是 conversations[session_id] 的引用，
            # 这里 append 直接修改内存中的历史列表，不需要回写 dict。
            if full_reply:
                history.append({"role": "assistant", "content": full_reply})

            # 发送结束事件（SSE 标准做法，方便前端知道流已结束）
            yield "data: [DONE]\n\n"

        except Exception as e:
            # 异常时也要通知客户端
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    # ------ 6.4 返回 StreamingResponse ------
    # StreamingResponse 接收一个生成器，边读边推，不等生成器执行完毕。
    # media_type="text/event-stream" 告诉浏览器这是 SSE 流。
    #
    # ★ 对照 Java：
    #   StreamingResponse ≈ ResponseEntity<Flux<ServerSentEvent<String>>>
    #   但 Python 不需要泛型体操，生成器直接返回字符串即可。
    #
    # headers 中加 session_id 方便前端首次调用时获取
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,  # 前端可以从此头获取 session_id
        },
    )


# ============================================================================
# ★ 7) 核心大戏：POST /v1/chat/fc — 原生 Function Calling 双向握手闭环
# ============================================================================
# 这是整个"手写 Agent"的发动机。
# 流程概览：
#
#   用户: "我今天吃了多少蛋白质？"
#     │
#     ▼
#   ┌─────────────────────────────────────────────────────┐
#   │ 第①轮调用：把 user 消息 + tools 声明发给大模型        │
#   │   POST DeepSeek API                                 │
#   │   messages = [history..., user_msg]                 │
#   │   tools    = TOOLS                                  │
#   │   stream   = False  ← 第①轮不用流式，方便拆 tool_call │
#   └──────────────┬──────────────────────────────────────┘
#                  │
#      大模型返回：finish_reason = "tool_calls"
#      message.tool_calls = [
#        {name: "get_user_fitness_stats",
#         arguments: '{"user_id":"u1","metric":"protein"}'}
#      ]
#                  │
#                  ▼
#   ┌─────────────────────────────────────────────────────┐
#   │ ②拦截 & 本地执行                                     │
#   │   1. 判断 finish_reason == "tool_calls"              │
#   │   2. 解析 tool_calls[0].function.name               │
#   │   3. json.loads(tool_calls[0].function.arguments)   │
#   │   4. 查 TOOL_DISPATCH 表 → 调用真实异步函数           │
#   │   5. 拿到 dict 结果                                  │
#   └──────────────┬──────────────────────────────────────┘
#                  │
#                  ▼
#   ┌─────────────────────────────────────────────────────┐
#   │ ③构造"完整剧本" — 把两轮信息追加到 messages            │
#   │                                                     │
#   │   messages 数组在 user 消息之后新增：                  │
#   │   [                                                 │
#   │     ...,                                            │
#   │     {role: "assistant", content: null,              │
#   │      tool_calls: [{id, type, function: {name, args}}│
#   │     ]},  ← 大模型说"我想调这个函数"                    │
#   │     {role: "tool",                                  │
#   │      tool_call_id: "call_xxx",                      │
#   │      content: '{"user_id":"u1",...}'},  ← 本地执行结果│
#   │   ]                                                 │
#   └──────────────┬──────────────────────────────────────┘
#                  │
#                  ▼
#   ┌─────────────────────────────────────────────────────┐
#   │ ④终局流式输出：把完整 messages 再次 POST 给大模型       │
#   │   POST DeepSeek API                                 │
#   │   messages = [完整历史]                              │
#   │   stream   = True   ← 这次要流式！打字机效果           │
#   │   tools    = TOOLS  ← 仍需传，大模型可能再次要求调工具  │
#   │                                                     │
#   │   大模型看到 tool 结果，组织语言 → SSE 逐块推给前端      │
#   └─────────────────────────────────────────────────────┘
#
# ★ 关键认知：
#   大模型本身不调用任何函数！它只是一个"意图识别 + 自然语言生成"引擎。
#   我们的代码（胶水层）负责：
#     - 拦截 tool_calls
#     - 真正执行函数
#     - 把结果塞回消息列表
#     - 再次请求大模型生成最终回复
#   这个循环可以递归 — 如果大模型觉得还需要调更多工具，会再次返回 tool_calls。

@app.post("/v1/chat/fc")
async def chat_function_calling(req: StreamChatRequest):
    """
    Function Calling 流式对话接口。

    支持自动工具调用：当用户问健身数据时，大模型自动识别并调用本地函数，
    拿到真实数据后再流式输出最终回复。

    ★ 测试用例（对应模拟数据）：
      curl -X POST http://127.0.0.1:8000/v1/chat/fc \\
        -H "Content-Type: application/json" \\
        -d '{"prompt":"我是 u1，今天吃了多少蛋白质？","session_id":"fc-test"}' \\
        -N

      curl -X POST http://127.0.0.1:8000/v1/chat/fc \\
        -H "Content-Type: application/json" \\
        -d '{"prompt":"我（u1）这周练了几天了？","session_id":"fc-test"}' \\
        -N
    """
    # ------ 7.1 会话管理（与 chat_stream 一致）------
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]

    history: List[dict] = conversations[session_id]

    # 追加用户当前消息到历史
    history.append({"role": "user", "content": req.prompt})

    # ========================================================================
    # ★ 第①轮调用：非流式，带 tools 声明
    # ========================================================================
    # 用非流式（stream=False）的原因：
    #   流式模式下 tool_calls 的 arguments 是分块 JSON 片段，需要手动拼接，
    #   增加复杂度且容易出错。非流式直接拿到完整的 tool_calls 对象，干净利落。
    #
    # 对照 Java：
    #   类似先调一次 CompletableFuture<Response> 拿结果，
    #   判断结果类型后再决定下一步是否开启 Flux<SSE> 流式返回。
    try:
        round1_response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=history,       # ★ 当前历史（含刚追加的 user 消息）
            tools=TOOLS,            # ★ 工具声明 — 告诉大模型"你能用这些函数"
            stream=False,           # ★ 第①轮关流式，方便拿完整 tool_calls
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"第①轮调用失败: {e}")

    # ------ 7.2 解析第①轮响应 ------
    choice = round1_response.choices[0]
    finish_reason = choice.finish_reason  # "stop" / "tool_calls" / "length"
    assistant_msg = choice.message

    # ========================================================================
    # ★ 核心判断：大模型想直接回答，还是想调用工具？
    # ========================================================================
    # finish_reason == "tool_calls" → 大模型说"我需要先查数据才能回答"
    # finish_reason == "stop"       → 大模型说"我不需要查数据，直接回答"
    # finish_reason == "length"     → 达到 max_tokens 被截断（边缘情况）
    #
    # tool_calls 结构（OpenAI SDK 的 ChatCompletionMessageToolCall 对象）：
    #   tc.id                → "call_abc123"  （本次调用的唯一 ID）
    #   tc.type              → "function"     （固定值）
    #   tc.function.name     → "get_user_fitness_stats"
    #   tc.function.arguments → '{"user_id":"u1","metric":"protein"}'  （JSON 字符串！）
    #
    # ★ 重要：arguments 是 JSON 字符串，不是 dict！必须 json.loads() 解析。

    if finish_reason == "tool_calls" and assistant_msg.tool_calls:
        # ====================================================================
        # ★ 第②步：拦截 tool_calls，本地执行
        # ====================================================================
        # 大模型不会真正调用函数 — 它只输出一个"调用意图"。
        # 我们的胶水代码负责：解析 → dispatch → 执行 → 结果回填。

        # ------ 7.3 把 assistant 的 tool_calls 消息追加进历史 ------
        # 这条消息告诉大模型（在后续轮次中）："你之前决定要调这些工具"。
        # 注意！content 必须设为 None（工具调用时 assistant 不产生文本）。
        #
        # ★ messages 追加格式必须与 OpenAI API 规范严格一致：
        #   {
        #     "role": "assistant",
        #     "content": null,
        #     "tool_calls": [
        #       {
        #         "id": "call_xxx",
        #         "type": "function",
        #         "function": {"name": "...", "arguments": "{...}"}
        #       }
        #     ]
        #   }
        tool_calls_for_history = []
        for tc in assistant_msg.tool_calls:
            tool_calls_for_history.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,  # ★ 保持 JSON 字符串形式
                },
            })

        history.append({
            "role": "assistant",
            "content": None,  # ★ 关键：tool_calls 时无文本内容
            "tool_calls": tool_calls_for_history,
        })

        # ------ 7.4 本地执行每个工具调用 ------
        # 遍历大模型要求的每一个 tool_call，真正执行本地函数。
        # 支持并行场景：如果大模型一次要求调多个工具，我们全部执行。
        #
        # ★ 对照 Java：
        #   这里相当于 Spring 的反射调用：
        #     Method method = bean.getClass().getMethod(toolName, params...);
        #     Object result = method.invoke(bean, args);
        #   或者在微服务中通过 Feign 调用远程服务：
        #     FitnessServiceClient client = feignClientFactory.get(FitnessServiceClient.class);
        #     Map<String, Object> result = client.getUserFitnessStats(userId, metric);
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name                    # 如 "get_user_fitness_stats"
            func_args_str = tc.function.arguments            # 如 '{"user_id":"u1","metric":"protein"}'
            tool_call_id = tc.id                             # 如 "call_abc123"

            # ---- json.loads：JSON 字符串 → Python dict ----
            # 大模型可能传的不完全合法（JSON 里有注释/尾部逗号），
            # 这里用 try/except 兜底，防止一个工具失败导致整个请求崩溃。
            try:
                func_args: dict = json.loads(func_args_str)
            except json.JSONDecodeError as parse_err:
                # 解析失败时返回错误信息作为 tool result，
                # 让大模型知道参数有问题，它会尝试修正或向用户道歉。
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        {"error": f"参数 JSON 解析失败: {parse_err}"},
                        ensure_ascii=False,
                    ),
                })
                continue  # 跳过这个失败的调用，继续处理下一个

            # ---- 查调度表，找到对应的本地函数 ----
            # TOOL_DISPATCH 是一个 dict[str, callable]：
            #   {"get_user_fitness_stats": <async function get_user_fitness_stats>}
            #
            # ★ 对照 Java：
            #   这个 dispatch 表 ≈ Spring Bean 容器。
            #   高级版本可以用 @Tool(name="xxx") 注解扫描 + 依赖注入自动注册。
            handler = TOOL_DISPATCH.get(func_name)

            if handler is None:
                # 大模型脑补了一个不存在的函数名（罕见但可能发生）
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(
                        {"error": f"未知函数: {func_name}"},
                        ensure_ascii=False,
                    ),
                })
                continue

            # ---- 真正执行本地异步函数！----
            # 这是整个流程中唯一"真实发生"的事情 —
            # 大模型只是说"我想调 get_user_fitness_stats(u1, protein)"，
            # 这里才是真正查数据库 / 调 API / 读文件的地方。
            try:
                tool_result: dict = await handler(**func_args)
                # handler(**func_args) 等价于：
                #   get_user_fitness_stats(user_id="u1", metric="protein")
            except Exception as exec_err:
                tool_result = {"error": f"函数执行失败: {exec_err}"}

            # ---- 追加 tool 角色消息到历史 ----
            # ★ 这条消息是"胶水层给出的真实答案"————大模型下一轮读到它，
            #    就知道查数据库的结果是什么，然后可以基于事实生成回复。
            #
            # ★ messages 追加格式：
            #   {
            #     "role": "tool",
            #     "tool_call_id": "call_xxx",   ← 必须与上文的 tool_calls[].id 对应
            #     "content": "{json字符串}"
            #   }
            history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        # ====================================================================
        # ★ 第④步：终局流式输出（第②轮调用）
        # ====================================================================
        # 此时 history 已经完整：
        #   [system, user, assistant(tool_calls), tool(result), ...可能还有更多轮]
        #
        # 这一次用 stream=True，让大模型基于 tool 结果组织语言，
        # 并以打字机效果推给前端。
        #
        # ★ 仍然传 tools=TOOLS！
        #   大模型可能在看到第一次结果后，觉得还需要查另一个指标。
        #   例如：用户问"我的蛋白质和锻炼情况如何？"
        #   大模型可能先调 protein，拿到结果后发现还需要 workout_days，
        #   再次返回 tool_calls。这里只演示了单轮工具调用，
        #   生产环境可以用 while 循环支持递归工具调用。

        async def fc_generate():
            """异步生成器 — 第②轮流式推送给客户端"""
            try:
                stream = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=history,       # ★ 包含 tool_calls + tool result 的完整历史
                    stream=True,            # ★ 终局用流式，打字机效果
                    max_tokens=2048,
                )

                full_reply: str = ""

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    content = delta.content
                    if content:
                        full_reply += content
                        sse_data = json.dumps(
                            {"delta": content, "session_id": session_id},
                            ensure_ascii=False,
                        )
                        yield f"data: {sse_data}\n\n"

                    finish = chunk.choices[0].finish_reason if chunk.choices else None
                    if finish:
                        break

                # 流结束，把 assistant 的最终回复追加到历史
                if full_reply:
                    history.append({"role": "assistant", "content": full_reply})

                yield "data: [DONE]\n\n"

            except Exception as e:
                error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            fc_generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session_id,
            },
        )

    else:
        # ====================================================================
        # ★ 降级路径：大模型不需要调用工具，直接回答
        # ====================================================================
        # finish_reason == "stop" — 大模型觉得不需要查数据就能回答。
        # 把 assistant 回复追加到历史，然后以流式形式返回给前端。
        #
        # 注意：这里 assistant_msg 已经有完整内容，但我们仍用流式输出
        # 以保持前端一致（前端期望接收 SSE 事件流）。
        direct_reply = assistant_msg.content or ""

        # 追加 assistant 回复到历史
        if direct_reply:
            history.append({"role": "assistant", "content": direct_reply})

        async def direct_generate():
            """非工具路径的生成器 — 把已有回复以 SSE 格式发出"""
            if direct_reply:
                sse_data = json.dumps(
                    {"delta": direct_reply, "session_id": session_id},
                    ensure_ascii=False,
                )
                yield f"data: {sse_data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            direct_generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session_id,
            },
        )


# ============================================================================
# ★ 8) POST /v1/chat/rag — RAG 增强流式对话（向量检索 + 大模型生成）
# ============================================================================
# 完整 RAG 闭环：用户问题 → Embedding → pgvector 检索 → Prompt 增强 → 流式输出
#
# 流程概览：
#   Step 1: 用户 prompt → Embedding API（BAAI/bge-m3）→ query_vector(1024D)
#   Step 2: query_vector → pgvector cosine 检索 → Top-2 文本切片
#   Step 3: 切片拼接成参考资料 → 注入 system prompt
#   Step 4: 增强后的 messages → DeepSeek API（stream=True）→ SSE 逐块推前端
#
# ★ 关键指标（终端质检打印）：
#   - 每个检索结果的 ID / 相似度 / 前 100 字预览
#   - 让你核查大模型有没有"看着小抄回答"

@app.post("/v1/chat/rag")
async def chat_rag(req: StreamChatRequest):
    """
    RAG 增强流式对话接口。

    测试用例（需先执行 rag_pipeline.py --ingest 入库）：
      curl -X POST http://127.0.0.1:8000/v1/chat/rag \\
        -H "Content-Type: application/json" \\
        -d '{"prompt":"我今天练背应该注意什么？","session_id":"rag-test"}' \\
        -N

      curl -X POST http://127.0.0.1:8000/v1/chat/rag \\
        -H "Content-Type: application/json" \\
        -d '{"prompt":"硬拉的标准动作是什么？","session_id":"rag-test"}' \\
        -N
    """
    # ------ 8.1 会话管理 ------
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in conversations:
        conversations[session_id] = []

    history: List[dict] = conversations[session_id]

    # ========================================================================
    # ★ Step 1 & 2: 向量检索（RAG 的 "Retrieval" 环节）
    # ========================================================================
    # 把用户问题变成向量 → 去 pgvector 里找最相关的知识切片。
    #
    # ★ 对照 Java：
    #   这里相当于一个同步的 Feign 调用链：
    #     EmbeddingService.embed(prompt) → pgvectorClient.search(vector)
    #   但 Python 的 async/await 让两个 I/O 操作串在一起不阻塞线程。
    #   在 Java WebFlux 中同样的效果需要写：
    #     return embeddingMono.flatMap(vector -> pgvectorMono.map(rows -> ...))
    #   嵌套回调的阅读体验远不如 await。

    try:
        query_vector, retrieved = await rag_retrieve(req.prompt, top_k=2)
    except HTTPException:
        raise  # 配置错误直接抛出
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 检索失败: {e}")

    # ========================================================================
    # ★ 质检打印：后端控制台可视化检索结果
    # ========================================================================
    # 生产环境可以改用 logging.info()，这里为了学习用 print 直出。
    width = 70
    print(f"\n{'═' * width}")
    print(f"  🔍 RAG 检索质检  |  Session: {session_id[:8]}...")
    print(f"{'═' * width}")
    print(f"  用户提问: \"{req.prompt}\"")
    print(f"  Embedding: {EMBEDDING_MODEL}（{len(query_vector)}D）")
    print(f"{'─' * width}")
    if not retrieved:
        print("  ⚠️ 未检索到相关切片（数据库为空或相似度太低）")
    else:
        for i, chunk in enumerate(retrieved, 1):
            preview = chunk["content"][:100].replace("\n", " ")
            print(f"  🏆 Top-{i}  id={chunk['id']}  相似度={chunk['similarity']:.4f}")
            print(f"     {preview}...")
    print(f"{'═' * width}\n")

    # ========================================================================
    # ★ Step 3: 组装增强 messages（RAG 的 "Augmented" 环节）
    # ========================================================================
    # build_rag_messages 把检索结果注入 system prompt，
    # 生成类似下面的结构：
    #   [
    #     {role:"system", content:"严格基于参考资料回答...\n[参考资料1]...\n[参考资料2]..."},
    #     ...历史消息（不含旧 system prompt）...,
    #     {role:"user", content:"我今天练背应该注意什么？"},
    #   ]
    #
    # ★ 对照 Java：
    #   这里相当于 Spring AI 的 PromptTemplate 或 Guava 的 Joiner：
    #     String context = retrieved.stream()
    #         .map(c -> "[参考资料" + c.id + "] " + c.content)
    #         .collect(Collectors.joining("\n---\n"));
    #     Message systemMsg = new SystemMessage(ragTemplate.render(context, query));
    augmented_messages = build_rag_messages(req.prompt, retrieved, history)

    # ========================================================================
    # ★ Step 4: 流式输出（RAG 的 "Generation" 环节）
    # ========================================================================
    # 此时 augmented_messages 已包含参考资料上下文，
    # 大模型会基于这些资料生成最终回复。
    #
    # ★ 仍然传 tools=TOOLS！
    #   RAG 检索 + Function Calling 可以共存：
    #   如果用户问的是知识库里的内容 → RAG 提供上下文
    #   如果用户问的是需要实时查询的数据（如蛋白质摄入）→ FC 调用工具
    #   两者同时传给大模型，大模型自行决策用哪个。
    #
    # ★ 对照 Java（大规模并发优化）：
    #   见本文件末尾的 "Java 思维对照：RAG 串行 I/O 的并发优化"

    async def rag_generate():
        """异步生成器 — RAG 增强后的流式推送给客户端"""
        full_reply: str = ""
        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=augmented_messages,  # ★ 注入参考资料后的完整 messages
                stream=True,
                max_tokens=2048,
                tools=TOOLS,                 # ★ RAG + FC 共存
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                content = delta.content
                if content:
                    full_reply += content
                    sse_data = json.dumps(
                        {"delta": content, "session_id": session_id},
                        ensure_ascii=False,
                    )
                    yield f"data: {sse_data}\n\n"

                finish = chunk.choices[0].finish_reason if chunk.choices else None
                if finish:
                    break

            # 流结束：把原始 user 消息 + assistant 最终回复追加到历史
            # 注意：这里追加的是不带 RAG 上下文的原始消息，
            # 因为 RAG 注入的 system prompt 只在当前轮有效。
            if full_reply:
                history.append({"role": "user", "content": req.prompt})
                history.append({"role": "assistant", "content": full_reply})

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        rag_generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# 9) 辅助路由：查看 / 清除历史
# ---------------------------------------------------------------------------
@app.get("/v1/chat/history/{session_id}")
async def get_history(session_id: str):
    """查看某个会话的完整历史（调试用）"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "messages": conversations[session_id]}


@app.delete("/v1/chat/history/{session_id}")
async def clear_history(session_id: str):
    """清除某个会话的历史"""
    if session_id in conversations:
        del conversations[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ============================================================================
# ★ Java 思维对照：Spring Cloud 微服务中如何落地 Function Calling
# ============================================================================
#
# 问题：在真实的 Java 微服务架构中，本地函数（Tools）和 AI 胶水层如何解耦？
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  核心矛盾                                                          │
# │  大模型输出的是"函数名字符串 + JSON 参数字符串"，                      │
# │  我们的胶水层需要把字符串映射到真实的业务逻辑执行。                     │
# │  在单体中很简单：一个 Map<String, Function> 就够了。                 │
# │  但在 20+ 微服务的公司里，Tools 散落在不同的服务中 —                  │
# │  怎么找到它们、调用它们？                                           │
# └─────────────────────────────────────────────────────────────────────┘
#
# 方案 A：Feign 接口动态反射（推荐中小规模）
# ─────────────────────────────────────────
# 思路：每个服务的 Tools 仍然是本地方法，AI 网关通过 Spring Cloud Feign
# 动态调用。
#
#   @FeignClient(name = "fitness-service", path = "/api/internal/tools")
#   public interface FitnessToolsClient {
#       @PostMapping("/getUserFitnessStats")
#       Map<String, Object> getUserFitnessStats(
#           @RequestParam String userId,
#           @RequestParam String metric
#       );
#   }
#
#   // AI 胶水层（网关服务内）
#   @Service
#   public class ToolDispatcher {
#       private final Map<String, FeignMethodExecutor> dispatch;
#
#       // 启动时通过反射扫描所有 @FeignClient，注册到 dispatch 表
#       public Object dispatch(String toolName, Map<String, Object> args) {
#           FeignMethodExecutor executor = dispatch.get(toolName);
#           return executor.invoke(args);  // 底层走 HTTP → 目标服务
#       }
#   }
#
#   优点：代码清晰，Feign 接口即文档，Spring 生态原生支持
#   缺点：Feign 接口和 Tools Schema 需要手动保持同步（两套描述同一件事）
#   改进：用注解同时生成 Feign 接口 + OpenAI Tools Schema
#
#         @Tool(name = "get_user_fitness_stats",
#               description = "查询用户健身数据")
#         @FeignClient(name = "fitness-service")
#         public Map<String, Object> getUserFitnessStats(
#             @ToolParam(description = "用户ID") String userId,
#             @ToolParam(enum = {"protein","workout_days"}) String metric
#         );
#
#         启动时扫描 @Tool 注解 → 自动生成 JSON Schema（给大模型）
#                              → 自动注册 Feign 调用链（给网关）
#                              → 一份代码，双重用途
#
# 方案 B：配置中心动态路由（推荐大规模 / 多团队）
# ─────────────────────────────────────────────
# 思路：把 Tools 注册到 Nacos / Apollo 配置中心，AI 网关从配置中心
# 实时拉取工具清单 + 路由规则，完全解耦。
#
#   # Nacos 配置示例（JSON）：
#   {
#     "tools": [
#       {
#         "name": "get_user_fitness_stats",
#         "description": "查询用户健身数据",
#         "parameters": { ... },           ← JSON Schema，直接喂给大模型
#         "route": {
#           "type": "dubbo",               ← 调用协议
#           "service": "com.xxx.FitnessService",
#           "method": "getUserFitnessStats",
#           "version": "1.0.0",
#           "timeout": 3000
#         }
#       },
#       {
#         "name": "create_order",
#         "route": {
#           "type": "http",                ← 另一种协议
#           "url": "http://order-service/api/tools/createOrder",
#           "method": "POST"
#         }
#       }
#     ]
#   }
#
#   // AI 网关（动态路由层）
#   @Service
#   public class DynamicToolDispatcher {
#       // 监听 Nacos 配置变更，实时刷新本地缓存
#       @NacosConfigListener(dataId = "ai-tools-config")
#       public void onConfigChange(String configJson) {
#           this.toolRegistry = parseTools(configJson);
#       }
#
#       public Object dispatch(String toolName, Map<String, Object> args) {
#           ToolDef tool = toolRegistry.get(toolName);
#           return switch (tool.route.type) {
#               case "dubbo" -> dubboRouter.invoke(tool, args);
#               case "http"  -> httpRouter.invoke(tool, args);
#               case "grpc"  -> grpcRouter.invoke(tool, args);
#               default      -> throw new UnsupportedOperationException();
#           };
#       }
#   }
#
#   优点：工具热更新不停服、跨团队只需配配置、支持异构协议统一调度
#   缺点：配置中心变成关键依赖、调试链路变长、需要额外的契约测试
#
# 方案 C：Agent 独立服务（终极解耦）
# ─────────────────────────────────────
# 把整个 AI 胶水层抽成一个独立的 "Agent Service"，它：
#   1. 对外暴露 OpenAI 兼容的 /v1/chat 接口（前端直接对接）
#   2. 对内通过消息队列（Kafka/RabbitMQ）编排多服务 Tools 调用
#   3. 用 Saga 模式处理跨服务工具调用的最终一致性
#
#   这种方案下，每个业务服务只需：
#     - 声明自己的 Tools（配置中心注册）
#     - 实现工具执行端点（如 POST /internal/tools/xxx）
#     - Agent Service 自动发现、自动路由、自动重试
#
# ────────────────────────────────────────────────────────────────────
# 总结建议（基于你的 7 年 Java 经验）：
#
#   阶段 1（当前 try-ai 阶段）：
#     用 Map<String, Callable> dispatch 表，像你现在做的这样。
#     这对应 Java 里一个 Map<String, Method> + reflection invoke。
#
#   阶段 2（团队内推广，3-5 个工具）：
#     用自定义 @Tool 注解 + 启动时扫描，自动生成 Schema + 注册路由。
#     这对应 Spring 的 @EventListener 扫描机制。
#
#   阶段 3（多团队/多服务，50+ 工具）：
#     上配置中心动态路由，Tools 注册从代码内移到 Nacos/Apollo。
#     这对应服务网格（Service Mesh）的控制面 — 配置与执行分离。
#
#   阶段 4（全公司 AI 中台）：
#     Agent Service 独立部署，MCP 协议替代自定义 Tool Schema，
#     变成公司级 AI 基础设施。此时你写的 try-ai 代码就是中台的雏形。


# ---------------------------------------------------------------------------
# 10) 静态文件服务 + 前端聊天页面
# ---------------------------------------------------------------------------
# FastAPI 的 StaticFiles 会挂载整个目录，访问 /static/chat.html 即可。
# 下面的 / 路由重定向到聊天页面，方便直接打开 http://127.0.0.1:8000

# 确保 static 目录存在
_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def root():
    """默认跳转到聊天页面"""
    chat_html = os.path.join(_static_dir, "chat.html")
    return FileResponse(chat_html)


# ============================================================================
# ★ Java 思维对照：RAG 串行 I/O 在高并发下的吞吐量优化
# ============================================================================
#
# 问题：每次聊天都要 "Embedding API → pgvector 查询 → LLM 流式生成"，
#       这是一个典型的串行 I/O 链路。在 Java 中如何进行吞吐量优化？
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Python 当前架构（单请求路径）                                        │
# │                                                                     │
# │  用户请求                                                           │
# │    → await embeddings.create()      // 网络 I/O，~50-200ms          │
# │    → await asyncio.to_thread(pg)    // DB I/O，~1-5ms               │
# │    → async for chunk in stream      // 流式 I/O，~1-10s             │
# │                                                                     │
# │  Python asyncio 的优势：                                            │
# │    每个 await 点释放事件循环，单线程可同时处理数百请求。              │
# │    不需要线程池、不需要 CompletableFuture、不需要回调。              │
# │    这是"协作式多任务"——代码看起来是同步的，执行上是并发的。           │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Java 工程优化方案（按复杂度递增）                                    │
# │                                                                     │
# │  方案 1 — CompletableFuture 异步编排（Spring Boot 标配）             │
# │  ─────────────────────────────────────────────────────               │
# │  把三个 I/O 调用包装成 CompletableFuture，用 thenCompose 串联：      │
# │                                                                     │
# │    CompletableFuture<float[]> embedFuture =                          │
# │        CompletableFuture.supplyAsync(                                │
# │            () -> embeddingService.embed(prompt),                     │
# │            embedExecutor       // ★ 自定义线程池！                   │
# │        );                                                           │
# │                                                                     │
# │    CompletableFuture<List<Chunk>> searchFuture =                     │
# │        embedFuture.thenComposeAsync(                                 │
# │            vector -> CompletableFuture.supplyAsync(                  │
# │                () -> pgvectorService.search(vector),                 │
# │                dbExecutor      // ★ 另一个线程池！                   │
# │            )                                                        │
# │        );                                                           │
# │                                                                     │
# │    CompletableFuture<StreamingResponse> llmFuture =                  │
# │        searchFuture.thenComposeAsync(                                │
# │            chunks -> {                                               │
# │                String context = buildContext(chunks);                │
# │                return llmService.streamChat(context, prompt);        │
# │            },                                                        │
# │            llmExecutor         // ★ 第三个线程池                     │
# │        );                                                           │
# │                                                                     │
# │  关键点：每种 I/O 用独立线程池，避免慢的调用耗尽共享池。              │
# │  但这只是单请求的异步化，高并发下还需要：                             │
# │                                                                     │
# │  方案 2 — Reactor / WebFlux 全链路非阻塞                             │
# │  ────────────────────────────────────────                             │
# │  用 Spring WebFlux + R2DBC + WebClient 替代阻塞组件：                │
# │                                                                     │
# │    return webClient.post(embedApi, prompt)                           │
# │        .bodyToMono(float[].class)                   // 非阻塞 HTTP  │
# │        .flatMap(vector -> r2dbcClient.sql(                          │
# │            "SELECT ... <=> $1 LIMIT 2"                               │
# │        ).bind(0, vector).fetch().all())             // 非阻塞 DB    │
# │        .flatMap(chunks -> {                                         │
# │            String ctx = buildContext(chunks);                        │
# │            return ServerSentEvent.fromPublisher(                     │
# │                llmClient.stream(ctx, prompt)        // 非阻塞 LLM   │
# │            );                                                       │
# │        });                                                          │
# │                                                                     │
# │  优势：整个链路零阻塞，一个 Netty 线程池（核数 × 2）即可承载         │
# │        数千并发。和 Python asyncio 的哲学最接近。                     │
# │  代价：调试地狱 —— stack trace 里全是 Mono.flatMap 嵌套。           │
# │                                                                     │
# │  方案 3 — 消息队列削峰 + 异步解耦                                    │
# │  ─────────────────────────────────────                                │
# │  适用于突发流量（如双11活动页面的 AI 问答）：                         │
# │                                                                     │
# │    用户请求 → Kafka topic: rag-requests                             │
# │              ↓                                                     │
# │    Consumer Group (可动态扩缩容):                                    │
# │      1) Embedding Worker  × N 台                                    │
# │      2) pgvector 本身已支持并发查询（PostgreSQL 连接池）              │
# │      3) LLM Worker        × M 台（按 API RPM 上限设定）              │
# │              ↓                                                     │
# │    结果写回 Kafka/Redis → 前端通过 SSE/WebSocket 订阅结果            │
# │                                                                     │
# │  优势：每个环节独立扩缩容、流量削峰、失败重试天然支持                 │
# │  代价：延迟增加（Kafka 往返 50-200ms），架构复杂度翻倍               │
# │                                                                     │
# │  方案 4 — 本地缓存（Caffeine / Guava Cache）                         │
# │  ─────────────────────────────────────────────                       │
# │  高频问题（如 "硬拉标准动作"）的 Embedding + 检索结果缓存：          │
# │                                                                     │
# │    Cache<String, List<Chunk>> ragCache = Caffeine.newBuilder()       │
# │        .maximumSize(10_000)                                         │
# │        .expireAfterWrite(30, TimeUnit.MINUTES)                      │
# │        .build();                                                    │
# │                                                                     │
# │    List<Chunk> chunks = ragCache.get(prompt, key -> {               │
# │        float[] vec = embed(key);                                    │
# │        return pgvector.search(vec, 2);                              │
# │    });                                                              │
# │                                                                     │
# │  效果：90%+ 的常见问题命中缓存，Embedding API 调用量降为 1/10。      │
# │  在 Python 中可用 functools.lru_cache 或 Redis 实现同样效果。         │
# │                                                                     │
# │  方案 5 — 虚拟线程（Java 21+ Virtual Threads）                       │
# │  ───────────────────────────────────────────────                      │
# │  如果你不想学 WebFlux 的 Mono/flatMap，Java 21 的虚拟线程             │
# │  让"一请求一线程"达到近似 asyncio 的并发能力：                        │
# │                                                                     │
# │    // Spring Boot 3.2 + 虚拟线程                                    │
# │    @GetMapping("/chat/rag")                                         │
# │    public StreamingResponse chatRag(String prompt) {                │
# │        float[] vec = embedService.embed(prompt);    // 阻塞但轻量   │
# │        List<Chunk> chunks = pgvectorRepo.search(vec);               │
# │        return llmService.stream(buildContext(chunks, prompt));       │
# │    }                                                                │
# │    // 每个请求占用一个虚拟线程（~1KB 栈内存），                         │
# │    // Tomcat 可以同时跑几十万个虚拟线程。                              │
# │    // 代码和同步写法一模一样，不需要 CompletableFuture。              │
# │                                                                     │
# │  这其实是 Python asyncio 的 "哲学翻版"——                           │
# │     Python: 一个线程 + await 挂起点 = 高并发                         │
# │     Java:   百万虚拟线程 + 阻塞式写法 = 高并发                       │
# │     Python 的 await 是编译器帮你插入的挂起点                          │
# │     Java 虚拟线程的挂起是 JVM 在阻塞 I/O 时自动做的                   │
# │     殊途同归。                                                      │
# │                                                                     │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  ▎实战建议（针对你当前的 try-ai 阶段）                               │
# │                                                                     │
# │  1) 当前的串行 await 架构先跑通，别过早优化。                        │
# │  2) 下一步加 lru_cache 或 Redis 缓存高频问答（最省钱、最有效）。      │
# │  3) 如果日活破千，把 Embedding + pgvector 拆成独立 Python 微服务，   │
# │     Java 主服务通过 HTTP/gRPC 调用（保持语言边界清晰）。              │
# │  4) 如果日活破万 → Kafka 异步解耦 + 各环节独立扩缩容。               │
# └─────────────────────────────────────────────────────────────────────┘
""" 文件尾注释结束 """