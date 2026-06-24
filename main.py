"""
最简 AI Web 后端服务 —— 给 Java 后端工程师的入门 Demo
=========================================================
对照 Spring Boot 帮你建立心智模型：
  - FastAPI 本身  ≈ Spring Boot (Web 部分)
  - uvicorn      ≈ 内嵌的 Tomcat / Netty (ASGI 服务器)
  - 装饰器 @app.post ≈ @RestController + @PostMapping
  - Pydantic 模型 ≈ Java 的 DTO + Bean Validation (JSR-303)
  - async def    ≈ Spring WebFlux 里的响应式方法 (Mono/Flux)，但写法是同步风格
"""

# ---------------------------------------------------------------------------
# 1) 导入依赖
# ---------------------------------------------------------------------------
# Python 的 import 比 Java 灵活：你可以只导入某个类/函数，类似 Java 的静态导入。
import os                                  # 读环境变量，相当于 System.getenv()
from fastapi import FastAPI, HTTPException # FastAPI 主类 + 异常类
from pydantic import BaseModel, Field      # 用于声明请求/响应体的 DTO
from openai import AsyncOpenAI             # 官方 SDK 的 “异步版” 客户端
from dotenv import load_dotenv             # 加载 .env，相当于 Spring 的配置加载

# ---------------------------------------------------------------------------
# 2) 加载配置
# ---------------------------------------------------------------------------
# load_dotenv() 把 .env 中的键值塞进 os.environ。
# 相当于 Spring Boot 启动时读取 application.properties。
load_dotenv()

# 从环境变量取出 API Key 与 BaseURL。
# 注意：与其在代码中硬编码 key，不如走环境变量，这点和 Java 一样。
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# 想换模型直接改这里，比如 "gpt-4o-mini" / "deepseek-chat" / "glm-4-flash"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    # Python 没有 “未捕获异常 → 启动失败” 的 Spring Boot 体验，
    # 这里我们手动抛出，让进程在启动期就崩，避免线上跑着才发现没 key。
    raise RuntimeError("环境变量 OPENAI_API_KEY 未设置，请在 .env 中配置")

# ---------------------------------------------------------------------------
# 3) 创建全局对象
# ---------------------------------------------------------------------------
# FastAPI() 实例 ≈ Spring Boot 启动后注册好的 DispatcherServlet。
# 整个进程一份，路由全部挂在它身上。
app = FastAPI(title="My First AI Backend", version="0.1.0")

# AsyncOpenAI 客户端是 “线程安全 + 异步友好” 的单例，
# 类比 Java 里 @Bean 注入的 WebClient / RestTemplate —— 全局复用一个就够。
# 千万不要每次请求都 new 一个，会泄漏连接。
client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# 4) 定义请求 / 响应 DTO
# ---------------------------------------------------------------------------
# Pydantic 的 BaseModel ≈ Java 里加了 @Data + @Validated 的 DTO。
# FastAPI 会自动：
#   - 反序列化 JSON → 对象 (相当于 Jackson 干的事)
#   - 校验字段类型 / 必填 (相当于 Hibernate Validator)
#   - 生成 OpenAPI 文档 (相当于 springdoc-openapi 的零配置版)
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="用户输入的提示词")


class ChatResponse(BaseModel):
    reply: str
    model: str


# ---------------------------------------------------------------------------
# 5) 路由：POST /v1/chat
# ---------------------------------------------------------------------------
# 装饰器 @app.post("/v1/chat") ≈ @PostMapping("/v1/chat")
# response_model 会让 FastAPI 用该 schema 做出参校验 + 自动生成文档。
#
# 关键概念 —— async / await：
# -------------------------------------
# Python 的协程不是线程！它在 “单线程 + 事件循环” 上跑，
# 类比 Spring WebFlux 的 Reactor，或 Node.js 的 event loop。
#
#   - async def 声明的函数 → 返回一个 Coroutine 对象（不会立刻执行）
#   - await 表达式        → 在 I/O 等待时把控制权交还事件循环，
#                            让其他请求继续被处理（“非阻塞”）
#
# 对应 Java：
#   - 传统 Spring MVC：每个请求绑定一个 Tomcat 线程，I/O 时线程被 park
#   - WebFlux/Reactor： Mono.flatMap(...) 链式回调
#   - 协程 (Kotlin / Project Loom 的虚拟线程) 才真正神似 Python 的 async
#
# 所以你看到 await 就当成 “这是一个非阻塞调用，会让出 CPU”。
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    接收 prompt → 调用大模型 → 返回回复。
    单接口里没有任何线程切换，但能高并发，秘密就在 await。
    """
    try:
        # 这里是“原生” SDK 调用，不走任何 LangChain 之类的高级框架。
        # client.chat.completions.create 是个 async 方法，必须 await。
        # 没有 await 的话你拿到的只是 Coroutine 对象 (类似 Java 里 Future 没 get())。
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                # OpenAI 的会话协议：每条消息有 role + content
                # role 取值：system / user / assistant
                # 这里只放一条 user 消息，最朴素的 “一问一答”
                {"role": "user", "content": req.prompt},
            ],
            # 可选参数，按需打开：
            # temperature=0.7,
            # max_tokens=1024,
        )
    except Exception as e:
        # SDK 抛出的异常种类繁多 (网络/鉴权/限流/模型不存在...)，
        # 这里简单粗暴地全部映射为 HTTP 500，生产环境建议分类处理。
        # HTTPException 类似 Spring 里抛 ResponseStatusException。
        raise HTTPException(status_code=500, detail=f"调用大模型失败: {e}")

    # OpenAI 的返回结构：
    #   completion.choices 是一个数组（因为可能要多个候选答案，n 参数控制）
    #   choices[0].message.content 才是真正的文本
    # 这套结构和 Java SDK 是一一对应的，只是 Python 用属性访问而非 getter。
    reply_text = completion.choices[0].message.content or ""

    return ChatResponse(reply=reply_text, model=MODEL_NAME)


# ---------------------------------------------------------------------------
# 6) 健康检查 (可选但强烈推荐)
# ---------------------------------------------------------------------------
# 类比 Spring Boot Actuator 的 /actuator/health
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
