# try-ai 项目上下文

## 用户背景
- 7 年 Java 后端经验，正在转型 AI 应用开发
- Python 3.10+ 环境，已配 DeepSeek API Key
- 习惯用 Spring Boot / Maven 心智模型理解 Python 生态

## 沟通约定
- **中文回复**
- 简洁，不浪费 token
- 不确定时先问，不要瞎猜
- 不要改超出范围的代码
- 多用 Java ↔ Python 对照来解释新概念

## 技术栈与对照
| Python | Java 对照 |
|---|---|
| uv | Maven / Gradle |
| pyproject.toml | pom.xml |
| uv.lock | 版本锁定文件 |
| FastAPI | Spring Boot (Web) |
| uvicorn | 内嵌 Tomcat (ASGI) |
| Pydantic BaseModel | Lombok + Jackson + Validator |
| async/await | WebFlux / Project Loom 虚拟线程 |
| AsyncOpenAI | OpenAI Java SDK |

## 项目当前状态

### 已完成
- ✅ uv 初始化项目，依赖装好（fastapi / uvicorn / openai / python-dotenv）
- ✅ [main.py](main.py) 实现 `POST /v1/chat` 同步返回接口（非流式）
- ✅ `.env` 配置 DeepSeek（`OPENAI_BASE_URL=https://api.deepseek.com/v1`，`MODEL_NAME=deepseek-chat`）
- ✅ `.env` 已加入 `.gitignore`，`.env.example` 作为模板提交
- ✅ git 初始化 + 推送到 GitHub
- ✅ VSCode 解释器配置（`.vscode/settings.json` 指向 `.venv`）

### 下一步：流式输出（用户已学到这里，未实现）
关键点：
1. SDK 调用加 `stream=True`，返回类型变成 `AsyncStream`
2. 用 `async for chunk in stream:` 迭代，取 `chunk.choices[0].delta.content`
3. FastAPI 用 `StreamingResponse` + 异步生成器（`async def` 含 `yield`）把流推给客户端
4. 响应 media_type 用 `text/event-stream`（SSE 协议）
5. 测试用 `curl -N`（关闭缓冲）才能看到打字机效果

参考实现已在对话中给出，新增接口 `/v1/chat/stream` 而非替换原接口，方便对比。

### 后续可选关卡
1. 多轮对话（维护 messages 列表，理解大模型无状态本质）
2. 结构化输出（Pydantic + `response_format`，模型返回 JSON）
3. Function Calling（Agent 雏形）

## 常用命令

```bash
# 启动服务
uv run uvicorn main:app --reload

# 加依赖
uv add <package>

# 同步依赖（新机器克隆后）
uv sync

# 测试接口
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好"}'

# Swagger UI
# http://127.0.0.1:8000/docs
```

## 关键文件
- [main.py](main.py) — FastAPI 主入口
- [pyproject.toml](pyproject.toml) — 依赖声明
- [.env](.env) — DeepSeek key（不进 git）
- [.env.example](.env.example) — 配置模板

## 用户常见困惑提醒
- VSCode 报"无法解析导入"通常是解释器没切到 `.venv`，让用户 `Cmd+Shift+P` → `Python: Select Interpreter`
- 用户对 docstring、Pydantic、async/await、Swagger 自动生成等概念已扫过盲，可直接引用
- DeepSeek 用 OpenAI 兼容协议，换厂商只改 `.env`，代码零修改