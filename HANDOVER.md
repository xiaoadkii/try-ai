# try-ai 项目交接备忘录

> 目的：在**另一台设备**上从零接续这个 AI 后端学习项目。
> 用法：把这份 markdown 连同项目一起 clone 到新机器，让新机器上的 AI 助手读它。

---

## 一、项目是什么

一个 Java 后端工程师转型 AI 应用开发的"通关式"学习项目。
当前完成度：**第 1 关（最简同步聊天接口）已通**，正在做**第 2 关（流式输出）**。

技术栈：FastAPI + 原生 OpenAI SDK + DeepSeek（OpenAI 兼容协议）+ uv。
GitHub 仓库：（在你 GitHub 个人主页找 `try-ai`）

---

## 二、新机器从零搭起来

### 1. 前置工具

```bash
# Python 包管理器
brew install uv

# 或者你已有 uv，跳过
```

### 2. 拉代码

```bash
git clone git@github.com:<你的GitHub用户名>/try-ai.git
cd try-ai
```

### 3. 装依赖（一条命令）

```bash
uv sync
```

这一步会：
- 根据 `pyproject.toml` + `uv.lock` 创建 `.venv`
- 装好 fastapi、uvicorn、openai、python-dotenv 等

### 4. 配 API Key

```bash
cp .env.example .env
# 编辑 .env，填入真实 DeepSeek key
```

`.env` 内容：

```
OPENAI_API_KEY=sk-你的-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

> DeepSeek key 在 https://platform.deepseek.com/api_keys 拿。

### 5. VSCode 配置

打开项目后：
- `Cmd+Shift+P` → `Python: Select Interpreter` → 选 `./.venv/bin/python`
- 仓库里已有 `.vscode/settings.json`，正常会自动识别

### 6. 启动

```bash
uv run uvicorn main:app --reload
```

浏览器打开 http://127.0.0.1:8000/docs 看到 Swagger 即成功。

### 7. 测试

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好"}'
```

---

## 三、我学到哪儿了

### ✅ 已完成
- uv 项目初始化与依赖管理（对照 Maven）
- FastAPI 异步 Web 服务搭建
- Pydantic DTO 校验（对照 Lombok + Jackson + Validator）
- 原生 `client.chat.completions.create` async 调用
- Swagger 自动文档（FastAPI 内置，零配置）
- git 初始化、`.gitignore` 配置、推送到 GitHub
- SSH 公钥配置 GitHub 认证

### 🚧 进行中：流式输出
学到的关键点：
- `stream=False`（当前代码）→ 返回 `ChatCompletion`，整段拿
- `stream=True` → 返回 `AsyncStream`，用 `async for chunk in stream` 迭代
- 每个 chunk 的新增内容在 `chunk.choices[0].delta.content`
- FastAPI 用 `StreamingResponse` + 异步生成器（`async def` + `yield`）推流
- 协议是 SSE（`text/event-stream`），格式 `data: xxx\n\n`
- 测试 curl 加 `-N` 关闭缓冲

**还没动手实现**，参考代码在对话历史里。建议新增 `/v1/chat/stream` 接口，保留原接口对比。

### 📋 后续待办（按推荐顺序）
1. 流式输出（SSE）— 当前正在做
2. 多轮对话（messages 历史维护）
3. 结构化输出（Pydantic + response_format）
4. Function Calling（Agent 雏形）

---

## 四、关键概念笔记（Java 视角速查）

| 概念 | 一句话理解 |
|---|---|
| `async def` | 协程函数，类似 WebFlux 里的响应式方法 |
| `await` | "这是非阻塞 I/O，让出事件循环" |
| `AsyncOpenAI` | OpenAI SDK 的异步客户端，全局单例（不要每次 new） |
| Pydantic `BaseModel` | 一行顶 Java 五行注解的 DTO |
| FastAPI 装饰器 | 等价于 `@RestController + @PostMapping` |
| docstring (`"""..."""`) | 类似 Javadoc，但能被反射读取，FastAPI 用它生成 Swagger |
| uv | 比 pip 快 100 倍的 Python 包管理器，类比 Maven |
| pyproject.toml | 项目配置 + 依赖声明，类比 pom.xml |
| uv.lock | 锁定依赖树，**要提交到 git** |
| `.venv` | 虚拟环境，**不进 git** |
| `.env` | 环境变量（含 key），**绝对不能进 git** |

---

## 五、对 AI 助手的指令

如果你（另一台机器上的 Claude）正在读这份文档：

1. **始终用中文回复**
2. **简洁，少废话**
3. **多用 Spring Boot / Java 的概念类比**来解释 Python / FastAPI / AI 相关概念
4. 用户已经掌握的概念（见上面"已完成"和"关键概念笔记"）**可以直接引用**，不要重复入门讲解
5. 用户当前要做的是**流式输出**，可以直接进入这关
6. 项目还有 `CLAUDE.md` 也是上下文，配合一起读

---

## 六、最近一次对话的核心结论

- DeepSeek 兼容 OpenAI 协议 → 换厂商只改 `.env`，代码零修改
- VSCode 报红一般是解释器没切 `.venv`，不是依赖没装
- Swagger UI（`/docs`）+ ReDoc（`/redoc`）+ openapi.json，FastAPI 全自动生成
- `stream=True` 返回的是异步迭代器，必须 `async for`，不能 `for`
- `.env` 一旦误提交 → 立即吊销 key 重新生成，因为 git 历史里永远查得到

---

下班愉快，祝学习顺利。