# grok-launch

Launcher for [Grok Build CLI](https://x.ai/cli) that routes its **Responses API** traffic through a local translation proxy to a **standard OpenAI Chat Completions** endpoint.

**No secrets or private endpoints are hard-coded.** Configure everything via `.env` or environment variables.

---

## Architecture & Implementation Principles (实现原理)

`grok-launch` functions as a lightweight middleware layer bridging Grok's backend API protocol with standard OpenAI Chat Completions endpoints:

```
+------------+                   +--------------------+                   +---------------+
|            |  Responses API    |                    |  Chat Completions |               |
| Grok Build | ----------------> | grok-launch Proxy  | ----------------> | Upstream LLM  |
| CLI        |  (Local Port)     | (Python Web Server)|  (Authorization)  | (OpenAI / AI) |
|            | <---------------- |                    | <---------------- |               |
+------------+    SSE Stream     +--------------------+    SSE Stream     +---------------+
```

### 1. Dynamic Port allocation & Proxy Threading
* When you run `grok-launch`, it locates a dynamic free TCP port on `127.0.0.1`.
* It spawns a background `ThreadingHTTPServer` (using HTTP/1.1 protocols) in a daemon thread.
* The main thread then runs the `grok` CLI as a subprocess, injecting the environment variables:
  - `GROK_CLI_CHAT_PROXY_BASE_URL="http://127.0.0.1:{port}/v1"`
  - `GROK_XAI_API_BASE_URL="http://127.0.0.1:{port}/v1"`

### 2. Request Translation (`Responses` -> `Chat Completions`)
The proxy intercepts the `POST /v1/responses` request payload and constructs a standard OpenAI `/v1/chat/completions` payload:
* **System Instructions**: Prepends top-level `instructions` as a `system` message.
* **Message Thread Mapping**: Translates user messages (`type: message`) and assistant messages (`type: assistant_message`) into corresponding OpenAI messages list.
* **Tool-Use Mapping**: Decodes assistant tool-use requests (`type: function_call`, `custom_tool_call`) and tool execution results (`type: function_call_output`, etc.) to standard `tool` roles with matching `tool_call_id`.
* **Namespaced Tool Declarations**: Flattens nested/namespaced tool definitions (e.g. `fs.read_file`) into flattened OpenAI function declarations (e.g. `fs__read_file`) to satisfy standard OpenAPI schemas.

### 3. SSE Stream Translation (`Chat Completions` -> `Responses`)
For streaming request execution (`stream: true`), the proxy converts the incremental OpenAI Server-Sent Events (SSE) stream back into Grok-compatible Responses SSE events:
* **Initial Connection**: Emits `response.created`.
* **Content Delta Output**: Streams incremental text pieces via `response.output_text.delta`.
* **Assistant Message Packaging**: Groups streamed text blocks within `response.output_item.added` and finalizes with `response.output_text.done` and `response.output_item.done`.
* **Terminal Markers**: Concludes the event stream with `response.done` and `response.completed`.

### 4. HTTP/1.1 Chunked Transfer-Encoding
* The translation proxy enforces `protocol_version = "HTTP/1.1"` in its HTTP handler. This ensures that the web socket server responds using HTTP/1.1 and correctly drives chunked transfer encoding (`Transfer-Encoding: chunked`) required by Grok CLI's HTTP engine (reqwest/hyper) to process real-time streams without socket connection teardown.

### 5. Token Usage Tracking
* It monitors and parses the final SSE chunk from the upstream API to extract `usage` data (`prompt_tokens`, `completion_tokens`).
* It inserts these numbers back into the final `response.done` and `response.completed` payloads, allowing Grok's TUI/CLI interface to report precise token statistics.

---

## Quick start

### 1. Install (wrapper in ~/.local/bin + config template)

```bash
./install.sh
```
`./install.sh` also auto-installs the upstream CLI when it is missing (use `--skip-cli` to opt out).

### 2. Edit user config

```bash
$EDITOR ~/.config/grok-launch/.env
```

### 3. Ensure PATH and run

```bash
export PATH="$HOME/.local/bin:$PATH"
grok-launch
grok-launch -p "hello"
```

## Configuration

### Required

| Variable | Meaning |
|----------|---------|
| `GROK_LAUNCH_BASE_URL` | OpenAI-compatible base, e.g. `https://api.openai.com/v1` |
| `GROK_LAUNCH_MODEL` | Real model name sent to `/chat/completions`, e.g. `gpt-4o` |
| `GROK_LAUNCH_API_KEY` | Bearer token |

### Optional

| Variable | Meaning |
|----------|---------|
| `GROK_LAUNCH_WIRE_API` | Upstream protocol mode: `"chat"` translates Responses to Chat Completions and supports tools (default); use `"responses"` only for native Responses API upstreams. |
| `GROK_LAUNCH_REASONING_EFFORT` | Default reasoning effort fallback (`"low"`, `"medium"`, `"high"` - only for reasoning models) |
| `GROK_LAUNCH_MAX_COMPLETION_TOKENS` | Default maximum completion tokens (thinking limit fallback) |
| `GROK_LAUNCH_MAX_TOKENS` | Default maximum context tokens limit fallback |
| `GROK_LAUNCH_CLI_MODEL` | Optional Grok CLI-side model override. Leave empty to let Grok choose its own default. |
| `GROK_BIN` | Path to `grok` binary (default `grok`) |
| `GROK_LAUNCH_PORT` | Fixed proxy port |
| `GROK_LAUNCH_VERBOSE` | Enable proxy logging (`1` or `true`) |

### `.env` load priority

1. `GROK_LAUNCH_ENV` if set
2. Package directory `.env` (launcher-local, next to `main.py`)
3. `./.env` or `./.grok-launch.env` (cwd)
4. Parent directories (up to 6 levels)
5. `~/.config/grok-launch/.env`
6. `~/.grok-launch.env`

For grok-launch-managed keys (`GROK_LAUNCH_*` and `GROK_BIN`), higher-priority `.env` files override stale shell exports.

## Usage (same flags as grok)

```bash
grok-launch
grok-launch -p "reply with hello"
GROK_LAUNCH_VERBOSE=true grok-launch -p "hello"
```
