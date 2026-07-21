#!/usr/bin/env python3
# Copyright (c) 2026 grok-launch contributors
# 仅供学习与研究使用；其他用途后果自负。
# For learning and research only; any other use is at your own risk.
"""grok-launch: local translation proxy and launcher for Grok Build CLI.

Routes Grok's Responses API traffic to an OpenAI-compatible Chat Completions API.
No secrets are hard-coded; configuration is managed through environment variables or .env files.
"""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import os
import sys
import json
import uuid
import random
import socket
import urllib.request
import urllib.error
import subprocess
import threading
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Generator

_HERE = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_KEYS = (
    "GROK_LAUNCH_BASE_URL",
    "GROK_LAUNCH_MODEL",
)
_MANAGED_ENV_PREFIXES = ("GROK_LAUNCH_",)
_MANAGED_ENV_KEYS = {"GROK_BIN"}

TARGET_BASE_URL: str = ""
TARGET_MODEL: str = ""
TARGET_API_KEY: str = ""
API_KEYS: list[str] = []
FROZEN_KEYS: dict[str, float] = {}
FROZEN_LOCK = threading.Lock()
GROK_LAUNCH_CLI_MODEL: str = ""
GROK_BIN: str = "grok"
VERBOSE: bool = False
DEBUG_DIR: str = ""
_LOADED_ENV_FILES: list[str] = []
WIRE_API: str = "chat"
REASONING_EFFORT: str = ""
MAX_COMPLETION_TOKENS: int = 0
MAX_TOKENS: int = 0
UPSTREAM_USER_AGENT: str = "grok-launch/1.0"
UPSTREAM_MIN_INTERVAL_SECONDS: float = 0.25
UPSTREAM_RETRIES: int = 2
UPSTREAM_429_FREEZE_SECONDS: float = 60.0
UPSTREAM_5XX_FREEZE_SECONDS: float = 30.0
UPSTREAM_MAX_RETRY_AFTER_SECONDS: float = 300.0
UPSTREAM_BACKOFF_INITIAL_SECONDS: float = 1.0
UPSTREAM_BACKOFF_MAX_SECONDS: float = 30.0
UPSTREAM_NEXT_REQUEST_AT: float = 0.0
UPSTREAM_COOLDOWN_UNTIL: float = 0.0
UPSTREAM_THROTTLE_LOCK = threading.Lock()

def _bounded_float(value: str | None, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value) if value is not None and value.strip() else default
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _bounded_int(value: str | None, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value) if value is not None and value.strip() else default
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return None
        return max(0.0, dt.timestamp() - (time.time() if now is None else now))
    except Exception:
        return None


def _clamp_retry_delay(delay: float | None, fallback: float) -> float:
    selected = fallback if delay is None else delay
    return min(max(0.0, selected), UPSTREAM_MAX_RETRY_AFTER_SECONDS)


def _retry_backoff_seconds(attempt: int) -> float:
    base = min(UPSTREAM_BACKOFF_MAX_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS * (2 ** max(0, attempt)))
    return base + random.uniform(0.0, min(1.0, base * 0.25))


def _apply_upstream_cooldown(seconds: float, reason: str) -> None:
    if seconds <= 0:
        return
    capped = min(seconds, UPSTREAM_MAX_RETRY_AFTER_SECONDS)
    with UPSTREAM_THROTTLE_LOCK:
        global UPSTREAM_COOLDOWN_UNTIL
        until = time.time() + capped
        if until > UPSTREAM_COOLDOWN_UNTIL:
            UPSTREAM_COOLDOWN_UNTIL = until
    if VERBOSE:
        print(f"[grok-launch] upstream cooldown {capped:.2f}s ({reason})", file=sys.stderr)


def _wait_for_upstream_slot() -> None:
    global UPSTREAM_NEXT_REQUEST_AT
    while True:
        with UPSTREAM_THROTTLE_LOCK:
            now = time.time()
            wait_until = max(UPSTREAM_COOLDOWN_UNTIL, UPSTREAM_NEXT_REQUEST_AT)
            wait_for = wait_until - now
            if wait_for <= 0:
                UPSTREAM_NEXT_REQUEST_AT = now + UPSTREAM_MIN_INTERVAL_SECONDS
                return
        time.sleep(min(wait_for, 5.0))


def _is_retryable_upstream_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _compact_upstream_error(message: str, *, max_chars: int = 1000) -> str:
    text = (message or "").strip()
    lower = text.lower()
    if "<html" in lower or "<!doctype html" in lower:
        title = ""
        title_start = lower.find("<title>")
        title_end = lower.find("</title>")
        if 0 <= title_start < title_end:
            title = text[title_start + len("<title>"):title_end].strip()
        text = f"HTML upstream error page: {title}" if title else "HTML upstream error page returned by gateway"
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def get_active_key() -> str:
    with FROZEN_LOCK:
        now = time.time()
        for key in API_KEYS:
            if FROZEN_KEYS.get(key, 0) <= now:
                return key
        
        candidates = []
        for key in API_KEYS:
            expire = FROZEN_KEYS.get(key, 0)
            if expire - now < 43200: # less than 12 hours
                candidates.append((expire, key))
                
        if candidates:
            candidates.sort()
            expire, key = candidates[0]
            FROZEN_KEYS[key] = 0
            return key
            
        return API_KEYS[0]


def mark_key_failed(key: str, status_code: int, retry_after: str | None = None) -> float:
    with FROZEN_LOCK:
        now = time.time()
        if status_code == 429:
            freeze_for = _clamp_retry_delay(
                _parse_retry_after(retry_after, now=now),
                UPSTREAM_429_FREEZE_SECONDS,
            )
            FROZEN_KEYS[key] = now + freeze_for
            print(f"[grok-launch] key {key[:10]}... frozen temporarily (429 rate limit) for {freeze_for:.0f}s", file=sys.stderr)
            return freeze_for
        elif status_code in (401, 402):
            FROZEN_KEYS[key] = now + 86400
            print(f"[grok-launch] key {key[:10]}... frozen permanently (401/402 auth error)", file=sys.stderr)
            return 86400.0
        elif 500 <= status_code <= 599:
            freeze_for = UPSTREAM_5XX_FREEZE_SECONDS
            FROZEN_KEYS[key] = now + freeze_for
            print(f"[grok-launch] key {key[:10]}... frozen temporarily ({status_code} server error) for {freeze_for:.0f}s", file=sys.stderr)
            return freeze_for
    return 0.0


def _parse_dotenv(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                out[key] = val
    except OSError:
        pass
    return out


def _candidate_env_paths() -> list[str]:
    paths: list[str] = []
    explicit = os.environ.get("GROK_LAUNCH_ENV")
    if explicit:
        paths.append(os.path.expanduser(explicit))

    paths.append(os.path.join(_HERE, ".env"))

    cwd = os.getcwd()
    paths.append(os.path.join(cwd, ".env"))
    paths.append(os.path.join(cwd, ".grok-launch.env"))

    parent = os.path.dirname(cwd)
    for _ in range(6):
        if not parent or parent == os.path.dirname(parent):
            break
        paths.append(os.path.join(parent, ".env"))
        paths.append(os.path.join(parent, ".grok-launch.env"))
        parent = os.path.dirname(parent)

    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    paths.append(os.path.join(xdg, "grok-launch", ".env"))
    paths.append(os.path.expanduser("~/.grok-launch.env"))

    seen: set[str] = set()
    uniq: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def _is_managed_env_key(key: str) -> bool:
    return key in _MANAGED_ENV_KEYS or key.startswith(_MANAGED_ENV_PREFIXES)


def load_dotenv_files() -> list[str]:
    loaded: list[str] = []
    claimed = {key for key in os.environ.keys() if not _is_managed_env_key(key)}

    for path in _candidate_env_paths():
        if not os.path.isfile(path):
            continue
        data = _parse_dotenv(path)
        if not data:
            continue
        any_applied = False
        for k, v in data.items():
            if k in claimed:
                continue
            os.environ[k] = v
            claimed.add(k)
            any_applied = True
        if any_applied or data:
            loaded.append(path)
    return loaded


def load_config() -> None:
    global TARGET_BASE_URL, TARGET_MODEL, TARGET_API_KEY, API_KEYS
    global GROK_LAUNCH_CLI_MODEL, GROK_BIN, VERBOSE, DEBUG_DIR, _LOADED_ENV_FILES, WIRE_API
    global REASONING_EFFORT, MAX_COMPLETION_TOKENS, MAX_TOKENS
    global UPSTREAM_USER_AGENT, UPSTREAM_MIN_INTERVAL_SECONDS, UPSTREAM_RETRIES
    global UPSTREAM_429_FREEZE_SECONDS, UPSTREAM_5XX_FREEZE_SECONDS
    global UPSTREAM_MAX_RETRY_AFTER_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS
    global UPSTREAM_BACKOFF_MAX_SECONDS

    _LOADED_ENV_FILES = load_dotenv_files()

    missing = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
    keys_str = os.environ.get("GROK_LAUNCH_API_KEYS") or os.environ.get("GROK_LAUNCH_API_KEY") or ""
    API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not API_KEYS:
        missing.append("GROK_LAUNCH_API_KEY (or GROK_LAUNCH_API_KEYS)")

    if missing:
        example = os.path.join(_HERE, ".env.example")
        xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        user_env = os.path.join(xdg, "grok-launch", ".env")
        print("error: missing required configuration:", ", ".join(missing), file=sys.stderr)
        print(file=sys.stderr)
        print("Set them in a .env file or environment variables.", file=sys.stderr)
        print("  project:  ./.env   (copy from .env.example)", file=sys.stderr)
        print(f"  user:     {user_env}", file=sys.stderr)
        print(f"  template: {example}", file=sys.stderr)
        print(file=sys.stderr)
        if _LOADED_ENV_FILES:
            print("loaded env files (still missing keys):", file=sys.stderr)
            for p in _LOADED_ENV_FILES:
                print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    TARGET_BASE_URL = os.environ["GROK_LAUNCH_BASE_URL"].strip().rstrip("/")
    TARGET_MODEL = os.environ["GROK_LAUNCH_MODEL"].strip()
    TARGET_API_KEY = API_KEYS[0]
    GROK_LAUNCH_CLI_MODEL = (os.environ.get("GROK_LAUNCH_CLI_MODEL") or "").strip()
    GROK_BIN = (os.environ.get("GROK_BIN") or "grok").strip()
    VERBOSE = (os.environ.get("GROK_LAUNCH_VERBOSE") or "").strip() in ("1", "true", "yes", "on")
    DEBUG_DIR = (
        os.environ.get("GROK_LAUNCH_DEBUG_DIR")
        or os.path.join(tempfile.gettempdir(), "grok-launch")
    ).strip()
    WIRE_API = (os.environ.get("GROK_LAUNCH_WIRE_API") or "chat").strip().lower()
    if WIRE_API not in ("responses", "chat"):
        print(f"warning: unknown GROK_LAUNCH_WIRE_API value '{WIRE_API}'. defaulting to 'chat'.", file=sys.stderr)
        WIRE_API = "chat"
    UPSTREAM_USER_AGENT = (os.environ.get("GROK_LAUNCH_USER_AGENT") or "grok-launch/1.0").strip()
    UPSTREAM_MIN_INTERVAL_SECONDS = _bounded_float(os.environ.get("GROK_LAUNCH_UPSTREAM_MIN_INTERVAL_SECONDS"), 0.25)
    UPSTREAM_RETRIES = _bounded_int(os.environ.get("GROK_LAUNCH_UPSTREAM_RETRIES"), 2)
    UPSTREAM_429_FREEZE_SECONDS = _bounded_float(os.environ.get("GROK_LAUNCH_429_FREEZE_SECONDS"), 60.0)
    UPSTREAM_5XX_FREEZE_SECONDS = _bounded_float(os.environ.get("GROK_LAUNCH_5XX_FREEZE_SECONDS"), 30.0)
    UPSTREAM_MAX_RETRY_AFTER_SECONDS = _bounded_float(os.environ.get("GROK_LAUNCH_MAX_RETRY_AFTER_SECONDS"), 300.0)
    UPSTREAM_BACKOFF_INITIAL_SECONDS = _bounded_float(os.environ.get("GROK_LAUNCH_BACKOFF_INITIAL_SECONDS"), 1.0)
    UPSTREAM_BACKOFF_MAX_SECONDS = _bounded_float(os.environ.get("GROK_LAUNCH_BACKOFF_MAX_SECONDS"), 30.0)
    REASONING_EFFORT = (os.environ.get("GROK_LAUNCH_REASONING_EFFORT") or "").strip().lower()
    try:
        MAX_COMPLETION_TOKENS = int(os.environ.get("GROK_LAUNCH_MAX_COMPLETION_TOKENS") or "0")
    except ValueError:
        print("warning: GROK_LAUNCH_MAX_COMPLETION_TOKENS must be an integer", file=sys.stderr)
        MAX_COMPLETION_TOKENS = 0
    try:
        MAX_TOKENS = int(os.environ.get("GROK_LAUNCH_MAX_TOKENS") or "0")
    except ValueError:
        print("warning: GROK_LAUNCH_MAX_TOKENS must be an integer", file=sys.stderr)
        MAX_TOKENS = 0


def build_upstream_url(path: str) -> str:
    base = TARGET_BASE_URL.rstrip("/")
    if "?" in path:
        path_part, query_part = path.split("?", 1)
        query = "?" + query_part
    else:
        path_part = path
        query = ""
    rel = path_part.lstrip("/")
    if base.endswith("/v1") and rel.startswith("v1/"):
        rel = rel[3:]
    return f"{base}/{rel}{query}"


def payload_with_upstream_model(payload: dict[str, Any]) -> bytes:
    upstream_payload = dict(payload)
    upstream_payload["model"] = TARGET_MODEL
    return json.dumps(upstream_payload).encode("utf-8")


def _open_upstream_with_retries(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    data: bytes | None = None,
    timeout: int = 300,
    label: str = "upstream",
) -> Any:
    last_error: Exception | None = None
    max_attempts = max(len(API_KEYS), 1) + max(0, UPSTREAM_RETRIES)
    for attempt in range(max_attempts):
        active_key = get_active_key()
        request_headers = dict(headers)
        request_headers["Authorization"] = f"Bearer {active_key}"
        request_headers.setdefault("User-Agent", UPSTREAM_USER_AGENT)
        if VERBOSE:
            print(f"[grok-launch] {label} request attempt={attempt + 1}/{max_attempts} key={active_key[:10]}...", file=sys.stderr)
        req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            _wait_for_upstream_slot()
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            frozen_for = mark_key_failed(active_key, exc.code, retry_after)
            last_error = exc
            if exc.code == 429:
                cooldown = _clamp_retry_delay(
                    _parse_retry_after(retry_after),
                    min(frozen_for or UPSTREAM_429_FREEZE_SECONDS, _retry_backoff_seconds(attempt)),
                )
                _apply_upstream_cooldown(cooldown, "429 rate limit")
            elif 500 <= exc.code <= 599:
                _apply_upstream_cooldown(_retry_backoff_seconds(attempt), f"{exc.code} upstream error")
            if _is_retryable_upstream_status(exc.code) and attempt < max_attempts - 1:
                continue
            if exc.code in (401, 402) and attempt < min(len(API_KEYS), max_attempts) - 1:
                continue
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                _apply_upstream_cooldown(_retry_backoff_seconds(attempt), "transport error")
                continue
            break
    if last_error is not None:
        raise last_error
    raise RuntimeError("upstream request failed")


def _debug_write(name: str, obj: Any) -> None:
    if not DEBUG_DIR:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(obj, (bytes, bytearray)):
                f.write(obj.decode("utf-8", errors="replace"))
            elif isinstance(obj, str):
                f.write(obj)
            else:
                json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def sanitize_chat_stream_chunk(chunk_bytes: bytes) -> bytes:
    line = chunk_bytes.decode("utf-8", errors="ignore")
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return chunk_bytes

    data_val = stripped[5:].strip()
    if not data_val or data_val == "[DONE]":
        return chunk_bytes

    try:
        parsed = json.loads(data_val)
    except Exception:
        return chunk_bytes

    changed = False
    for choice in parsed.get("choices") or []:
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for tool_call in delta.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function")
            if isinstance(fn, dict) and fn.get("name") == "":
                del fn["name"]
                changed = True

    if not changed:
        return chunk_bytes
    return f"data: {json.dumps(parsed, ensure_ascii=False)}\n".encode("utf-8")


def responses_request_to_chat(payload: dict[str, Any], active_model: str) -> dict[str, Any]:
    model = active_model

    input_items = payload.get("input") or []
    if isinstance(input_items, str):
        input_items = [{"type": "message", "role": "user", "content": input_items}]
    elif not isinstance(input_items, list):
        input_items = []

    messages = []
    system_texts = []

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system_texts.append(instructions.strip())

    raw_tools = payload.get("tools") or []
    for item in input_items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            tools_list = item.get("tools")
            if isinstance(tools_list, list):
                raw_tools.extend(tools_list)

    namespaces = set()
    for tool in raw_tools:
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            name = tool.get("name")
            if isinstance(name, str):
                namespaces.add(name.strip())

    for idx, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or ""

        if item_type in ("message", "assistant_message", "agent_message"):
            raw_role = item.get("role")
            if not raw_role:
                if item_type == "message":
                    role = "user"
                else:
                    role = "assistant"
            else:
                role = "system" if raw_role == "developer" else raw_role

            raw_content = item.get("content")
            content = ""
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                parts = []
                for p in raw_content:
                    if isinstance(p, dict):
                        t = p.get("text") or p.get("output_text") or p.get("output_text_delta")
                        if t:
                            parts.append(t)
                content = "".join(parts)
            else:
                content = item.get("text") or ""

            if role == "system":
                if content:
                    system_texts.append(content)
            else:
                messages.append({"role": role, "content": content})

        elif item_type in ("function_call", "custom_tool_call", "mcp_tool_call"):
            call_id = item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}"
            name = item.get("name") or "tool"

            raw_args = item.get("arguments") or item.get("input") or "{}"
            if isinstance(raw_args, dict):
                arguments = json.dumps(raw_args)
            elif isinstance(raw_args, str):
                arguments = raw_args
            else:
                arguments = "{}"

            if messages and messages[-1].get("role") == "assistant":
                if "tool_calls" not in messages[-1]:
                    messages[-1]["tool_calls"] = []
                messages[-1]["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments
                        }
                    }]
                })

        elif item_type in ("function_call_output", "custom_tool_call_output", "tool_search_output", "mcp_tool_call_output"):
            call_id = item.get("call_id") or ""
            name = item.get("name") or ""

            raw_output = item.get("content") or item.get("output") or item.get("text") or ""
            output_str = ""
            if isinstance(raw_output, str):
                output_str = raw_output
            elif isinstance(raw_output, list):
                parts = []
                for p in raw_output:
                    if isinstance(p, dict):
                        t = p.get("text") or p.get("output_text") or p.get("output_text_delta")
                        if t:
                            parts.append(t)
                output_str = "".join(parts)

            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": output_str
            }
            messages.append(tool_msg)

    if system_texts:
        system_msg = {"role": "system", "content": "\n\n".join(system_texts)}
        messages.insert(0, system_msg)

    openai_tools = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        t_type = tool.get("type")
        if t_type == "function":
            fn = tool.get("function") or tool
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {})
                }
            })
        elif t_type == "namespace":
            ns_name = tool.get("name") or ""
            for subtool in tool.get("tools") or []:
                if isinstance(subtool, dict):
                    fn = subtool.get("function") or subtool
                    sub_name = fn.get("name") or ""
                    flat_name = f"{ns_name}__{sub_name}" if ns_name else sub_name
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": flat_name,
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {})
                        }
                    })
        elif t_type == "custom":
            name = tool.get("name") or ""
            if name == "exec_command":
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": tool.get("description", "Execute command"),
                        "parameters": tool.get("parameters", {})
                    }
                })

    out_payload = {
        "model": model,
        "messages": messages,
        "stream": bool(payload.get("stream", False))
    }
    if openai_tools:
        out_payload["tools"] = openai_tools
        out_payload["tool_choice"] = "auto"

    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens", "max_completion_tokens", "reasoning_effort"):
        if key in payload:
            out_payload[key] = payload[key]

    if "reasoning_effort" not in out_payload and REASONING_EFFORT:
        out_payload["reasoning_effort"] = REASONING_EFFORT
    if "max_completion_tokens" not in out_payload and MAX_COMPLETION_TOKENS > 0:
        out_payload["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    if "max_tokens" not in out_payload and MAX_TOKENS > 0:
        out_payload["max_tokens"] = MAX_TOKENS

    return out_payload


def chat_response_to_responses(chat_resp: dict[str, Any]) -> dict[str, Any]:
    choices = chat_resp.get("choices") or []
    output_items = []
    output_text = ""

    if choices:
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        output_text = content

        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            output_items.append({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "content": [{"type": "reasoning_text", "text": reasoning}],
                "encrypted_content": None
            })

        if content:
            output_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}]
            })

        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            output_items.append({
                "type": "function_call",
                "id": f"tc_{uuid.uuid4().hex[:12]}",
                "call_id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments", "{}")
            })

    usage = chat_resp.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)

    return {
        "id": f"resp_{uuid.uuid4().hex[:12]}",
        "object": "response",
        "output": output_items,
        "output_text": output_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        },
        "status": "completed"
    }


def stream_chat_to_responses(upstream_response: Any) -> Generator[bytes, None, None]:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    assistant_item_id = f"msg_{uuid.uuid4().hex[:12]}"

    yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': response_id}})}\n\n".encode("utf-8")

    message_started = False
    accumulated_text = ""
    tool_calls_map = {}
    usage = None

    for line_bytes in upstream_response:
        line = line_bytes.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("data:"):
            data_val = line[5:].strip()
            if data_val == "[DONE]":
                break
            try:
                parsed_chunk = json.loads(data_val)
            except Exception:
                continue

            if "usage" in parsed_chunk and parsed_chunk["usage"]:
                usage = parsed_chunk["usage"]

            choices = parsed_chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning_delta:
                yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'delta': reasoning_delta})}\n\n".encode("utf-8")

            content_delta = delta.get("content") or ""
            if content_delta:
                if not message_started:
                    message_started = True
                    item_added = {
                        "type": "response.output_item.added",
                        "item": {
                            "id": assistant_item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": ""}]
                        }
                    }
                    yield f"event: response.output_item.added\ndata: {json.dumps(item_added)}\n\n".encode("utf-8")

                accumulated_text += content_delta
                yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'delta': content_delta})}\n\n".encode("utf-8")

            tc_list = delta.get("tool_calls") or []
            for tc in tc_list:
                index = tc.get("index")
                if index is None:
                    continue
                if index not in tool_calls_map:
                    tool_calls_map[index] = {"id": "", "name": "", "arguments": "", "item_id": f"tc_{uuid.uuid4().hex[:12]}"}

                tc_entry = tool_calls_map[index]
                if tc.get("id"):
                    tc_entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tc_entry["name"] += fn["name"]
                if fn.get("arguments"):
                    tc_entry["arguments"] += fn["arguments"]

    # Finalize tool calls
    for output_index, (index, tc_entry) in enumerate(sorted(tool_calls_map.items())):
        call_id = tc_entry["id"] or f"call_{uuid.uuid4().hex[:12]}"
        name = tc_entry["name"] or "tool"
        arguments = tc_entry["arguments"] or "{}"
        item_id = tc_entry["item_id"]

        item_added = {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": ""
            }
        }
        yield f"event: response.output_item.added\ndata: {json.dumps(item_added)}\n\n".encode("utf-8")

        if arguments:
            args_delta = {
                "type": "response.function_call_arguments.delta",
                "item_id": item_id,
                "output_index": output_index,
                "delta": arguments
            }
            yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(args_delta)}\n\n".encode("utf-8")

        args_done = {
            "type": "response.function_call_arguments.done",
            "item_id": item_id,
            "output_index": output_index,
            "arguments": arguments
        }
        yield f"event: response.function_call_arguments.done\ndata: {json.dumps(args_done)}\n\n".encode("utf-8")

        tc_done = {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments
            }
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(tc_done)}\n\n".encode("utf-8")

    # Finalize assistant message
    if message_started:
        yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'text': accumulated_text})}\n\n".encode("utf-8")

        item_done = {
            "type": "response.output_item.done",
            "item": {
                "id": assistant_item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": accumulated_text}]
            }
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(item_done)}\n\n".encode("utf-8")

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    if usage:
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)

    resp_done = {
        "type": "response.done",
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "status": "completed"
        }
    }
    yield f"event: response.done\ndata: {json.dumps(resp_done)}\n\n".encode("utf-8")

    resp_completed = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "status": "completed"
        }
    }
    yield f"event: response.completed\ndata: {json.dumps(resp_completed)}\n\n".encode("utf-8")


class TranslationProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        if VERBOSE:
            sys.stderr.write(f"[grok-launch proxy] {format % args}\n")

    def do_GET(self) -> None:
        if self.path in ("/v1/models", "/models"):
            self._handle_models()
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path in ("/v1/responses", "/responses"):
            self._handle_responses()
            return
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
            return
        self.send_error(404, "Not Found")

    def _handle_models(self) -> None:
        url = build_upstream_url(self.path)
        try:
            with _open_upstream_with_retries(
                url,
                method="GET",
                headers={"User-Agent": UPSTREAM_USER_AGENT},
                timeout=10,
                label="models",
            ) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        except Exception as exc:
            if VERBOSE:
                print(f"[grok-launch] models endpoint failed: {_compact_upstream_error(str(exc))}. returning mock model list.", file=sys.stderr)

        # Fallback Mock
        mock_data = {
            "object": "list",
            "data": [
                {
                    "id": TARGET_MODEL,
                    "object": "model",
                    "created": 1686935000,
                    "owned_by": "openai",
                }
            ],
        }
        encoded = json.dumps(mock_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_chat_completions(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8") or "{}")
        except Exception as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        url = build_upstream_url(self.path)
        wants_stream = self.headers.get("Accept") == "text/event-stream" or payload.get("stream") == True
        upstream_body = payload_with_upstream_model(payload)
        _debug_write("chat_incoming_request.json", payload)
        _debug_write("chat_outgoing_request.json", json.loads(upstream_body.decode("utf-8")))

        try:
            upstream_resp = _open_upstream_with_retries(
                url,
                method="POST",
                data=upstream_body,
                headers={
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                    "Accept": "text/event-stream" if wants_stream else "application/json",
                    "User-Agent": UPSTREAM_USER_AGENT,
                },
                timeout=300,
                label="chat completions",
            )
        except Exception as exc:
            msg = _compact_upstream_error(str(exc))
            code = 502
            if isinstance(exc, urllib.error.HTTPError):
                code = exc.code
                try:
                    msg = _compact_upstream_error(exc.read().decode("utf-8", errors="replace"))
                except Exception:
                    pass
            print(f"[grok-launch] Chat completions forwarding failed: {msg}", file=sys.stderr)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Chat completions forwarding failed: {msg}"}}).encode("utf-8"))
            return

        if wants_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            try:
                debug_chunks = []
                for chunk_bytes in upstream_resp:
                    chunk_bytes = sanitize_chat_stream_chunk(chunk_bytes)
                    if VERBOSE:
                        print(f"[grok-launch] forwarding chunk: {chunk_bytes}", file=sys.stderr)
                    if DEBUG_DIR:
                        debug_chunks.append(chunk_bytes.decode("utf-8", errors="replace"))
                    self.wfile.write(f"{len(chunk_bytes):X}\r\n".encode("utf-8") + chunk_bytes + b"\r\n")
                    self.wfile.flush()
                if DEBUG_DIR:
                    _debug_write("chat_stream_response.sse", "".join(debug_chunks))
            except Exception as e:
                if VERBOSE:
                    print(f"[grok-launch] error during stream forwarding: {e}", file=sys.stderr)
            finally:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
        else:
            try:
                data = upstream_resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(500, f"Error processing chat completions forwarding: {e}")

    def _handle_responses(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8") or "{}")
        except Exception as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return

        if WIRE_API == "responses":
            url = build_upstream_url(self.path)
            wants_stream = self.headers.get("Accept") == "text/event-stream" or payload.get("stream") == True
            upstream_body = payload_with_upstream_model(payload)

            try:
                upstream_resp = _open_upstream_with_retries(
                    url,
                    method="POST",
                    data=upstream_body,
                    headers={
                        "Content-Type": self.headers.get("Content-Type", "application/json"),
                        "Accept": "text/event-stream" if wants_stream else "application/json",
                        "User-Agent": UPSTREAM_USER_AGENT,
                    },
                    timeout=300,
                    label="responses direct",
                )
            except Exception as exc:
                code = 502
                msg = _compact_upstream_error(str(exc))
                if isinstance(exc, urllib.error.HTTPError):
                    code = exc.code
                    try:
                        msg = _compact_upstream_error(exc.read().decode("utf-8", errors="replace"))
                    except Exception:
                        pass
                print(f"[grok-launch] Responses direct forwarding failed: {msg}", file=sys.stderr)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": f"Forwarding failed: {msg}"}}).encode("utf-8"))
                return

            if wants_stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                try:
                    for chunk_bytes in upstream_resp:
                        if VERBOSE:
                            print(f"[grok-launch] forwarding chunk: {chunk_bytes}", file=sys.stderr)
                        self.wfile.write(f"{len(chunk_bytes):X}\r\n".encode("utf-8") + chunk_bytes + b"\r\n")
                        self.wfile.flush()
                except Exception as e:
                    if VERBOSE:
                        print(f"[grok-launch] error during stream forwarding: {e}", file=sys.stderr)
                finally:
                    try:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except Exception:
                        pass
            else:
                try:
                    data = upstream_resp.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as e:
                    self.send_error(500, f"Error processing responses forwarding: {e}")
            return

        chat_payload = responses_request_to_chat(payload, TARGET_MODEL)
        wants_stream = bool(chat_payload.get("stream"))
        url = build_upstream_url("/v1/chat/completions")

        try:
            upstream_resp = _open_upstream_with_retries(
                url,
                method="POST",
                data=json.dumps(chat_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream" if wants_stream else "application/json",
                    "User-Agent": UPSTREAM_USER_AGENT,
                },
                timeout=300,
                label="responses translation",
            )
        except Exception as exc:
            code = 502
            msg = _compact_upstream_error(str(exc))
            if isinstance(exc, urllib.error.HTTPError):
                code = exc.code
                try:
                    msg = _compact_upstream_error(exc.read().decode("utf-8", errors="replace"))
                except Exception:
                    pass
            print(f"[grok-launch] Chat completions forwarding failed: {msg}", file=sys.stderr)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Chat completions forwarding failed: {msg}"}}).encode("utf-8"))
            return

        if wants_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            try:
                for chunk_bytes in stream_chat_to_responses(upstream_resp):
                    if VERBOSE:
                        print(f"[grok-launch] writing chunk: {chunk_bytes}", file=sys.stderr)
                    self.wfile.write(f"{len(chunk_bytes):X}\r\n".encode("utf-8") + chunk_bytes + b"\r\n")
                    self.wfile.flush()
            except Exception as e:
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
            finally:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
        else:
            try:
                data = upstream_resp.read()
                chat_resp = json.loads(data.decode("utf-8") or "{}")
                translated = chat_response_to_responses(chat_resp)
                encoded = json.dumps(translated).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception as e:
                self.send_error(500, f"Error processing response: {e}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    load_config()

    port = int(os.environ.get("GROK_LAUNCH_PORT") or 0) or find_free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), TranslationProxy)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    args = sys.argv[1:]

    model_override = None
    for idx, arg in enumerate(args):
        if arg in ("-m", "--model"):
            if idx + 1 < len(args):
                model_override = args[idx + 1]
                break

    cli_model = model_override if model_override else GROK_LAUNCH_CLI_MODEL

    if not model_override and cli_model:
        args = ["--model", cli_model] + args

    env = os.environ.copy()
    env["GROK_CLI_CHAT_PROXY_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["GROK_XAI_API_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["XAI_API_KEY"] = TARGET_API_KEY

    if VERBOSE:
        env_note = ", ".join(_LOADED_ENV_FILES) if _LOADED_ENV_FILES else "(none)"
        print(
            f"[grok-launch] proxy=http://127.0.0.1:{port} "
            f"upstream={TARGET_BASE_URL} "
            f"cli_model={cli_model} "
            f"upstream_model={TARGET_MODEL}\n"
            f"[grok-launch] env files: {env_note}",
            file=sys.stderr,
        )

    try:
        res = subprocess.run([GROK_BIN, *args], env=env)
        sys.exit(res.returncode)
    except FileNotFoundError:
        print(f"error: cannot find grok binary ({GROK_BIN})", file=sys.stderr)
        sys.exit(127)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
