#!/usr/bin/env python3
# Copyright (c) 2026 grok-launch contributors
# 仅供学习与研究使用；其他用途后果自负。
# For learning and research only; any other use is at your own risk.
"""grok-launch: local translation proxy and launcher for Grok Build CLI.

Routes Grok's Responses API traffic to an OpenAI-compatible Chat Completions API.
No secrets are hard-coded; configuration is managed through environment variables or .env files.
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import socket
import urllib.request
import urllib.error
import subprocess
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Generator

_HERE = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_KEYS = (
    "GROK_LAUNCH_BASE_URL",
    "GROK_LAUNCH_MODEL",
    "GROK_LAUNCH_API_KEY",
)

TARGET_BASE_URL: str = ""
TARGET_MODEL: str = ""
TARGET_API_KEY: str = ""
GROK_LAUNCH_CLI_MODEL: str = "gpt-4o"
GROK_BIN: str = "grok"
VERBOSE: bool = False
DEBUG_DIR: str = ""
_LOADED_ENV_FILES: list[str] = []
WIRE_API: str = "responses"


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

    paths.append(os.path.join(_HERE, ".env"))

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


def load_dotenv_files() -> list[str]:
    loaded: list[str] = []
    claimed = set(os.environ.keys())

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
    global TARGET_BASE_URL, TARGET_MODEL, TARGET_API_KEY
    global GROK_LAUNCH_CLI_MODEL, GROK_BIN, VERBOSE, DEBUG_DIR, _LOADED_ENV_FILES, WIRE_API

    _LOADED_ENV_FILES = load_dotenv_files()

    missing = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
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
    TARGET_API_KEY = os.environ["GROK_LAUNCH_API_KEY"].strip()
    GROK_LAUNCH_CLI_MODEL = (os.environ.get("GROK_LAUNCH_CLI_MODEL") or "gpt-4o").strip()
    GROK_BIN = (os.environ.get("GROK_BIN") or "grok").strip()
    VERBOSE = (os.environ.get("GROK_LAUNCH_VERBOSE") or "").strip() in ("1", "true", "yes", "on")
    DEBUG_DIR = (
        os.environ.get("GROK_LAUNCH_DEBUG_DIR")
        or os.path.join(tempfile.gettempdir(), "grok-launch")
    ).strip()
    WIRE_API = (os.environ.get("GROK_LAUNCH_WIRE_API") or "responses").strip().lower()
    if WIRE_API not in ("responses", "chat"):
        print(f"warning: unknown GROK_LAUNCH_WIRE_API value '{WIRE_API}'. defaulting to 'responses'.", file=sys.stderr)
        WIRE_API = "responses"


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

    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens", "max_completion_tokens"):
        if key in payload:
            out_payload[key] = payload[key]

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
    for index, tc_entry in sorted(tool_calls_map.items()):
        call_id = tc_entry["id"] or f"call_{uuid.uuid4().hex[:12]}"
        name = tc_entry["name"] or "tool"
        arguments = tc_entry["arguments"] or "{}"

        tc_done = {
            "type": "response.output_item.done",
            "item": {
                "id": tc_entry["item_id"],
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
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {TARGET_API_KEY}",
                "User-Agent": "grok-launch/1.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        except Exception as e:
            if VERBOSE:
                print(f"[grok-launch] models endpoint failed: {e}. returning mock model list.", file=sys.stderr)

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

        if VERBOSE:
            print(f"[grok-launch] forwarding chat completions directly to: {url} stream={wants_stream}", file=sys.stderr)

        upstream_req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Authorization": f"Bearer {TARGET_API_KEY}",
                "Accept": "text/event-stream" if wants_stream else "application/json",
                "User-Agent": "grok-launch/1.0",
            },
            method="POST",
        )
        try:
            upstream_resp = urllib.request.urlopen(upstream_req, timeout=300)
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            print(f"[grok-launch] Upstream error {exc.code}: {err_body}", file=sys.stderr)
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Upstream error: {err_body}"}}).encode("utf-8"))
            return
        except Exception as e:
            print(f"[grok-launch] Upstream connection failed: {e}", file=sys.stderr)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Bad Gateway: {e}"}}).encode("utf-8"))
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

            if VERBOSE:
                print(f"[grok-launch] forwarding responses directly to: {url} stream={wants_stream}", file=sys.stderr)

            upstream_req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                    "Authorization": f"Bearer {TARGET_API_KEY}",
                    "Accept": "text/event-stream" if wants_stream else "application/json",
                    "User-Agent": "grok-launch/1.0",
                },
                method="POST",
            )
            try:
                upstream_resp = urllib.request.urlopen(upstream_req, timeout=300)
            except urllib.error.HTTPError as exc:
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
                print(f"[grok-launch] Upstream error {exc.code}: {err_body}", file=sys.stderr)
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": f"Upstream error: {err_body}"}}).encode("utf-8"))
                return
            except Exception as e:
                print(f"[grok-launch] Upstream connection failed: {e}", file=sys.stderr)
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": f"Bad Gateway: {e}"}}).encode("utf-8"))
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

        if VERBOSE:
            print(f"[grok-launch] responses -> chat/completions model={TARGET_MODEL} stream={wants_stream}", file=sys.stderr)

        url = build_upstream_url("/v1/chat/completions")
        upstream_req = urllib.request.Request(
            url,
            data=json.dumps(chat_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TARGET_API_KEY}",
                "Accept": "text/event-stream" if wants_stream else "application/json",
                "User-Agent": "grok-launch/1.0",
            },
            method="POST",
        )

        try:
            upstream_resp = urllib.request.urlopen(upstream_req, timeout=300)
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            print(f"[grok-launch] Upstream error {exc.code}: {err_body}", file=sys.stderr)
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Upstream error: {err_body}"}}).encode("utf-8"))
            return
        except Exception as e:
            print(f"[grok-launch] Upstream connection failed: {e}", file=sys.stderr)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": f"Bad Gateway: {e}"}}).encode("utf-8"))
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

    active_model = model_override if model_override else TARGET_MODEL

    if not model_override:
        args = ["--model", active_model] + args

    env = os.environ.copy()
    env["GROK_CLI_CHAT_PROXY_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["GROK_XAI_API_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["XAI_API_KEY"] = TARGET_API_KEY

    if VERBOSE:
        env_note = ", ".join(_LOADED_ENV_FILES) if _LOADED_ENV_FILES else "(none)"
        print(
            f"[grok-launch] proxy=http://127.0.0.1:{port} "
            f"upstream={TARGET_BASE_URL} "
            f"active_model={active_model}\n"
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
