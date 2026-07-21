import json
import os
import tempfile
import sys
import unittest
import io
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


class TranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        main.TARGET_MODEL = "gpt-4o"
        main.TARGET_BASE_URL = "https://api.openai.com/v1"
        main.TARGET_API_KEY = "mock-key"

    def test_responses_request_to_chat(self) -> None:
        payload = {
            "model": "gpt-4o",
            "instructions": "You are a helpful coding assistant.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Write a python script to reverse a list."
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Sure, here it is."}
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "exec_command",
                    "arguments": "{\"cmd\": \"echo hi\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "name": "exec_command",
                    "content": "hi\n"
                }
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "fs",
                    "tools": [
                        {
                            "type": "function",
                            "name": "read_file",
                            "description": "Read file content",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"]
                            }
                        }
                    ]
                }
            ],
            "stream": True
        }

        chat_payload = main.responses_request_to_chat(payload, "gpt-4o")

        self.assertEqual(chat_payload["model"], "gpt-4o")
        self.assertEqual(chat_payload["stream"], True)

        messages = chat_payload["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful coding assistant.")

        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Write a python script to reverse a list.")

        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Sure, here it is.")
        self.assertEqual(len(messages[2]["tool_calls"]), 1)
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "exec_command")

        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_123")
        self.assertEqual(messages[3]["content"], "hi\n")

        tools = chat_payload["tools"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "fs__read_file")
        self.assertEqual(tools[0]["function"]["description"], "Read file content")

        # Test reasoning effort and token limit copying & defaulting
        payload_with_effort = {"model": "gpt-4o", "reasoning_effort": "high"}
        res = main.responses_request_to_chat(payload_with_effort, "gpt-4o")
        self.assertEqual(res.get("reasoning_effort"), "high")

        # Test defaulting fallback values
        main.REASONING_EFFORT = "low"
        main.MAX_COMPLETION_TOKENS = 4000
        main.MAX_TOKENS = 8000
        
        payload_empty = {"model": "gpt-4o"}
        res_defaulted = main.responses_request_to_chat(payload_empty, "gpt-4o")
        self.assertEqual(res_defaulted.get("reasoning_effort"), "low")
        self.assertEqual(res_defaulted.get("max_completion_tokens"), 4000)
        self.assertEqual(res_defaulted.get("max_tokens"), 8000)

        # Cleanup globals
        main.REASONING_EFFORT = ""
        main.MAX_COMPLETION_TOKENS = 0
        main.MAX_TOKENS = 0

    def test_chat_response_to_responses(self) -> None:
        chat_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "reasoning_content": "Thought process",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "fs__read_file",
                                    "arguments": "{\"path\":\"README.md\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

        resp = main.chat_response_to_responses(chat_resp)

        self.assertEqual(resp["object"], "response")
        self.assertEqual(resp["output_text"], "Hello!")
        self.assertEqual(resp["usage"]["input_tokens"], 10)
        self.assertEqual(resp["usage"]["output_tokens"], 20)

        output = resp["output"]
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(output[0]["content"][0]["text"], "Thought process")

        self.assertEqual(output[1]["type"], "message")
        self.assertEqual(output[1]["content"][0]["text"], "Hello!")

        self.assertEqual(output[2]["type"], "function_call")
        self.assertEqual(output[2]["call_id"], "call_abc")
        self.assertEqual(output[2]["name"], "fs__read_file")

    def test_stream_chat_to_responses(self) -> None:
        sse_lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_999","function":{"name":"exec_command","arguments":"{\\"cmd\\":"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"ls\\""}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}\n',
            b"data: [DONE]\n"
        ]

        upstream_response = io.BytesIO(b"".join(sse_lines))
        generator = main.stream_chat_to_responses(upstream_response)
        
        events = []
        for chunk in generator:
            for sse_block in chunk.decode("utf-8").split("\n\n"):
                if not sse_block.strip():
                    continue
                event_line = ""
                data_line = ""
                for line in sse_block.split("\n"):
                    if line.startswith("event:"):
                        event_line = line[6:].strip()
                    elif line.startswith("data:"):
                        data_line = line[5:].strip()
                if event_line or data_line:
                    events.append({"event": event_line, "data": json.loads(data_line) if data_line else {}})

        self.assertEqual(events[0]["event"], "response.created")
        self.assertEqual(events[1]["event"], "response.output_item.added")
        self.assertEqual(events[2]["event"], "response.output_text.delta")
        self.assertEqual(events[3]["event"], "response.output_text.delta")
        
        tool_added_event = [
            e for e in events
            if e["event"] == "response.output_item.added"
            and e["data"].get("item", {}).get("type") == "function_call"
        ]
        self.assertEqual(len(tool_added_event), 1)
        self.assertEqual(tool_added_event[0]["data"]["item"]["call_id"], "call_999")
        self.assertEqual(tool_added_event[0]["data"]["item"]["name"], "exec_command")

        args_delta_event = [e for e in events if e["event"] == "response.function_call_arguments.delta"]
        self.assertEqual(len(args_delta_event), 1)
        self.assertEqual(args_delta_event[0]["data"]["delta"], "{\"cmd\":\"ls\"}")

        args_done_event = [e for e in events if e["event"] == "response.function_call_arguments.done"]
        self.assertEqual(len(args_done_event), 1)
        self.assertEqual(args_done_event[0]["data"]["arguments"], "{\"cmd\":\"ls\"}")

        tool_done_event = [
            e for e in events
            if e["event"] == "response.output_item.done"
            and e["data"].get("item", {}).get("type") == "function_call"
        ]
        self.assertEqual(len(tool_done_event), 1)
        self.assertEqual(tool_done_event[0]["data"]["item"]["call_id"], "call_999")
        self.assertEqual(tool_done_event[0]["data"]["item"]["name"], "exec_command")

        # Text done
        text_done_event = [e for e in events if e["event"] == "response.output_text.done"]
        self.assertEqual(len(text_done_event), 1)
        self.assertEqual(text_done_event[0]["data"]["text"], "Hello world")

    def test_key_rotation(self) -> None:
        main.API_KEYS = ["key1", "key2", "key3"]
        main.FROZEN_KEYS = {}

        # 1. Initially first key is returned
        self.assertEqual(main.get_active_key(), "key1")

        # 2. Freeze key1 temporarily
        main.mark_key_failed("key1", 429)
        self.assertEqual(main.get_active_key(), "key2")

        # 3. Freeze key2 permanently
        main.mark_key_failed("key2", 401)
        self.assertEqual(main.get_active_key(), "key3")

        # 4. If all keys are frozen, the one that expires earliest (key1) is chosen
        main.mark_key_failed("key3", 429)
        self.assertEqual(main.get_active_key(), "key1")

        # Clean up
        main.API_KEYS = ["mock-key"]
        main.FROZEN_KEYS = {}

    def test_sanitize_chat_stream_chunk_removes_empty_tool_name(self) -> None:
        chunk = (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"name":"","arguments":"{}"}}]}}]}\n'
        )
        sanitized = main.sanitize_chat_stream_chunk(chunk)
        parsed = json.loads(sanitized.decode("utf-8")[5:].strip())
        fn = parsed["choices"][0]["delta"]["tool_calls"][0]["function"]
        self.assertNotIn("name", fn)
        self.assertEqual(fn["arguments"], "{}")

    def test_payload_with_upstream_model_rewrites_model(self) -> None:
        main.TARGET_MODEL = "gpt-5.5"
        body = main.payload_with_upstream_model({"model": "grok-4.5", "stream": True})
        self.assertEqual(json.loads(body.decode("utf-8"))["model"], "gpt-5.5")

    def test_main_does_not_inject_cli_model_by_default(self) -> None:
        main.TARGET_MODEL = "upstream-model"
        main.TARGET_API_KEY = "mock-key"
        main.GROK_LAUNCH_CLI_MODEL = ""
        main.GROK_BIN = "grok"
        main.VERBOSE = False
        main._LOADED_ENV_FILES = []

        class FakeServer:
            def __init__(self, *args, **kwargs):
                pass

            def serve_forever(self):
                pass

            def shutdown(self):
                pass

            def server_close(self):
                pass

        completed = type("Completed", (), {"returncode": 0})()

        with patch.object(main, "load_config"), \
            patch.object(main, "find_free_port", return_value=12345), \
            patch.object(main, "ThreadingHTTPServer", FakeServer), \
            patch.object(main.threading, "Thread", autospec=True), \
            patch.object(main.subprocess, "run", return_value=completed) as run_mock, \
            patch.object(sys, "argv", ["main.py", "-p", "hello"]), \
            self.assertRaises(SystemExit) as cm:
            main.main()

        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(run_mock.call_args.args[0], ["grok", "-p", "hello"])

    def test_main_injects_explicit_cli_model(self) -> None:
        main.TARGET_MODEL = "upstream-model"
        main.TARGET_API_KEY = "mock-key"
        main.GROK_LAUNCH_CLI_MODEL = "gpt-4o"
        main.GROK_BIN = "grok"
        main.VERBOSE = False
        main._LOADED_ENV_FILES = []

        class FakeServer:
            def __init__(self, *args, **kwargs):
                pass

            def serve_forever(self):
                pass

            def shutdown(self):
                pass

            def server_close(self):
                pass

        completed = type("Completed", (), {"returncode": 0})()

        with patch.object(main, "load_config"), \
            patch.object(main, "find_free_port", return_value=12345), \
            patch.object(main, "ThreadingHTTPServer", FakeServer), \
            patch.object(main.threading, "Thread", autospec=True), \
            patch.object(main.subprocess, "run", return_value=completed) as run_mock, \
            patch.object(sys, "argv", ["main.py", "--model", "custom-cli-model", "-p", "hello"]), \
            self.assertRaises(SystemExit) as cm:
            main.main()

        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(run_mock.call_args.args[0], ["grok", "--model", "custom-cli-model", "-p", "hello"])

    def test_candidate_env_paths_prefers_launcher_env_over_cwd_and_user_env(self) -> None:
        old_here = main._HERE
        old_env = os.environ.copy()
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as home:
                repo_dir = os.path.join(home, "repo", "grok-launch")
                os.makedirs(repo_dir, exist_ok=True)
                main._HERE = repo_dir
                os.chdir(cwd)
                os.environ.clear()
                os.environ["HOME"] = home
                paths = main._candidate_env_paths()
                self.assertLess(
                    paths.index(os.path.join(repo_dir, ".env")),
                    paths.index(os.path.join(cwd, ".env")),
                )
                self.assertLess(
                    paths.index(os.path.join(repo_dir, ".env")),
                    paths.index(os.path.join(home, ".config", "grok-launch", ".env")),
                )
        finally:
            main._HERE = old_here
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def test_load_dotenv_files_launcher_env_overrides_stale_managed_env(self) -> None:
        old_here = main._HERE
        old_env = os.environ.copy()
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as repo_dir:
                main._HERE = repo_dir
                os.chdir(cwd)
                os.environ.clear()
                os.environ.update(
                    {
                        "GROK_LAUNCH_API_KEY": "old-exported-key",
                        "GROK_LAUNCH_MODEL": "old-exported-model",
                        "UNRELATED_SETTING": "shell-value",
                    }
                )
                with open(os.path.join(repo_dir, ".env"), "w", encoding="utf-8") as f:
                    f.write("GROK_LAUNCH_API_KEY=new-launcher-key\n")
                    f.write("GROK_LAUNCH_MODEL=new-launcher-model\n")
                    f.write("UNRELATED_SETTING=dotenv-value\n")

                main.load_dotenv_files()

                self.assertEqual(os.environ["GROK_LAUNCH_API_KEY"], "new-launcher-key")
                self.assertEqual(os.environ["GROK_LAUNCH_MODEL"], "new-launcher-model")
                self.assertEqual(os.environ["UNRELATED_SETTING"], "shell-value")
        finally:
            main._HERE = old_here
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def test_load_dotenv_files_launcher_env_wins_over_cwd_env(self) -> None:
        old_here = main._HERE
        old_env = os.environ.copy()
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as repo_dir:
                main._HERE = repo_dir
                os.chdir(cwd)
                os.environ.clear()
                with open(os.path.join(repo_dir, ".env"), "w", encoding="utf-8") as f:
                    f.write("GROK_LAUNCH_API_KEY=launcher-key\n")
                with open(os.path.join(cwd, ".env"), "w", encoding="utf-8") as f:
                    f.write("GROK_LAUNCH_API_KEY=cwd-key\n")

                main.load_dotenv_files()

                self.assertEqual(os.environ["GROK_LAUNCH_API_KEY"], "launcher-key")
        finally:
            main._HERE = old_here
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
