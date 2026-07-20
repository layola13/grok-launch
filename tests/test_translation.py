import json
import os
import sys
import unittest
import io

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
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"ls\\"}}"}}]}}]}\n',
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
        
        tool_done_event = [e for e in events if e["event"] == "response.output_item.done" and e["data"].get("item", {}).get("type") == "function_call"]
        self.assertEqual(len(tool_done_event), 1)
        self.assertEqual(tool_done_event[0]["data"]["item"]["call_id"], "call_999")
        self.assertEqual(tool_done_event[0]["data"]["item"]["name"], "exec_command")

        text_done_event = [e for e in events if e["event"] == "response.output_text.done"]
        self.assertEqual(len(text_done_event), 1)
        self.assertEqual(text_done_event[0]["data"]["text"], "Hello world")


if __name__ == "__main__":
    unittest.main()
