import os
import sys
import json
import threading
import subprocess
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 19090


class MockOpenAiHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            req_data = json.loads(body.decode("utf-8"))

            stream = req_data.get("stream", False)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                chunks = [
                    {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 12345, "model": "grok-mock", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                    {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 12345, "model": "grok-mock", "choices": [{"index": 0, "delta": {"content": "Hello! I am "}}]},
                    {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 12345, "model": "grok-mock", "choices": [{"index": 0, "delta": {"content": "a mock grok model."}}]},
                    {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 12345, "model": "grok-mock", "choices": [{"index": 0, "delta": {}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                resp_data = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hello! I am a mock grok model."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                }
                self.wfile.write(json.dumps(resp_data).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    global PORT
    PORT = find_free_port()
    
    server = HTTPServer(("127.0.0.1", PORT), MockOpenAiHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Mock OpenAI server started on 127.0.0.1:{PORT}")

    # Set up env
    env = os.environ.copy()
    env["GROK_LAUNCH_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["GROK_LAUNCH_MODEL"] = "grok-mock"
    env["GROK_LAUNCH_API_KEY"] = "mock-key"
    env["GROK_LAUNCH_WIRE_API"] = "chat"
    env["GROK_LAUNCH_VERBOSE"] = "1"
    
    # Path to grok-launch wrapper we installed
    wrapper_path = os.path.expanduser("~/.local/bin/grok-launch")
    if not os.path.isfile(wrapper_path):
        print(f"Error: grok-launch wrapper not found at {wrapper_path}")
        sys.exit(1)

    print("Running: grok-launch -p \"hello\"")
    try:
        proc = subprocess.run(
            [wrapper_path, "-p", "hello"],
            env=env,
            cwd="/root/projects/grok-build",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15
        )
        stdout = proc.stdout.decode("utf-8")
        stderr = proc.stderr.decode("utf-8")
        
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        
        if proc.returncode != 0:
            print(f"Failure: grok-launch exited with non-zero code {proc.returncode}")
            sys.exit(1)
            
        if "Hello! I am a mock grok model." not in stdout:
            print("Failure: expected output not found in stdout")
            sys.exit(1)
            
        print("Success: integration test passed!")
        sys.exit(0)
    except subprocess.TimeoutExpired:
        print("Failure: grok-launch timed out after 15s")
        sys.exit(1)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
