"""Local control bridge — a tiny localhost-only HTTP server that exposes the
agent tools of the *running* app to external callers (the MCP server, a future
TS frontend, curl).

Endpoints:
  GET  /ping   → {"ok": true, "app": "fantasia-conductor"}
  GET  /tools  → the Claude tool definitions (name/description/input_schema)
  POST /call   → {"name": ..., "args": {...}} → the tool's JSON result

Headless (no Qt). The caller injects ``get_definitions`` and ``execute``; in the
app, ``execute`` marshals onto the UI thread exactly like agent tool calls, so a
bridge edit behaves identically to an in-app agent edit (undoable, refreshes).
Binds 127.0.0.1 only — this is a local control surface, not a network service.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

DEFAULT_PORT = 8765


class ControlBridge:
    def __init__(self, get_definitions: Callable[[], list],
                 execute: Callable[[str, dict], object],
                 host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self._get_defs = get_definitions
        self._execute = execute
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        """Start serving in a daemon thread. False if the port is taken."""
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the console quiet
                pass

            def _send(self, code: int, payload) -> None:
                body = json.dumps(payload, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/ping":
                    self._send(200, {"ok": True, "app": "fantasia-conductor"})
                elif self.path == "/tools":
                    try:
                        self._send(200, bridge._get_defs())
                    except Exception as exc:  # noqa: BLE001
                        self._send(500, {"error": str(exc)})
                else:
                    self._send(404, {"error": "unknown endpoint"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/call":
                    self._send(404, {"error": "unknown endpoint"})
                    return
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    name = req.get("name")
                    if not name:
                        self._send(400, {"error": "missing tool name"})
                        return
                    result = bridge._execute(name, req.get("args") or {})
                    self._send(200, {"result": result})
                except Exception as exc:  # noqa: BLE001
                    self._send(500, {"error": str(exc)})

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        except OSError:
            self._server = None
            return False
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="fantasia-bridge", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
