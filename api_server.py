#!/usr/bin/env python3
"""
Simple HTTP API server for LexUZ Q&A.

Endpoints:
  GET  /health          - Health check
  POST /ask             - Ask a question (JSON body: {"question": "..."})
  GET  /stats           - Get database statistics

Run with: python api_server.py
Default port: 8080 (override with PORT env var)
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Import from main module
from lexuz_mvp_sqlite_patched import Config, Pipeline, rag_ask


class LexUZHandler(BaseHTTPRequestHandler):
    cfg = None
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    
    def _send_error(self, message: str, status: int = 400):
        self._send_json({"error": message}, status)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == "/health":
            self._send_json({"status": "ok"})
        
        elif path == "/stats":
            try:
                pipe = Pipeline(self.cfg)
                cur = pipe.store.conn.cursor()
                
                docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                docs_ok = cur.execute("SELECT COUNT(*) FROM documents WHERE status='ok'").fetchone()[0]
                docs_error = cur.execute("SELECT COUNT(*) FROM documents WHERE status='error'").fetchone()[0]
                chunks = cur.execute("SELECT COUNT(*) FROM chunks WHERE active=1").fetchone()[0]
                vectors = cur.execute("SELECT COUNT(*) FROM vectors WHERE active=1").fetchone()[0]
                
                pipe.close()
                
                self._send_json({
                    "documents": docs,
                    "documents_ok": docs_ok,
                    "documents_error": docs_error,
                    "chunks": chunks,
                    "vectors": vectors,
                })
            except Exception as e:
                self._send_error(str(e), 500)
        
        else:
            self._send_error("Not found", 404)
    
    def do_POST(self):
        path = urlparse(self.path).path
        
        if path == "/ask":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body) if body else {}
                
                question = data.get("question", "").strip()
                if not question:
                    self._send_error("Missing 'question' field")
                    return
                
                answer = rag_ask(self.cfg, question)
                self._send_json({"question": question, "answer": answer})
            
            except json.JSONDecodeError:
                self._send_error("Invalid JSON")
            except Exception as e:
                self._send_error(str(e), 500)
        
        else:
            self._send_error("Not found", 404)
    
    def log_message(self, format, *args):
        print(f"[API] {args[0]}")


def main():
    port = int(os.getenv("PORT", "8080"))
    
    print(f"Loading configuration...")
    LexUZHandler.cfg = Config.from_env()
    
    print(f"Starting API server on port {port}...")
    server = HTTPServer(("0.0.0.0", port), LexUZHandler)
    
    print(f"API ready at http://0.0.0.0:{port}")
    print(f"  GET  /health - Health check")
    print(f"  GET  /stats  - Database statistics")
    print(f"  POST /ask    - Ask a question")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
