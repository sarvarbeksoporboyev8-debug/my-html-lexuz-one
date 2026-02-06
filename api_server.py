#!/usr/bin/env python3
"""
Simple HTTP API server for LexAI Q&A.

Endpoints:
  GET  /health          - Health check
  POST /ask             - Ask a question (JSON body: {"question": "..."})

Run with: python api_server.py
Default port: 8080 (override with PORT env var)
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Import search_ask function
from search_ask import search_ask, ask_gemini_structured, ask_perplexity_structured, GEMINI_API_KEY, PERPLEXITY_API_KEY


class LexUZHandler(BaseHTTPRequestHandler):
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
            self._send_json({
                "status": "ok",
                "perplexity": bool(PERPLEXITY_API_KEY),
                "gemini": bool(GEMINI_API_KEY),
            })
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
                
                # Get chat history if provided
                history = data.get("history", [])
                
                # Check if client wants structured response
                structured = data.get("structured", False)
                
                if structured:
                    result = None
                    
                    # Try Perplexity first (best quality)
                    if PERPLEXITY_API_KEY:
                        print("[API] Trying Perplexity structured...")
                        result = ask_perplexity_structured(question, history=history)
                    
                    # Fallback to Gemini
                    if not result and GEMINI_API_KEY:
                        print("[API] Trying Gemini structured...")
                        result = ask_gemini_structured(question, history=history)
                    
                    if result:
                        self._send_json({
                            "question": question,
                            "structured": True,
                            "blocks": result.get("blocks", []),
                            "sources": result.get("sources", {}),
                            "relatedQuestions": result.get("relatedQuestions", [])
                        })
                        return
                
                # Fallback to legacy string response
                answer = search_ask(question, history=history)
                
                self._send_json({
                    "question": question,
                    "answer": answer
                })
            
            except json.JSONDecodeError:
                self._send_error("Invalid JSON")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_error(str(e), 500)
        
        else:
            self._send_error("Not found", 404)
    
    def log_message(self, format, *args):
        print(f"[API] {args[0]}")


def main():
    port = int(os.getenv("PORT", "8080"))
    
    print(f"Starting LexAI API server...")
    print(f"  Perplexity: {'Yes' if PERPLEXITY_API_KEY else 'No'}")
    print(f"  Gemini: {'Yes' if GEMINI_API_KEY else 'No'}")
    
    server = HTTPServer(("0.0.0.0", port), LexUZHandler)
    
    print(f"\nAPI ready at http://0.0.0.0:{port}")
    print(f"  GET  /health - Health check")
    print(f"  POST /ask    - Ask a question")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
