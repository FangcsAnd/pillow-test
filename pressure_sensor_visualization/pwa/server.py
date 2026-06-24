#!/usr/bin/env python3
"""Minimal server for pillow PWA - supports saving records.json"""
import http.server, json, os, cgi

ROOT = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
        if self.path == '/api/save_records':
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            filepath = os.path.join(ROOT, 'data', 'records.json')
            existing = []
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            for r in data.get('records', []):
                existing.append(r)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status':'ok','count':len(existing)}).encode())
        else:
            self.send_error(404)

if __name__ == '__main__':
    port = 9000
    print(f'PWA server running at http://localhost:{port}')
    http.server.HTTPServer(('', port), Handler).serve_forever()
