#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/': path = '/index.html'
        filepath = os.path.join(ROOT, path.lstrip('/'))
        if os.path.isfile(filepath):
            self.send_response(200)
            ct = 'text/html' if filepath.endswith('.html') else 'application/javascript' if filepath.endswith('.js') else 'application/json' if filepath.endswith('.json') else 'text/plain'
            self.send_header('Content-Type', ct)
            self.end_headers()
            with open(filepath, 'rb') as f: self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/save_records':
            length = int(self.headers['Content-Length'])
            body = self.rfile.read(length)
            data = json.loads(body)
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
    print('http://localhost:9000')
    HTTPServer(('', 9000), Handler).serve_forever()
