#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def save_json_file(filename, data):
    filepath = os.path.join(ROOT, 'data', filename)
    if isinstance(data, dict) and ('users' in data or 'pillows' in data):
        key = 'users' if 'users' in data else 'pillows'
        items = data[key]
    else:
        items = data if isinstance(data, list) else []
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return len(items)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/': path = '/index.html'
        filepath = os.path.join(ROOT, path.lstrip('/'))
        if os.path.isfile(filepath):
            self.send_response(200)
            ct = 'text/html' if filepath.endswith('.html') else 'application/javascript' if filepath.endswith('.js') else 'application/json' if filepath.endswith('.json') else 'text/plain'
            self.send_header('Content-Type', ct)
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            with open(filepath, 'rb') as f: self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''
        data = json.loads(body) if body else {}

        if self.path == '/api/save_records':
            filepath = os.path.join(ROOT, 'data', 'records.json')
            existing = []
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            seen = set()
            for r in existing:
                k = str(r.get('time',''))+'_'+str(r.get('user_id',''))+'_'+str(r.get('pillow_id',''))
                seen.add(k)
            for r in data.get('records', []):
                k2 = str(r.get('time',''))+'_'+str(r.get('user_id',''))+'_'+str(r.get('pillow_id',''))
                if k2 not in seen:
                    seen.add(k2)
                    existing.append(r)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            self._json_response({'status':'ok','count':len(existing)})

        elif self.path == '/api/save_users':
            count = save_json_file('users.json', data)
            self._json_response({'status':'ok','count':count})

        elif self.path == '/api/save_pillows':
            count = save_json_file('pillows.json', data)
            self._json_response({'status':'ok','count':count})

        else:
            self.send_error(404)

    def _json_response(self, obj):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode())

if __name__ == '__main__':
    print('http://localhost:9000')
    HTTPServer(('', 9000), Handler).serve_forever()
