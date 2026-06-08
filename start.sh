#!/bin/bash
# 枕头压力测试系统 一键启动

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/pillow_server.log"
CFLOG="/tmp/pillow_cf.log"
URLFILE="/tmp/pillow_public_url.txt"

echo "=== 枕头压力测试系统 ==="
echo ""

# Kill old instances
pkill -f "server.py" 2>/dev/null
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 2

# Start web server
cd "$DIR/pressure_sensor_visualization/web"
nohup python3 server.py > "$LOG" 2>&1 &
sleep 2

if ! lsof -i :8080 | grep -q Python; then
    echo "[错误] 服务器启动失败:"
    tail -5 "$LOG"
    exit 1
fi
echo "[OK] 本地服务: http://localhost:8080"

LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
if [ -n "$LOCAL_IP" ]; then
    echo "[OK] 局域网:   http://$LOCAL_IP:8080"
fi

# Start Cloudflare tunnel
echo -n "[..] 启动公网隧道..."
nohup cloudflared tunnel --url http://localhost:8080 > "$CFLOG" 2>&1 &
for i in $(seq 1 15); do
    sleep 1
    URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' "$CFLOG" 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
        echo ""
        echo "[OK] 公网访问: $URL"
        echo "$URL" > "$URLFILE"
        break
    fi
    echo -n "."
done

if [ -z "$URL" ]; then
    echo ""
    echo "[警告] 公网隧道启动较慢，请稍后查看: cat $CFLOG"
fi

echo ""
echo "服务运行中，关闭此窗口停止所有服务。"
echo ""

# Keep running
wait
