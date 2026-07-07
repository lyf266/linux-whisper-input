#!/usr/bin/env python3
import socket
import os
import sys
import json

def main():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    socket_path = os.path.join(runtime_dir, "voice-input.sock")
    if not os.path.exists(socket_path):
        print(f"Service not running (socket {socket_path} not found)")
        # Show desktop notification of failure
        try:
            import subprocess
            subprocess.run([
                "notify-send",
                "-t", "2000",
                "❌ 语音输入服务未启动",
                "请运行 systemctl --user start voice-input.service"
            ])
        except Exception:
            pass
        sys.exit(1)
        
    payload = {
        "command": "toggle",
        "env": {
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY", ""),
            "DISPLAY": os.environ.get("DISPLAY", ""),
            "XDG_RUNTIME_DIR": runtime_dir
        }
    }
    
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        client.sendall(json.dumps(payload).encode("utf-8"))
        client.close()
    except Exception as e:
        print(f"Error connecting to service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
