"""一键启动入口（等价 Spring Boot 启动类）：IDEA 中右键 Run 本文件即可，无需命令行。

自动完成：
  1. 检查并启动中间件容器（Neo4j / Qdrant / Redis，已在运行则跳过）；
  2. 启动 FastAPI 服务（与 uvicorn 命令行等价）。
启动后浏览器访问 http://127.0.0.1:8092 即可使用。
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIDDLEWARE = ["medical_neo4j", "medical_qdrant", "medical_redis"]
HOST, PORT = "127.0.0.1", 8092


def ensure_middleware() -> None:
    """确保中间件容器处于运行状态（停止态的自动 start，运行态跳过）。"""
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=30).stdout
        stopped = [c for c in MIDDLEWARE if c not in out]
        if stopped:
            subprocess.run(["docker", "start", *stopped], timeout=180)
            print(f"[main] 已启动中间件容器: {', '.join(stopped)}")
        else:
            print("[main] 中间件容器均在运行")
    except Exception as e:
        print(f"[main] 中间件容器检查失败: {e}")
        print("[main] 若 Docker Desktop 未启动，请先打开 Docker Desktop 后重新运行")


def main() -> None:
    ensure_middleware()
    import uvicorn

    print(f"[main] 服务启动中 → http://{HOST}:{PORT}  （停止：IDEA 红色方块或 Ctrl+C）")
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
