from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import sys
import traceback


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
LOG = ROOT / "preview-server.log"


class PreviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    os.chdir(ROOT)
    with ThreadingHTTPServer((HOST, PORT), PreviewHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.write_text(traceback.format_exc(), encoding="utf-8")
        sys.exit(1)
