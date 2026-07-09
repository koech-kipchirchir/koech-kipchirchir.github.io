"""
AIOS Desktop — entry point.

Usage:
    python -m desktop.main [--dev] [--port PORT]
"""

from __future__ import annotations

import argparse
import sys

from desktop.app import create_app, create_splash
from desktop.window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIOS Desktop")
    parser.add_argument("--dev", action="store_true", help="Enable developer mode")
    parser.add_argument("--port", type=int, default=8000, help="Backend API port")
    parser.add_argument("--theme", choices=["dark", "light"], default=None, help="Force theme")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    app = create_app(sys.argv)

    if args.theme:
        app.apply_theme(args.theme)

    splash = create_splash()
    splash.show()
    app.processEvents()

    window = MainWindow(app)
    window.show()

    splash.finish(window)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
