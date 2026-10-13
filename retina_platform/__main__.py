from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Local retinal simulator desktop interface")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    import tkinter as tk
    from .app import RetinaApp
    root = tk.Tk()
    app = RetinaApp(root, args.project_root)
    if args.config:
        app.load_config(args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
