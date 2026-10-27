from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Local retinal simulator desktop interface")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cell6-replay", action="store_true", help="Read-only replay of verified Cell6 caches; no model execution")
    parser.add_argument("--registry", type=Path, help="Cell6 replay cache registry")
    args = parser.parse_args()
    import tkinter as tk
    root = tk.Tk()
    if args.cell6_replay:
        from .replay import ReplayApp
        ReplayApp(root, args.project_root, registry=args.registry)
        root.mainloop()
        return
    from .app import RetinaApp
    app = RetinaApp(root, args.project_root)
    if args.config:
        app.load_config(args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
