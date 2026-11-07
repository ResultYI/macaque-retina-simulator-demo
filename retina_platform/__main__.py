from __future__ import annotations

import argparse
from pathlib import Path
import sys


def enable_high_dpi():
    """Use native screen pixels instead of Windows stretching the entire Tk window."""
    if sys.platform != 'win32':
        return
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    try:
        set_awareness = user32.SetProcessDpiAwarenessContext
    except AttributeError:
        user32.SetProcessDPIAware()
    else:
        set_awareness.argtypes = [ctypes.c_void_p]
        set_awareness.restype = wintypes.BOOL
        set_awareness(ctypes.c_void_p(-2))  # System DPI; must precede the first Tk window.


def main():
    parser = argparse.ArgumentParser(description="Local retinal simulator desktop interface")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cell6-replay", action="store_true", help="Read-only replay of verified Cell6 caches; no model execution")
    parser.add_argument("--registry", type=Path, help="Cell6 replay cache registry")
    parser.add_argument("--response-replay", action="store_true", help="Read-only spatial playback of complete response directories or ZIP packages")
    parser.add_argument("--internal-states", action="store_true", help="Read saved internal states and current 69#6 results")
    parser.add_argument("--result", type=Path, help="Complete response ZIP or directory")
    args = parser.parse_args()
    enable_high_dpi()
    import tkinter as tk
    root = tk.Tk()
    if args.internal_states or (args.response_replay and args.result is None):
        from .internal_state_view import InternalStateApp
        InternalStateApp(root, args.project_root, result=args.result)
    elif args.response_replay:
        from .response_view import ResponseApp
        ResponseApp(root, args.project_root, result=args.result if args.result.exists() else None)
    elif args.cell6_replay:
        from .replay import ReplayApp
        ReplayApp(root, args.project_root, registry=args.registry)
    else:
        from .app import RetinaApp
        app = RetinaApp(root, args.project_root)
        if args.config:
            app.load_config(args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
