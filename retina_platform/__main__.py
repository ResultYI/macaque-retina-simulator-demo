from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Local retinal simulator desktop interface")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    # Apply the selected process-only resource profile before importing NumPy.
    import hashlib,json,os
    main_config=json.loads((args.project_root/'configurations/retina_platform_v1.json').read_bytes())
    binding=main_config.get('execution_resource_profile')
    if binding:
        path=(args.project_root/binding['path']).resolve()
        if not path.is_relative_to(args.project_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
            raise ValueError('Selected execution resource profile identity differs')
        environment=json.loads(path.read_bytes()).get('process_environment',{})
        if environment and environment!={'OPENBLAS_NUM_THREADS':'1'}:raise ValueError('Unsupported execution environment override')
        os.environ.update(environment)
    import tkinter as tk
    from .app import RetinaApp
    root = tk.Tk()
    app = RetinaApp(root, args.project_root)
    if args.config:
        app.load_config(args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
