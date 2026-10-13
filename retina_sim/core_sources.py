"""Merge independently verified new sources without rewriting frozen reference ledgers."""
from pathlib import Path
from .provenance import Registry

def load_core_registry(root):
    root=Path(root)
    registry=Registry.read(root/'sources/registry.json')
    extension=Registry.read(root/'sources/core_registry.json',root=root)
    for source in extension.sources.values(): registry.register_source(source)
    for record in extension.parameters.values(): registry.register_parameter(record)
    for profile in extension.profiles.values(): registry.register_profile(profile)
    return registry
