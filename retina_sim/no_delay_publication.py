"""Explicit execution procedure: omit unused packing before empty link publish."""
import ast
import copy
import hashlib
from pathlib import Path
from .contracts import ContractError
from .hybrid import HybridRunner

_SOURCE = '29c5d19e495d76b8fc86c17adb5eb54235ac3fb2d73e17d1d77b8e03609b643b'
_compiled = None


def _publication(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'publish_links')


class _Guard(ast.NodeTransformer):
    count = 0

    def visit_Expr(self, node):
        if not _publication(node):
            return self.generic_visit(node)
        call = node.value
        if (len(call.args) != 2 or ast.unparse(call.args[1]) != 'self._flat(state)'):
            raise ContractError('Review publication procedure after argument changes')
        self.count += 1
        return ast.copy_location(ast.If(test=ast.parse('stimulus.delay_buffers', mode='eval').body,
                                       body=[node], orelse=[]), node)


class _StripGuards(ast.NodeTransformer):
    def visit_If(self, node):
        if (ast.unparse(node.test) == 'stimulus.delay_buffers' and len(node.body) == 1
                and not node.orelse and _publication(node.body[0])):
            return node.body[0]
        return self.generic_visit(node)


def procedure_identity():
    path = Path(__file__).with_name('hybrid.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != _SOURCE:
        raise ContractError('Review empty-publication proof after original hybrid source changes')
    return {'procedure': 'skip_empty_link_publication_v1',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_hybrid_sha256': _SOURCE, 'guarded_publication_sites': 4,
            'model_state_solver_RNG_history_checks': 'unchanged',
            'scope': 'Skip only pure unused _flat before an empty publication; original nonempty path retained'}


def no_empty_link_advance(runner):
    """Return a private bound callable; do not patch the runner or model identity."""
    global _compiled
    identity = procedure_identity()
    if (runner.advance.__func__ is not HybridRunner.advance
            or runner._flat.__func__ is not HybridRunner._flat):
        raise ContractError('Procedure requires the original advance and pure _flat implementation')
    if _compiled is None:
        tree = ast.parse(Path(__file__).with_name('hybrid.py').read_text(encoding='utf8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'HybridRunner')
        original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'advance')
        guard = _Guard()
        changed = guard.visit(copy.deepcopy(original))
        ast.fix_missing_locations(changed)
        stripped = _StripGuards().visit(copy.deepcopy(changed))
        if guard.count != 4 or ast.dump(stripped) != ast.dump(original):
            raise ContractError('Publication guards changed more than the four reviewed call sites')
        local = {}
        exec(compile(ast.Module(body=[changed], type_ignores=[]), str(Path(__file__)), 'exec'),
             HybridRunner.advance.__globals__, local)
        _compiled = local['advance']
    return _compiled.__get__(runner, type(runner)), identity
