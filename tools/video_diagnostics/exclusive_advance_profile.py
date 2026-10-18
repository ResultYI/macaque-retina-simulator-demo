"""Reversible process-local timing of original function ASTs; no numerical edits."""
import ast
from collections import defaultdict
from contextlib import contextmanager
import copy
import functools
import inspect
from pathlib import Path
import time


class ExclusiveTimer:
    def __init__(self):
        self.stack = []
        self.rows = defaultdict(lambda: {'calls': 0, 'inclusive_seconds': 0., 'exclusive_seconds': 0.})
        self.edges = defaultdict(int)
        self.solver_depth = 0
        self.solver_results = []

    @contextmanager
    def region(self, name):
        parent = self.stack[-1] if self.stack else None
        frame = [name, time.perf_counter(), 0.]
        self.stack.append(frame)
        if name == 'solver.call':
            self.solver_depth += 1
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[1]
            if name == 'solver.call':
                self.solver_depth -= 1
            assert self.stack.pop() is frame
            row = self.rows[name]
            row['calls'] += 1
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += elapsed - frame[2]
            if parent is not None:
                parent[2] += elapsed
                self.edges[parent[0] + ' -> ' + name] += 1

    def call(self, name, operation):
        with self.region(name):
            result = operation()
        if name == 'solver.call':
            self.solver_results.append({'nfev': result['nfev'], 'accepted_steps': result['steps'],
                                        'dense_output_points': len(result['t'])})
        return result

    def report(self):
        root = self.rows['advance.other']['inclusive_seconds']
        total = sum(row['exclusive_seconds'] for row in self.rows.values())
        if self.stack or abs(root - total) > 1e-7:
            raise AssertionError('Nested timing does not conserve advance wall time')
        return {'root_seconds': root, 'sum_exclusive_seconds': total,
                'accounting_error_seconds': total - root,
                'rows': dict(sorted(self.rows.items(), key=lambda item: -item[1]['exclusive_seconds'])),
                'nested_calls': dict(self.edges),
                'solver_results': list(self.solver_results),
                'semantics': 'Only exclusive_seconds are additive; inclusive_seconds overlap their descendants'}


FUNCTION_REGIONS = {
    'advance': 'advance.other', '_sample': 'sample.other', '_integrate': 'integration.prepare_finalize',
    'record': 'record.values', 'record_requested': 'record.dispatch',
    'flush_events': 'events.flush', 'update_delays': 'history.observation_delay',
    'publish_links': 'history.publish_links',
}


class TimingAST(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        name = FUNCTION_REGIONS.get(node.name)
        if name:
            wrapper = ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Name(id='_probe_region', ctx=ast.Load()), args=[ast.Constant(name)], keywords=[]))],
                body=node.body, type_comment=None)
            node.body = [ast.copy_location(wrapper, node)]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        expression = ast.unparse(node.func)
        name = None
        if expression == 'validate_state':
            name = 'state.validation'
        elif expression == 'integrate':
            name = 'solver.call'
        elif expression == 'self.model.evaluate':
            name = 'signals.CPU_evaluate'
        elif expression in ('np.concatenate', 'np.stack', 'self.model.unpack') or expression.endswith('.copy'):
            name = 'state.array_copy'
        elif expression in ('fork_timestamped_delay_payloads', 'json.loads', 'json.dumps'):
            name = 'history.fork_or_JSON'
        elif expression.endswith('.sample_spikes'):
            name = 'events.spikes_RNG'
        elif expression.endswith('.transition'):
            name = 'sample.causal_transition'
        elif expression.endswith(('.next_arrival', '.prune', '.payload')):
            name = 'history.arrival_prune_payload'
        elif expression == 'execution_metadata':
            name = 'record.metadata'
        elif expression == 'NetworkStimulus':
            name = 'input.stimulus_construction'
        elif expression == 'plan.indices':
            name = 'record.index_selection'
        if not name:
            return node
        return ast.copy_location(ast.Call(func=ast.Name(id='_probe_call', ctx=ast.Load()),
            args=[ast.Constant(name), ast.Lambda(args=ast.arguments(posonlyargs=[], args=[],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=node)], keywords=[]), node)


class StripTiming(ast.NodeTransformer):
    def visit_With(self, node):
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
            call = node.items[0].context_expr
            if isinstance(call.func, ast.Name) and call.func.id == '_probe_region':
                return [self.visit(statement) for statement in node.body]
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == '_probe_call':
            return self.visit(node.args[1].body)
        return self.generic_visit(node)


@contextmanager
def instrument(timer, output):
    import retina_sim.hybrid as hybrid
    import retina_sim.gpu_hybrid as gpu
    import retina_sim.contracts as contracts
    replacements, globals_to_restore, proofs = [], [], []
    try:
        for module in (hybrid, gpu):
            for key, value in (('_probe_call', timer.call), ('_probe_region', timer.region)):
                if key in module.__dict__:
                    raise RuntimeError('Probe global collision')
                module.__dict__[key] = value
                globals_to_restore.append((module, key))
        for owner, names in ((hybrid.HybridRunner, ('advance', '_sample')),
                             (gpu.CudaHybridRunner, ('_integrate',))):
            module = inspect.getmodule(owner)
            path = Path(inspect.getsourcefile(owner))
            tree = ast.parse(path.read_text(encoding='utf8'))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner.__name__)
            for name in names:
                original_node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
                changed = TimingAST().visit(copy.deepcopy(original_node))
                ast.fix_missing_locations(changed)
                stripped = StripTiming().visit(copy.deepcopy(changed))
                if ast.dump(stripped, include_attributes=False) != ast.dump(original_node, include_attributes=False):
                    raise AssertionError('Instrumentation changed original AST: ' + name)
                local = {}
                exec(compile(ast.Module(body=[changed], type_ignores=[]), str(path), 'exec'), module.__dict__, local)
                original = getattr(owner, name)
                replacements.append((owner, name, original))
                setattr(owner, name, local[name])
                (output / (owner.__name__ + '_' + name + '_instrumented.txt')).write_text(ast.unparse(changed) + '\n', encoding='utf8')
                proofs.append({'function': owner.__name__ + '.' + name, 'original_source': str(path),
                               'stripped_instrumentation_AST_identical': True})
        for name in ('_flat', '_replace'):
            original = getattr(hybrid.HybridRunner, name)
            replacements.append((hybrid.HybridRunner, name, original))

            def make_wrapper(operation):
                @functools.wraps(operation)
                def timed(*args, **kwargs):
                    with timer.region('state.flat_replace'):
                        return operation(*args, **kwargs)
                return timed

            setattr(hybrid.HybridRunner, name, make_wrapper(original))
        original_finite = contracts.finite_array
        replacements.append((contracts, 'finite_array', original_finite))

        @functools.wraps(original_finite)
        def timed_finite(*args, **kwargs):
            # Keep the solver inclusive boundary intact. Observe constructor
            # copies/checks only outside the logical solve call.
            if not timer.stack or timer.solver_depth:
                return original_finite(*args, **kwargs)
            with timer.region('state.constructor_copy_and_finite_check'):
                return original_finite(*args, **kwargs)

        contracts.finite_array = timed_finite
        yield proofs
    finally:
        for owner, name, original in reversed(replacements):
            setattr(owner, name, original)
        for module, key in globals_to_restore:
            del module.__dict__[key]
