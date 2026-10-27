"""Reuse dead CPU inner temporaries; retain every instruction and FP64 operation."""
from bisect import bisect_right
import numpy as np

from .contracts import ContractError
from .compiled_kernel import (ScratchWorkspace, ADD, SUB, MUL, DIV, POW,
    SIGMOID, LOG, EXP, MINIMUM, MAXIMUM, POSITIVE_MASK, CSR, ROW_CSR,
    TAKE, SCATTER, ZERO, FINITE, NONNEG, POSITIVE)


def reuse_inner_scratch(scratch, program, derivatives, input_size):
    """Keep inputs/constants fixed for each call; reuse only dead temporaries.

    Intervals include their final read instruction, so its destination cannot
    overwrite an operand. Derivative sources stay live until result assembly.
    Native fused layouts contain embedded addresses and are deliberately refused.
    """
    binary = {ADD, SUB, MUL, DIV, POW, SIGMOID, LOG, EXP, MINIMUM, MAXIMUM, POSITIVE_MASK}
    checks = {FINITE, NONNEG, POSITIVE}
    allowed = binary | checks | {CSR, ROW_CSR, TAKE, SCATTER, ZERO}
    if (scratch._values is not None or not 0 <= input_size <= scratch.shape[0]
            or program.ndim != 2 or program.shape[1] != 7):
        raise ContractError('Scratch reuse requires an unmaterialized complete inner plan')
    blocks = {}
    for offset, values in scratch.constants:
        if len(values):
            blocks[int(offset)] = dict(size=len(values), first=-1, last=len(program), constant=values)
    for i, row in enumerate(program):
        op, dest, n, *_ = map(int, row)
        if op not in allowed:
            raise ContractError('Scratch reuse does not rewrite native/unknown address layouts')
        if op in checks or op == SCATTER or n == 0:
            continue
        if dest in blocks:
            raise ContractError('Scratch reuse requires distinct original allocations')
        blocks[dest] = dict(size=n, first=i, last=i, constant=None)
    starts = sorted(blocks)
    end = input_size
    for offset in starts:
        block = blocks[offset]
        if offset < end or block['size'] <= 0:
            raise ContractError('Scratch allocation overlaps inputs or another allocation')
        end = offset + block['size']
        if end > scratch.shape[0]:
            raise ContractError('Scratch allocation exceeds its owned buffer')

    def owner(offset, size):
        if size == 0:
            return None
        if offset < 0 or size < 0:
            raise ContractError('Negative scratch reference')
        if offset < input_size and offset + size <= input_size:
            return None
        index = bisect_right(starts, offset) - 1
        if index < 0 or offset + size > starts[index] + blocks[starts[index]]['size']:
            raise ContractError('Scratch reference has no complete allocation')
        return starts[index]

    def touch(offset, size, step):
        key = owner(offset, size)
        if key is not None:
            block = blocks[key]
            if block['first'] >= step:
                raise ContractError('Scratch input is read before its definition')
            block['last'] = max(block['last'], step)

    for i, row in enumerate(program):
        op, dest, n, a, an, b, bn = map(int, row)
        if op in checks:
            touch(a, n, i)
        elif op != ZERO:
            touch(a, an, i)
            if op in binary:
                touch(b, bn, i)
            elif op == SCATTER and n:
                key = owner(dest, 1)
                if key is None or blocks[key]['constant'] is not None:
                    raise ContractError('Scatter must target a prior mutable temporary')
                touch(key, blocks[key]['size'], i)
    for dest, source, n in derivatives:
        if dest < 0 or n < 0 or dest + n > input_size:
            raise ContractError('Derivative destination exceeds original state/input region')
        if source >= 0:
            touch(int(source), int(n), len(program))

    relocated = {}
    constants = []
    fixed_end = input_size
    for offset in starts:
        block = blocks[offset]
        if block['constant'] is not None:
            relocated[offset] = fixed_end
            constants.append((fixed_end, block['constant']))
            fixed_end += block['size']
    arena_end = fixed_end
    free, active = [], []
    for offset, block in sorted(blocks.items(), key=lambda item: item[1]['first']):
        if block['constant'] is not None:
            continue
        step = block['first']
        remaining = []
        for key in active:
            old = blocks[key]
            if old['last'] < step:
                free.append((relocated[key], old['size']))
            else:
                remaining.append(key)
        active = remaining
        merged = []
        for start, size in sorted(free):
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((start, size))
        free = merged
        size = block['size']
        for j, (start, available) in enumerate(free):
            if available >= size:
                relocated[offset] = start
                free[j:j+1] = [(start+size, available-size)] if available > size else []
                break
        else:
            relocated[offset] = arena_end
            arena_end += size
        active.append(offset)

    def address(offset, size):
        if size == 0:
            return 0
        key = owner(offset, size)
        return offset if key is None else relocated[key] + offset-key

    compact = program.copy()
    for i, row in enumerate(program):
        op, dest, n, a, an, b, bn = map(int, row)
        if op in checks:
            compact[i, 3] = address(a, n)
            continue
        compact[i, 1] = address(dest, 1 if op == SCATTER and n else n)
        if op != ZERO:
            compact[i, 3] = address(a, an)
            if op in binary:
                compact[i, 5] = address(b, bn)
    output = derivatives.copy()
    for i, (_, source, n) in enumerate(derivatives):
        if source >= 0:
            output[i, 1] = address(int(source), int(n))
    workspace = ScratchWorkspace(arena_end, constants)
    return workspace, compact, output, dict(
        policy='cpu_inner_dead_temporary_reuse_v1', original_bytes=scratch.nbytes,
        compact_bytes=workspace.nbytes, saved_bytes=scratch.nbytes-workspace.nbytes,
        fixed_input_and_constant_bytes=fixed_end*8,
        instructions_reordered=False, arithmetic_changed=False)
