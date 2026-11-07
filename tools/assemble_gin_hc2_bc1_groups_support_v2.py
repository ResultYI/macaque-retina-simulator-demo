"""Project the support experiment through the shared group builder."""
from tools.assemble_gin_hc2_bc1_groups_v1 import (
    ROOT, OLD, read, bundle, select_members, build as _build,
)

OUT=ROOT/'reports/GIN_69_6_HC2_BC1_SUPPORT_V2'
FINE=OUT/'executions/assemble_fine_01'
NATIVE=OUT/'executions/assemble_native_01'


def build(output,guard):
    return _build(output,guard,source_out=OUT)
