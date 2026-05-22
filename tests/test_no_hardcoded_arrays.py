import ast
from pathlib import Path

import pytest

FIGURE_SCRIPTS = list(Path("figures").rglob("generate_*.py"))

ALLOWED_NAMES = {
    "METRIC_ORDER",
    "METRIC_LABELS",
    "METRIC_LABELS_HRV",
    "CATEGORY_MAP",
    "CATEGORY_ORDER",
    "DIRECTION_LABELS",
    "PERIOD_TICKS",
    "PANEL_LAYOUT",
    "COLORS",
    "STYLE",
    "FIGSIZE",
}


@pytest.mark.parametrize("script", FIGURE_SCRIPTS, ids=lambda p: p.name)
def test_no_long_numeric_literals(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if target_names & ALLOWED_NAMES:
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                elts = node.value.elts
                num_count = sum(
                    1
                    for e in elts
                    if (isinstance(e, ast.Constant) and isinstance(e.value, (int, float)))
                    or (
                        isinstance(e, ast.UnaryOp)
                        and isinstance(e.operand, ast.Constant)
                        and isinstance(e.operand.value, (int, float))
                    )
                )
                assert num_count <= 5, (
                    f"{script.name} line {node.lineno}: numeric literal sequence of "
                    f"length {num_count} must be loaded from data/derived/ instead"
                )
