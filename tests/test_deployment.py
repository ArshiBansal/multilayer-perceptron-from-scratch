"""Deployment configuration and page import smoke guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_runtime_is_pinned_to_python_313() -> None:
    runtime = PROJECT_ROOT / "runtime.txt"

    assert runtime.is_file()
    assert runtime.read_text(encoding="utf-8").strip() == "python-3.13"

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in pyproject


@pytest.mark.parametrize(
    "page",
    [
        "app.py",
        "pages/1_Prediction_Studio.py",
        "pages/2_Model_Explainability.py",
        "pages/3_Training_Experiments.py",
        "pages/4_Model_Benchmark.py",
    ],
)
def test_streamlit_page_smoke(page: str) -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=20)
    app.run()

    assert not app.exception


def test_pil_import_names_exist() -> None:
    modules = [
        PROJECT_ROOT / "pages" / "1_Prediction_Studio.py",
        PROJECT_ROOT / "src" / "image_processing.py",
        PROJECT_ROOT / "tests" / "test_image_processing.py",
    ]

    for module_path in modules:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "PIL":
                continue
            for alias in node.names:
                assert getattr(__import__("PIL"), alias.name, None) is not None


def test_page_3d_visualization_imports_are_exported() -> None:
    visualize = ast.parse(
        (PROJECT_ROOT / "src" / "visualize.py").read_text(encoding="utf-8")
    )
    exports: set[str] = set()
    for node in visualize.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            exports.update(
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )

    pages = [
        PROJECT_ROOT / "app.py",
        *(PROJECT_ROOT / "pages").glob("*.py"),
    ]
    for page_path in pages:
        tree = ast.parse(page_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {"src.visualize", "src.experiment_visualization"}:
                continue
            for alias in node.names:
                if alias.name.endswith("_3d"):
                    if node.module == "src.visualize":
                        assert alias.name in exports
