# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze and print field dependencies from icon4py factories."""

from __future__ import annotations

import ast
import pathlib
from collections import defaultdict
from typing import Annotated, Final

import typer

from . import _common as common


cli = typer.Typer(no_args_is_help=True, name="field-deps", help=__doc__)


# --- Paths to source files ---

_MODEL_COMMON_SRC: Final = (
    common.REPO_ROOT / "model" / "common" / "src" / "icon4py" / "model" / "common"
)

_ATTRS_FILES: Final[dict[str, pathlib.Path]] = {
    "geometry": _MODEL_COMMON_SRC / "grid" / "geometry_attributes.py",
    "interpolation": _MODEL_COMMON_SRC / "interpolation" / "interpolation_attributes.py",
    "metrics": _MODEL_COMMON_SRC / "metrics" / "metrics_attributes.py",
}

_FACTORY_FILES: Final[dict[str, pathlib.Path]] = {
    "geometry": _MODEL_COMMON_SRC / "grid" / "geometry.py",
    "interpolation": _MODEL_COMMON_SRC / "interpolation" / "interpolation_factory.py",
    "metrics": _MODEL_COMMON_SRC / "metrics" / "metrics_factory.py",
}


# --- Constant parsing ---


def _parse_constants(file_path: pathlib.Path) -> dict[str, str]:
    """Parse an attrs module and extract {CONSTANT_NAME: "string_value"} for all Final[str] constants."""
    source = file_path.read_text()
    tree = ast.parse(source)
    constants: dict[str, str] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper() and node.value is not None:
                resolved = _resolve_constant_value(node.value, constants)
                if resolved is not None:
                    constants[name] = resolved
    return constants


def _resolve_constant_value(node: ast.expr, known: dict[str, str]) -> str | None:
    """Resolve a constant assignment value, supporting string literals, f-strings, and references."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                inner = _resolve_constant_value(value.value, known)
                if inner is None:
                    return None
                parts.append(inner)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    return None


def _build_lookups() -> (
    tuple[dict[str, str], dict[str, str], dict[str, str]]
):
    """Build lookup tables from all attrs modules.

    Returns:
        name_to_value: {PYTHON_CONSTANT_NAME: "field_string_value"}
        value_to_name: {"field_string_value": PYTHON_CONSTANT_NAME}
        value_to_category: {"field_string_value": "geometry"|"interpolation"|"metrics"}
    """
    name_to_value: dict[str, str] = {}
    value_to_name: dict[str, str] = {}
    value_to_category: dict[str, str] = {}

    for category, file_path in _ATTRS_FILES.items():
        constants = _parse_constants(file_path)
        for const_name, const_value in constants.items():
            name_to_value[const_name] = const_value
            value_to_name[const_value] = const_name
            value_to_category[const_value] = category

    return name_to_value, value_to_name, value_to_category


# --- AST resolution helpers ---


def _resolve_ast_value(node: ast.expr, name_to_value: dict[str, str]) -> str | None:
    """Resolve an AST expression node to a field name string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        # e.g. attrs.Z_MC, geometry_attrs.CELL_AREA
        if node.attr in name_to_value:
            return name_to_value[node.attr]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                inner = _resolve_ast_value(value.value, name_to_value)
                if inner is None:
                    return None
                parts.append(inner)
            else:
                return None
        return "".join(parts)
    return None


def _extract_dict_values(node: ast.Dict, name_to_value: dict[str, str]) -> set[str]:
    """Extract resolved string values from a dict's values."""
    result = set()
    for v in node.values:
        resolved = _resolve_ast_value(v, name_to_value)
        if resolved is not None:
            result.add(resolved)
    return result


def _extract_dict_keys(node: ast.Dict, name_to_value: dict[str, str]) -> set[str]:
    """Extract resolved string values from a dict's keys."""
    result = set()
    for k in node.keys:
        if k is not None:
            resolved = _resolve_ast_value(k, name_to_value)
            if resolved is not None:
                result.add(resolved)
    return result


def _extract_tuple_elements(node: ast.Tuple, name_to_value: dict[str, str]) -> set[str]:
    """Extract resolved string values from a tuple's elements."""
    result = set()
    for elt in node.elts:
        resolved = _resolve_ast_value(elt, name_to_value)
        if resolved is not None:
            result.add(resolved)
    return result


# --- Provider extraction ---


def _get_call_func_name(node: ast.Call) -> str | None:
    """Get the function/class name from a Call node."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _get_keyword(node: ast.Call, name: str) -> ast.expr | None:
    """Get the value of a keyword argument from a Call node."""
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _extract_standard_provider(
    call_node: ast.Call, name_to_value: dict[str, str]
) -> tuple[set[str], set[str]] | None:
    """Extract (fields, deps) from ProgramFieldProvider/NumpyDataProvider/EmbeddedFieldOperatorProvider."""
    fields: set[str] = set()
    deps: set[str] = set()

    fields_arg = _get_keyword(call_node, "fields")
    if fields_arg is not None:
        if isinstance(fields_arg, ast.Dict):
            fields = _extract_dict_values(fields_arg, name_to_value)
        elif isinstance(fields_arg, ast.Tuple):
            fields = _extract_tuple_elements(fields_arg, name_to_value)

    deps_arg = _get_keyword(call_node, "deps")
    if deps_arg is not None and isinstance(deps_arg, ast.Dict):
        deps = _extract_dict_values(deps_arg, name_to_value)

    if fields:
        return fields, deps
    return None


def _extract_precomputed_provider(
    call_node: ast.Call, name_to_value: dict[str, str]
) -> tuple[set[str], set[str]] | None:
    """Extract (fields, deps) from PrecomputedFieldProvider (keys of dict arg, no deps)."""
    if call_node.args and isinstance(call_node.args[0], ast.Dict):
        fields = _extract_dict_keys(call_node.args[0], name_to_value)
        if fields:
            return fields, set()
    return None


def _extract_inverse_provider(
    call_node: ast.Call, name_to_value: dict[str, str]
) -> tuple[set[str], set[str]] | None:
    """Extract (fields, deps) from self._inverse_field_provider(field_name) calls."""
    if call_node.args:
        resolved = _resolve_ast_value(call_node.args[0], name_to_value)
        if resolved is not None:
            inverse_name = f"inverse_of_{resolved}"
            return {inverse_name}, {resolved}
    return None


def _extract_provider_info(
    call_node: ast.Call, name_to_value: dict[str, str]
) -> tuple[set[str], set[str]] | None:
    """Extract (fields, deps) from any recognized provider constructor call."""
    func_name = _get_call_func_name(call_node)

    if func_name == "PrecomputedFieldProvider":
        return _extract_precomputed_provider(call_node, name_to_value)
    if func_name in ("ProgramFieldProvider", "NumpyDataProvider", "EmbeddedFieldOperatorProvider"):
        return _extract_standard_provider(call_node, name_to_value)
    if func_name == "_inverse_field_provider":
        return _extract_inverse_provider(call_node, name_to_value)
    if func_name == "SparseFieldProviderWrapper":
        # Fields from `fields=` keyword, deps resolved from wrapped provider (handled separately)
        fields_arg = _get_keyword(call_node, "fields")
        if fields_arg is not None and isinstance(fields_arg, ast.Tuple):
            fields = _extract_tuple_elements(fields_arg, name_to_value)
            if fields:
                return fields, set()  # deps filled in later from wrapped provider
    return None


# --- Factory analysis ---


def _analyze_factory(file_path: pathlib.Path, name_to_value: dict[str, str]) -> dict[str, set[str]]:
    """Parse a factory source file and extract the dependency graph.

    Processes the AST sequentially (statement by statement) to correctly handle
    variable assignments in conditional branches (match/case for geometry types).

    Returns: {field_name: {dep1, dep2, ...}}
    """
    source = file_path.read_text()
    tree = ast.parse(source)
    graph: dict[str, set[str]] = {}
    var_providers: dict[str, tuple[set[str], set[str]]] = {}

    def _add_to_graph(fields: set[str], deps: set[str]) -> None:
        for field in fields:
            if field in graph:
                graph[field] |= deps
            else:
                graph[field] = set(deps)

    def _process_assignment(node: ast.Assign) -> None:
        """Process an assignment statement, tracking provider variable bindings."""
        if not isinstance(node.value, ast.Call):
            return
        func_name = _get_call_func_name(node.value)

        # Handle SparseFieldProviderWrapper: resolve deps from wrapped provider immediately
        if func_name == "SparseFieldProviderWrapper":
            info = _extract_provider_info(node.value, name_to_value)
            if info is not None:
                wrapper_fields, _ = info
                wrapped_deps: set[str] = set()
                if node.value.args:
                    first_arg = node.value.args[0]
                    if isinstance(first_arg, ast.Name) and first_arg.id in var_providers:
                        _, wrapped_deps = var_providers[first_arg.id]
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_providers[target.id] = (wrapper_fields, wrapped_deps)
            return

        # Handle regular provider constructors
        info = _extract_provider_info(node.value, name_to_value)
        if info is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_providers[target.id] = info

    def _process_expr(node: ast.Expr) -> None:
        """Process an expression statement, looking for register_provider calls."""
        if not isinstance(node.value, ast.Call):
            return
        call = node.value
        func = call.func
        is_register = isinstance(func, ast.Attribute) and func.attr == "register_provider"
        if not is_register or not call.args:
            return

        arg = call.args[0]
        if isinstance(arg, ast.Call):
            info = _extract_provider_info(arg, name_to_value)
            if info is not None:
                _add_to_graph(*info)
        elif isinstance(arg, ast.Name) and arg.id in var_providers:
            _add_to_graph(*var_providers[arg.id])

    def _process_stmts(stmts: list[ast.stmt]) -> None:
        """Process a list of statements sequentially."""
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                _process_assignment(stmt)
            elif isinstance(stmt, ast.Expr):
                _process_expr(stmt)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _process_stmts(stmt.body)
            elif isinstance(stmt, ast.ClassDef):
                _process_stmts(stmt.body)
            elif isinstance(stmt, (ast.If, ast.With)):
                _process_stmts(stmt.body)
                if hasattr(stmt, "orelse") and stmt.orelse:
                    _process_stmts(stmt.orelse)
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    # Save and restore var_providers for each case branch
                    saved = dict(var_providers)
                    _process_stmts(case.body)
                    var_providers.update(saved)
            elif isinstance(stmt, (ast.For, ast.While)):
                _process_stmts(stmt.body)
            elif isinstance(stmt, ast.Try):
                _process_stmts(stmt.body)
                for handler in stmt.handlers:
                    _process_stmts(handler.body)
                if stmt.orelse:
                    _process_stmts(stmt.orelse)
                if stmt.finalbody:
                    _process_stmts(stmt.finalbody)

    _process_stmts(tree.body)
    return graph


def build_dependency_graph() -> dict[str, set[str]]:
    """Build the combined dependency graph from all factory source files."""
    name_to_value, _, _ = _build_lookups()
    graph: dict[str, set[str]] = {}

    for category, file_path in _FACTORY_FILES.items():
        factory_graph = _analyze_factory(file_path, name_to_value)
        for field, deps in factory_graph.items():
            if field in graph:
                graph[field] |= deps
            else:
                graph[field] = set(deps)

    # Ensure dependency-only fields (leaf nodes) are in the graph
    all_deps: set[str] = set()
    for deps in graph.values():
        all_deps.update(deps)
    for dep in all_deps:
        if dep not in graph:
            graph[dep] = set()

    return graph


# --- Graph traversal ---


def _get_transitive_deps(field: str, graph: dict[str, set[str]]) -> set[str]:
    """Get all transitive dependencies for a field (everything it needs)."""
    visited: set[str] = set()
    stack = [field]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for dep in graph.get(current, set()):
            if dep not in visited:
                stack.append(dep)
    visited.discard(field)
    return visited


def _build_reverse_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """Build a reverse graph: {field: set of fields that directly depend on it}."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for field, deps in graph.items():
        for dep in deps:
            reverse[dep].add(field)
    return dict(reverse)


def _get_transitive_dependents(field: str, graph: dict[str, set[str]]) -> set[str]:
    """Get all fields that transitively depend on this field."""
    reverse = _build_reverse_graph(graph)
    visited: set[str] = set()
    stack = [field]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for dependent in reverse.get(current, set()):
            if dependent not in visited:
                stack.append(dependent)
    visited.discard(field)
    return visited


# --- Display ---


def _format_field(value: str, value_to_name: dict[str, str]) -> str:
    """Format a field for display, showing both string value and constant name if available."""
    const_name = value_to_name.get(value)
    if const_name:
        return f"{value} ({const_name})"
    return value


def _print_dep_tree(
    field: str,
    graph: dict[str, set[str]],
    value_to_name: dict[str, str],
    _indent: str = "",
    _visited: set[str] | None = None,
    _is_last: bool = True,
    _is_root: bool = True,
) -> None:
    """Print a dependency tree using box-drawing characters."""
    if _visited is None:
        _visited = set()

    if _is_root:
        typer.echo(_format_field(field, value_to_name))
    else:
        connector = "\u2514\u2500\u2500 " if _is_last else "\u251c\u2500\u2500 "
        marker = " [circular]" if field in _visited else ""
        typer.echo(f"{_indent}{connector}{_format_field(field, value_to_name)}{marker}")

    if field in _visited:
        return
    _visited = _visited | {field}

    deps = sorted(graph.get(field, set()))
    if _is_root:
        child_indent = ""
    else:
        child_indent = _indent + ("\u2502   " if not _is_last else "    ")

    for i, dep in enumerate(deps):
        is_last_dep = i == len(deps) - 1
        _print_dep_tree(
            dep, graph, value_to_name, child_indent, _visited, is_last_dep, _is_root=False
        )


def _resolve_field_name(
    name: str, name_to_value: dict[str, str], value_to_name: dict[str, str]
) -> str | None:
    """Resolve a user-provided name to a field string value.

    Tries: exact string value match, then Python constant name, then case-insensitive.
    """
    # Exact match as string value
    if name in value_to_name:
        return name

    # Match as Python constant name
    if name in name_to_value:
        return name_to_value[name]

    # Case-insensitive match on constant names
    name_lower = name.lower()
    for const_name, value in name_to_value.items():
        if const_name.lower() == name_lower:
            return value

    # Case-insensitive match on string values
    for value in value_to_name:
        if value.lower() == name_lower:
            return value

    return None


# --- CLI commands ---


@cli.command()
def deps(
    field_name: Annotated[
        str,
        typer.Argument(
            help="Field name: Python constant (e.g. Z_MC) or string value (e.g. 'height')."
        ),
    ],
    reverse: Annotated[
        bool,
        typer.Option("--reverse", "-r", help="Also show fields that depend on this field."),
    ] = False,
) -> None:
    """Print the dependency tree for a field."""
    name_to_value, value_to_name, value_to_category = _build_lookups()
    graph = build_dependency_graph()

    resolved = _resolve_field_name(field_name, name_to_value, value_to_name)
    if resolved is None:
        typer.echo(f"Error: unknown field '{field_name}'.", err=True)
        typer.echo("Use 'list-fields' to see all known fields.", err=True)
        raise typer.Exit(1)

    if resolved not in graph:
        typer.echo(f"Error: field '{resolved}' not found in any factory.", err=True)
        raise typer.Exit(1)

    # Print dependencies
    typer.echo(f"Dependencies for {_format_field(resolved, value_to_name)}:\n")
    direct_deps = graph.get(resolved, set())
    if not direct_deps:
        typer.echo("  (no dependencies - precomputed or leaf field)")
    else:
        _print_dep_tree(resolved, graph, value_to_name)

    if reverse:
        typer.echo(f"\nDependents of {_format_field(resolved, value_to_name)}:\n")
        dependents = _get_transitive_dependents(resolved, graph)
        if not dependents:
            typer.echo("  (no dependents)")
        else:
            reverse_graph = _build_reverse_graph(graph)
            _print_dep_tree(resolved, reverse_graph, value_to_name)


@cli.command()
def list_fields() -> None:
    """List all known fields grouped by factory."""
    name_to_value, value_to_name, value_to_category = _build_lookups()
    graph = build_dependency_graph()

    # Group fields by category
    by_category: dict[str, list[str]] = defaultdict(list)
    for field in sorted(graph.keys()):
        category = value_to_category.get(field, "other")
        by_category[category].append(field)

    total = sum(len(fields) for fields in by_category.values())
    typer.echo(f"All fields ({total} total):\n")

    for category in ("geometry", "interpolation", "metrics", "other"):
        fields = by_category.get(category, [])
        if not fields:
            continue
        typer.echo(f"  {category.upper()} ({len(fields)} fields):")
        for field in sorted(fields):
            n_deps = len(graph.get(field, set()))
            dep_info = f" [{n_deps} deps]" if n_deps > 0 else " [precomputed]"
            typer.echo(f"    {_format_field(field, value_to_name)}{dep_info}")
        typer.echo()
