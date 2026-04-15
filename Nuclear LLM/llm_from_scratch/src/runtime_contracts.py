"""Runtime API contracts for the locked nuclear LM pipeline."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import ast
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class ContractSpec:
    """Describe one public runtime function contract."""

    required: tuple[str, ...]
    optional_defaults: Dict[str, Any] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    targets: tuple[str, ...] = field(default_factory=tuple)


CONTRACTS: Dict[str, ContractSpec] = {
    "generate_text": ContractSpec(
        required=("query",),
        optional_defaults={
            "model": None,
            "stoi": None,
            "itos": None,
            "dataset_package": None,
            "runtime": None,
            "return_metadata": False,
        },
        aliases={"seed_text": "query", "prompt": "query"},
        targets=("generate.generate_text",),
    ),
    "train_step": ContractSpec(
        required=("model", "optimizer", "xb", "yb"),
        optional_defaults={
            "scheduler": None,
            "grad_clip": 1.0,
            "pcgs_score": 1.0,
            "pcgs_lambda": 0.0,
            "sas_score": 1.0,
            "sas_lambda": 0.0,
            "preference_pair": None,
            "dpo_lambda": 0.0,
        },
        targets=("train.train_step",),
    ),
    "evaluate_model": ContractSpec(
        required=("queries",),
        optional_defaults={
            "model": None,
            "stoi": None,
            "itos": None,
            "dataset_package": None,
            "runtime": None,
        },
        aliases={"query_set": "queries"},
        targets=("evaluate.evaluate_model",),
    ),
    "load_runtime": ContractSpec(
        required=(),
        optional_defaults={
            "dataset_package": None,
            "checkpoint_path": None,
            "require_checkpoint": False,
            "quiet": True,
        },
        targets=("generate.load_runtime", "evaluate.build_runtime"),
    ),
}


def get_contract(name: str) -> ContractSpec:
    """Return one canonical contract definition."""
    if name not in CONTRACTS:
        raise KeyError(f"Unknown runtime contract: {name}")
    return CONTRACTS[name]


def _normalize_kwargs(contract_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Apply alias mapping without forcing optional defaults before binding."""
    spec = get_contract(contract_name)
    normalized = dict(kwargs)

    for alias, canonical in spec.aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)

    return normalized


def enforce_contract(contract_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Validate a function call against the shared runtime contract."""

    spec = get_contract(contract_name)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            normalized_kwargs = _normalize_kwargs(contract_name, kwargs)
            try:
                bound = signature.bind_partial(*args, **normalized_kwargs)
            except TypeError as exc:
                raise TypeError(f"{fn.__name__} contract violation: {exc}") from exc

            arguments = dict(bound.arguments)
            for name, default in spec.optional_defaults.items():
                arguments.setdefault(name, default)

            missing = [name for name in spec.required if arguments.get(name) is None]
            if missing:
                raise TypeError(
                    f"{fn.__name__} contract violation: missing required argument(s): {', '.join(missing)}"
                )

            return fn(**arguments)

        wrapper._runtime_contract = contract_name  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _signature_issues(signature: inspect.Signature, spec: ContractSpec) -> List[str]:
    """Compare one parsed function signature with one shared contract."""
    issues: List[str] = []
    parameters = signature.parameters

    for required in spec.required:
        if required not in parameters:
            issues.append(f"missing required parameter '{required}'")

    for optional in spec.optional_defaults:
        if optional not in parameters:
            issues.append(f"missing optional parameter '{optional}'")

    return issues


def _resolve_module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise FileNotFoundError(f"Cannot resolve module path for {module_name}")
    return Path(spec.origin)


def _function_signature_from_source(module_name: str, function_name: str) -> inspect.Signature:
    path = _resolve_module_path(module_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            parameters: List[inspect.Parameter] = []
            positional = list(node.args.args)
            defaults = list(node.args.defaults)
            default_offset = len(positional) - len(defaults)
            for index, arg in enumerate(positional):
                default = inspect._empty
                if index >= default_offset:
                    default = object()
                parameters.append(
                    inspect.Parameter(
                        arg.arg,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default,
                    )
                )
            if node.args.vararg is not None:
                parameters.append(
                    inspect.Parameter(node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
                )
            for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
                default = inspect._empty if default_node is None else object()
                parameters.append(
                    inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY, default=default)
                )
            if node.args.kwarg is not None:
                parameters.append(
                    inspect.Parameter(node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
                )
            return inspect.Signature(parameters)
    raise AttributeError(f"Function {function_name} not found in {module_name}")


def check_api_drift() -> Dict[str, Any]:
    """Inspect core pipeline functions and report contract drift."""
    broken_calls: List[str] = []
    report_lines: List[str] = []

    for contract_name, spec in CONTRACTS.items():
        for target in spec.targets:
            module_name, function_name = target.rsplit(".", 1)
            try:
                signature = _function_signature_from_source(module_name, function_name)
            except Exception as exc:  # pragma: no cover - audit path
                broken_calls.append(target)
                report_lines.append(f"{target}: import failed ({exc})")
                continue

            issues = _signature_issues(signature, spec)
            if issues:
                broken_calls.append(target)
                report_lines.append(f"{target}: " + "; ".join(issues))
            else:
                report_lines.append(f"{target}: OK")

    if not broken_calls:
        drift_level = "LOW"
        status = "PASS"
    elif len(broken_calls) <= 2:
        drift_level = "MEDIUM"
        status = "FAIL"
    else:
        drift_level = "HIGH"
        status = "FAIL"

    return {
        "status": status,
        "drift_level": drift_level,
        "broken_calls": broken_calls,
        "report_lines": report_lines,
    }
