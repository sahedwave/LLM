"""Closed execution graph enforcement for the locked nuclear LM."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Sequence

from src import config
from src.locked_artifacts import load_locked_artifacts


EXECUTION_GRAPH = (
    "BUILD",
    "FREEZE",
    "TRAIN",
    "LOCKED",
    "EVAL_ONLY",
)
ALLOWED_TRANSITIONS = {
    "BUILD": "FREEZE",
    "FREEZE": "TRAIN",
    "TRAIN": "LOCKED",
    "LOCKED": "EVAL_ONLY",
}
GRAPH_NODE_ALLOWED_STATES: Mapping[str, FrozenSet[str]] = {
    "BUILD": frozenset({"BUILD"}),
    "FREEZE": frozenset({"FREEZE"}),
    "TRAIN": frozenset({"TRAIN"}),
    "EVAL": frozenset({"EVAL_ONLY"}),
}
FUNCTION_ALLOWED_STATES: Mapping[str, FrozenSet[str]] = {
    "build_vocab": frozenset({"BUILD"}),
    "build_phase3_dataset": frozenset({"BUILD"}),
    "rebuild_locked_dataset": frozenset({"BUILD"}),
    "freeze_dataset_artifacts": frozenset({"FREEZE"}),
    "build_version_manifest": frozenset({"BUILD", "FREEZE"}),
    "load_training_bundle": frozenset({"TRAIN"}),
    "train_step": frozenset({"TRAIN"}),
    "run_training": frozenset({"TRAIN"}),
    "load_runtime": frozenset({"EVAL_ONLY"}),
    "generate_text": frozenset({"EVAL_ONLY"}),
    "run_generation": frozenset({"EVAL_ONLY"}),
    "build_runtime": frozenset({"EVAL_ONLY"}),
    "evaluate_model": frozenset({"EVAL_ONLY"}),
    "run_evaluation": frozenset({"EVAL_ONLY"}),
    "run_contract_audit": frozenset({"EVAL_ONLY"}),
}
GRAPH_MODULE_REQUIREMENTS: Mapping[str, str] = {
    "dataset_pipeline.py": "BUILD",
    "reset_and_build_dataset.py": "BUILD",
    "train.py": "TRAIN",
    "generate.py": "EVAL",
    "evaluate.py": "EVAL",
    "contract_audit.py": "EVAL",
    "dataset_auditor.py": "FREEZE",
    "dataset_report.py": "FREEZE",
}
RUNTIME_FORBIDDEN_IMPORTS: Mapping[str, FrozenSet[str]] = {
    "generate.py": frozenset({"dataset_pipeline", "reset_and_build_dataset"}),
    "evaluate.py": frozenset({"dataset_pipeline", "reset_and_build_dataset"}),
    "train.py": frozenset({"dataset_pipeline", "reset_and_build_dataset"}),
    "src/utils.py": frozenset({"dataset_pipeline", "reset_and_build_dataset"}),
}
_EXECUTION_CONTEXT: "ExecutionContext | None" = None
_BOOTSTRAPPING = True
_DAG_ACTIVE = False
_FINALIZED = False


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable singleton describing the only allowed execution universe."""

    state: str
    fingerprint: str
    allowed_modules: Mapping[str, FrozenSet[str]]
    allowed_transitions: Mapping[str, str]
    allowed_functions: Mapping[str, FrozenSet[str]]
    entrypoint: str

    def allows_module(self, graph_node: str) -> bool:
        return self.state in self.allowed_modules.get(graph_node, frozenset())

    def allows(self, function_name: str) -> bool:
        return self.state in self.allowed_functions.get(function_name, frozenset())

    @property
    def allowed_states(self) -> FrozenSet[str]:
        states: set[str] = set()
        for allowed in self.allowed_modules.values():
            states.update(allowed)
        return frozenset(states)


def _write_state(state: str) -> None:
    config.EXECUTION_STATE_PATH.write_text(json.dumps({"state": state}, indent=2) + "\n", encoding="utf-8")
    os.environ[config.EXECUTION_STATE_ENV] = state


def _read_state() -> str:
    if config.EXECUTION_STATE_PATH.exists():
        data = json.loads(config.EXECUTION_STATE_PATH.read_text(encoding="utf-8"))
        state = data.get("state")
        if state not in EXECUTION_GRAPH:
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: INVALID STATE {state}")
        return state
    return "BUILD"


def _bundle_fingerprint(bundle: Mapping[str, object]) -> str:
    manifest = bundle["manifest"]
    return "|".join(
        (
            str(manifest["dataset_hash"]),
            str(manifest["tokenizer_hash"]),
            str(manifest["vocab_size"]),
            str(bundle["manifest_id"]),
        )
    )


def _current_fingerprint(require_artifacts: bool) -> str:
    try:
        return _bundle_fingerprint(load_locked_artifacts())
    except FileNotFoundError:
        if require_artifacts:
            raise RuntimeError("ARTIFACT FINGERPRINT VIOLATION")
        return "UNFROZEN"


def _build_context(state: str, require_artifacts: bool = False) -> ExecutionContext:
    return ExecutionContext(
        state=state,
        fingerprint=_current_fingerprint(require_artifacts=require_artifacts),
        allowed_modules=GRAPH_NODE_ALLOWED_STATES,
        allowed_transitions=ALLOWED_TRANSITIONS,
        allowed_functions=FUNCTION_ALLOWED_STATES,
        entrypoint="run_pipeline.py",
    )


def state_requires_frozen_artifacts(state: str) -> bool:
    """Return whether a graph state must already have frozen artifacts."""
    return state in {"TRAIN", "LOCKED", "EVAL_ONLY"}


def _set_context(state: str, require_artifacts: bool = False) -> ExecutionContext:
    global _EXECUTION_CONTEXT
    _EXECUTION_CONTEXT = _build_context(state, require_artifacts=require_artifacts)
    return _EXECUTION_CONTEXT


def authorize_entrypoint() -> None:
    """Authorize imports that originate from the single pipeline entrypoint."""
    os.environ[config.EXECUTION_ENTRYPOINT_ENV] = "run_pipeline.py"
    state = _read_state()
    if not config.EXECUTION_STATE_PATH.exists():
        _write_state(state)
    _set_context(state, require_artifacts=state_requires_frozen_artifacts(state))


def require_entrypoint() -> None:
    """Crash if code is imported or executed outside the sanctioned entrypoint."""
    if _BOOTSTRAPPING:
        return
    if os.environ.get(config.EXECUTION_ENTRYPOINT_ENV) != "run_pipeline.py":
        raise RuntimeError("EXECUTION GRAPH VIOLATION: SIDE SCRIPT EXECUTION")


def require_runtime_execution() -> None:
    """Require the live DAG runtime instead of the temporary bootstrap import window."""
    if os.environ.get(config.EXECUTION_ENTRYPOINT_ENV) != "run_pipeline.py":
        raise RuntimeError("EXECUTION GRAPH VIOLATION: SIDE SCRIPT EXECUTION")
    if not _DAG_ACTIVE:
        raise RuntimeError("EXECUTION GRAPH VIOLATION: DAG EXECUTION REQUIRED")


def close_bootstrap_window() -> None:
    """End the bootstrap import window and enable strict root checks."""
    global _BOOTSTRAPPING
    _BOOTSTRAPPING = False


def activate_dag_execution() -> None:
    """Enable function-level execution only while the DAG is running."""
    global _DAG_ACTIVE
    require_entrypoint()
    _DAG_ACTIVE = True


def finalize_execution_graph() -> Dict[str, object]:
    """Freeze the graph after DAG completion and return the final proof bundle."""
    global _FINALIZED, _DAG_ACTIVE, _BOOTSTRAPPING
    proof = prove_no_drift()
    _FINALIZED = True
    _DAG_ACTIVE = False
    _BOOTSTRAPPING = False
    return proof


def current_state() -> str:
    """Return the current execution graph state."""
    require_entrypoint()
    state = _read_state()
    os.environ[config.EXECUTION_STATE_ENV] = state
    if _EXECUTION_CONTEXT is None or _EXECUTION_CONTEXT.state != state:
        _set_context(state, require_artifacts=state_requires_frozen_artifacts(state))
    return state


def get_execution_context(require_artifacts: bool = False) -> ExecutionContext:
    """Return the immutable execution context singleton."""
    require_entrypoint()
    state = current_state()
    if _EXECUTION_CONTEXT is None or _EXECUTION_CONTEXT.state != state:
        return _set_context(state, require_artifacts=require_artifacts)
    if require_artifacts and _EXECUTION_CONTEXT.fingerprint == "UNFROZEN":
        return _set_context(state, require_artifacts=True)
    return _EXECUTION_CONTEXT


def transition_state(next_state: str) -> str:
    """Advance the immutable state machine and reject invalid transitions."""
    require_runtime_execution()
    if next_state not in EXECUTION_GRAPH:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: UNKNOWN STATE {next_state}")
    state = current_state()
    if state == next_state:
        if state == "EVAL_ONLY":
            return state
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: RE-ENTRY INTO {state}")
    allowed = ALLOWED_TRANSITIONS.get(state)
    if allowed != next_state:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: INVALID TRANSITION {state} -> {next_state}")
    _write_state(next_state)
    _set_context(next_state, require_artifacts=state_requires_frozen_artifacts(next_state))
    return next_state


def mark_state(state: str) -> None:
    """Force the state during controlled pipeline orchestration."""
    require_runtime_execution()
    if state not in EXECUTION_GRAPH:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: UNKNOWN STATE {state}")
    _write_state(state)
    _set_context(state, require_artifacts=state_requires_frozen_artifacts(state))


def import_guard(graph_node: str, require_artifacts: bool = False) -> None:
    """Enforce entrypoint authority, graph-node admission, and locked-artifact identity."""
    if graph_node not in GRAPH_NODE_ALLOWED_STATES:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: UNKNOWN GRAPH NODE {graph_node}")
    if _BOOTSTRAPPING and not _FINALIZED:
        return
    context = get_execution_context(require_artifacts=require_artifacts)
    if not context.allows_module(graph_node):
        raise RuntimeError(
            "EXECUTION GRAPH VIOLATION: MODULE NODE {0} NOT ALLOWED IN STATE {1}".format(
                graph_node,
                context.state,
            )
        )
    if require_artifacts:
        bundle = load_locked_artifacts()
        if _bundle_fingerprint(bundle) != context.fingerprint:
            raise RuntimeError("ARTIFACT FINGERPRINT VIOLATION")


def assert_execution_allowed(function_name: str, graph_node: str | None = None) -> None:
    """Block function entry unless the state machine explicitly allows it."""
    require_runtime_execution()
    active_state = current_state()
    context = get_execution_context(
        require_artifacts=context_requires_artifacts(function_name) and active_state != "BUILD"
    )
    if graph_node is not None and not context.allows_module(graph_node):
        raise RuntimeError(
            "EXECUTION GRAPH VIOLATION: FUNCTION {0} NOT ALLOWED THROUGH NODE {1} IN STATE {2}".format(
                function_name,
                graph_node,
                context.state,
            )
        )
    if function_name not in FUNCTION_ALLOWED_STATES:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: UNKNOWN FUNCTION {function_name}")
    if not context.allows(function_name):
        raise RuntimeError(
            "EXECUTION GRAPH VIOLATION: FUNCTION {0} NOT ALLOWED IN STATE {1}".format(
                function_name,
                context.state,
            )
        )


def execution_guard(function_name: str, graph_node: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator that enforces function-level graph admission."""

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            assert_execution_allowed(function_name, graph_node)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def context_requires_artifacts(function_name: str) -> bool:
    """Return whether the function may run only once frozen artifacts exist."""
    if function_name in {"build_version_manifest", "freeze_dataset_artifacts"}:
        return False
    allowed_states = FUNCTION_ALLOWED_STATES.get(function_name, frozenset())
    return any(state != "BUILD" for state in allowed_states)


def assert_side_execution_forbidden() -> None:
    """Shared main-guard failure for all non-pipeline scripts."""
    raise RuntimeError("EXECUTION GRAPH VIOLATION: SIDE SCRIPT EXECUTION")


def validate_state(node_name: str, context: Mapping[str, object]) -> str:
    """Advance the runtime state machine to the state required by one DAG node."""
    require_runtime_execution()
    state = current_state()
    if node_name == "BUILD":
        if state != "BUILD":
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: BUILD REQUIRES STATE BUILD, FOUND {state}")
        return state
    if node_name == "FREEZE":
        if state == "BUILD":
            return transition_state("FREEZE")
        if state != "FREEZE":
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: FREEZE REQUIRES BUILD/FREEZE, FOUND {state}")
        return state
    if node_name == "TRAIN":
        if state == "FREEZE":
            return transition_state("TRAIN")
        if state != "TRAIN":
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: TRAIN REQUIRES FREEZE/TRAIN, FOUND {state}")
        return state
    if node_name == "EVAL":
        if state == "TRAIN":
            transition_state("LOCKED")
            return transition_state("EVAL_ONLY")
        if state == "LOCKED":
            return transition_state("EVAL_ONLY")
        if state != "EVAL_ONLY":
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: EVAL REQUIRES TRAIN/LOCKED/EVAL_ONLY, FOUND {state}")
        return state
    if node_name == "REPORT":
        if state != "EVAL_ONLY":
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: REPORT REQUIRES EVAL_ONLY, FOUND {state}")
        return state
    raise RuntimeError(f"EXECUTION GRAPH VIOLATION: UNKNOWN DAG NODE {node_name}")


def reachable_states(start: str = "BUILD") -> FrozenSet[str]:
    """Compute the reachable states of the linear execution graph."""
    visited: set[str] = set()
    state = start
    while state in EXECUTION_GRAPH and state not in visited:
        visited.add(state)
        state = ALLOWED_TRANSITIONS.get(state, "")
    return frozenset(visited)


def _python_files() -> List[Path]:
    return sorted(path for path in config.PROJECT_DIR.rglob("*.py") if ".venv" not in path.parts)


def _find_build_vocab_defs() -> List[Path]:
    owners: List[Path] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_vocab":
                owners.append(path)
    return owners


def _module_graph_nodes() -> Dict[str, str]:
    nodes: Dict[str, str] = {}
    for path in _python_files():
        rel = path.relative_to(config.PROJECT_DIR).as_posix()
        if rel not in GRAPH_MODULE_REQUIREMENTS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "GRAPH_NODE":
                        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: INVALID GRAPH_NODE IN {rel}")
                        nodes[rel] = node.value.value
        if rel not in nodes:
            raise RuntimeError(f"EXECUTION GRAPH VIOLATION: MISSING GRAPH_NODE IN {rel}")
    return nodes


def _find_runtime_build_calls() -> List[str]:
    violations: List[str] = []
    forbidden_patterns = (
        "build_phase3_dataset(",
        "build_vocab(",
        "_build_vocab_impl(",
    )
    for path in _python_files():
        rel = path.relative_to(config.PROJECT_DIR).as_posix()
        if rel not in RUNTIME_FORBIDDEN_IMPORTS:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                violations.append(f"{rel}: {pattern}")
    return violations


def _find_dataset_pipeline_leaks() -> List[str]:
    leaks: List[str] = []
    for path in _python_files():
        rel = path.relative_to(config.PROJECT_DIR).as_posix()
        forbidden_imports = RUNTIME_FORBIDDEN_IMPORTS.get(rel)
        if not forbidden_imports:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        leaks.append(f"{rel}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in forbidden_imports:
                    leaks.append(f"{rel}: from {node.module} import ...")
    return leaks


def _check_side_execution_guards() -> List[str]:
    violations: List[str] = []
    for path in _python_files():
        rel = path.relative_to(config.PROJECT_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if_main_nodes = []
        for node in tree.body:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                if (
                    isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                    and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.comparators[0].value == "__main__"
                ):
                    if_main_nodes.append(node)
        if not if_main_nodes:
            continue
        for node in if_main_nodes:
            calls = [
                child
                for child in node.body
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
            ]
            if rel == "run_pipeline.py":
                if not any(isinstance(call.value.func, ast.Name) and call.value.func.id == "main" for call in calls):
                    violations.append(f"{rel}: main guard does not call main()")
            else:
                if not any(
                    isinstance(call.value.func, ast.Name) and call.value.func.id == "assert_side_execution_forbidden"
                    for call in calls
                ):
                    violations.append(f"{rel}: missing side-execution guard")
    return violations


def _check_function_guard_presence() -> List[str]:
    required = {
        "dataset_pipeline.py": ("build_vocab", "build_phase3_dataset", "build_version_manifest", "freeze_dataset_artifacts"),
        "reset_and_build_dataset.py": ("rebuild_locked_dataset",),
        "train.py": ("load_training_bundle", "train_step", "run_training"),
        "generate.py": ("load_runtime", "generate_text", "run_generation"),
        "evaluate.py": ("build_runtime", "evaluate_model", "run_evaluation"),
        "contract_audit.py": ("run_contract_audit",),
    }
    violations: List[str] = []
    for rel, functions in required.items():
        text = (config.PROJECT_DIR / rel).read_text(encoding="utf-8")
        for function_name in functions:
            decorator_marker = f'@execution_guard("{function_name}"'
            inline_marker = f'assert_execution_allowed("{function_name}"'
            if decorator_marker not in text and inline_marker not in text:
                violations.append(f"{rel}: missing function guard for {function_name}")
    return violations


def prove_no_drift() -> Dict[str, object]:
    """Prove that the closed execution graph invariants hold."""
    context = get_execution_context(require_artifacts=current_state() != "BUILD")
    owners = [path.relative_to(config.PROJECT_DIR).as_posix() for path in _find_build_vocab_defs()]
    single_vocab_authority = owners == ["dataset_pipeline.py"]

    runtime_build_calls = _find_runtime_build_calls()
    dataset_pipeline_leaks = _find_dataset_pipeline_leaks()
    no_runtime_vocab_rebuild = not runtime_build_calls and not dataset_pipeline_leaks

    artifact_hashes_match = True
    if context.state != "BUILD":
        bundle = load_locked_artifacts()
        artifact_hashes_match = _bundle_fingerprint(bundle) == context.fingerprint

    execution_path_is_unique = os.environ.get(config.EXECUTION_ENTRYPOINT_ENV) == "run_pipeline.py"
    module_nodes = _module_graph_nodes()
    graph_nodes_valid = all(module_nodes.get(path) == GRAPH_MODULE_REQUIREMENTS[path] for path in GRAPH_MODULE_REQUIREMENTS)

    reachable = reachable_states("BUILD")
    allowed_states = context.allowed_states
    graph_closure_proved = (reachable & allowed_states) == allowed_states

    state_machine_is_valid = current_state() in EXECUTION_GRAPH
    side_execution_guards = _check_side_execution_guards()
    function_guards = _check_function_guard_presence()

    if not single_vocab_authority:
        raise RuntimeError(f"VOCAB DRIFT VIOLATION: build_vocab owners = {owners}")
    if not no_runtime_vocab_rebuild:
        raise RuntimeError(
            "VOCAB DRIFT VIOLATION: runtime rebuild paths detected -> {0} | leaks -> {1}".format(
                runtime_build_calls,
                dataset_pipeline_leaks,
            )
        )
    if not artifact_hashes_match:
        raise RuntimeError("ARTIFACT FINGERPRINT VIOLATION")
    if not execution_path_is_unique:
        raise RuntimeError("EXECUTION GRAPH VIOLATION: NON-UNIQUE ENTRYPOINT")
    if not graph_nodes_valid:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: GRAPH_NODE MISMATCH {module_nodes}")
    if not graph_closure_proved:
        raise RuntimeError("EXECUTION GRAPH VIOLATION: GRAPH CLOSURE PROOF FAILED")
    if not state_machine_is_valid:
        raise RuntimeError("EXECUTION GRAPH VIOLATION: INVALID STATE MACHINE")
    if side_execution_guards:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: {side_execution_guards}")
    if function_guards:
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: {function_guards}")

    return {
        "single_vocab_authority": single_vocab_authority,
        "no_runtime_vocab_rebuild": no_runtime_vocab_rebuild,
        "artifact_hashes_match": artifact_hashes_match,
        "execution_path_is_unique": execution_path_is_unique,
        "state_machine_is_valid": state_machine_is_valid,
        "graph_closure_proved": graph_closure_proved,
        "vocab_authority_owner": owners,
        "runtime_build_calls": runtime_build_calls,
        "dataset_pipeline_leaks": dataset_pipeline_leaks,
        "module_nodes": module_nodes,
        "reachable_states": sorted(reachable),
        "allowed_states": sorted(allowed_states),
    }
