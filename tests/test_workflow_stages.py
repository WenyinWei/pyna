from __future__ import annotations

from pathlib import Path

import pytest

from pyna.workflow import (
    CommandStage as PublicCommandStage,
    WorkflowPlan as PublicWorkflowPlan,
)
from pyna.workflow.stages import (
    CommandStage,
    WorkflowPlan,
    compose_workflow_plans,
    run_workflow_plan,
)


def test_stage_plan_primitives_are_public_workflow_infrastructure() -> None:
    assert PublicCommandStage is CommandStage
    assert PublicWorkflowPlan is WorkflowPlan


def test_plan_composes_causal_file_stages_without_touching_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    solved = tmp_path / "solved.npz"
    replayed = tmp_path / "replayed.npz"
    solve = WorkflowPlan(
        name="solve",
        stages=(
            CommandStage(
                name="solve-500",
                argv=("python", "solve.py", "--output", str(solved)),
                inputs=(source,),
                outputs=(solved,),
                working_directory=tmp_path,
            ),
        ),
    )
    replay = WorkflowPlan(
        name="replay",
        stages=(
            CommandStage(
                name="replay",
                argv=("python", "replay.py", str(solved)),
                inputs=(solved,),
                outputs=(replayed,),
                working_directory=tmp_path,
            ),
        ),
    )

    combined = compose_workflow_plans("solve-replay", (solve, replay))

    assert tuple(stage.name for stage in combined.stages) == (
        "solve-500",
        "replay",
    )
    assert not source.exists()
    assert not solved.exists()
    assert not replayed.exists()


def test_plan_rejects_duplicate_outputs_and_future_dependencies(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared.npz"
    first = CommandStage("first", ("one",), outputs=(shared,))
    duplicate = CommandStage("duplicate", ("two",), outputs=(shared,))
    with pytest.raises(ValueError, match="produced by both"):
        WorkflowPlan("duplicate", (first, duplicate))

    consumer = CommandStage("consumer", ("read",), inputs=(shared,))
    with pytest.raises(ValueError, match="before it is produced"):
        WorkflowPlan("backwards", (consumer, first))


def test_stage_rejects_in_place_and_symlink_alias_outputs(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared.npz"
    with pytest.raises(ValueError, match="in-place input/output"):
        CommandStage(
            "in-place",
            ("mutate",),
            inputs=(shared,),
            outputs=(shared,),
        )

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    alias_directory = tmp_path / "alias"
    try:
        alias_directory.symlink_to(real_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    first = CommandStage(
        "first",
        ("one",),
        outputs=(real_directory / "result.npz",),
    )
    alias = CommandStage(
        "alias",
        ("two",),
        outputs=(alias_directory / "result.npz",),
    )
    with pytest.raises(ValueError, match="produced by both"):
        WorkflowPlan("alias-collision", (first, alias))


def test_relative_artifacts_are_resolved_per_stage_working_directory(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_output = CommandStage(
        "left",
        ("one",),
        outputs=(Path("result.npz"),),
        working_directory=left,
    )
    right_output = CommandStage(
        "right",
        ("two",),
        outputs=(Path("result.npz"),),
        working_directory=right,
    )
    WorkflowPlan("distinct-cwds", (left_output, right_output))

    absolute_alias = CommandStage(
        "absolute-alias",
        ("three",),
        outputs=(left / "result.npz",),
        working_directory=right,
    )
    with pytest.raises(ValueError, match="produced by both"):
        WorkflowPlan("same-target", (left_output, absolute_alias))

    relative_consumer = CommandStage(
        "consumer",
        ("read",),
        inputs=(Path("result.npz"),),
        working_directory=left,
    )
    WorkflowPlan("causal", (left_output, relative_consumer))
    with pytest.raises(ValueError, match="before it is produced"):
        WorkflowPlan("backwards", (relative_consumer, left_output))


def test_stage_freezes_expanded_absolute_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    work = tmp_path / "work"
    work.mkdir()
    stage = CommandStage(
        "frozen-cwd",
        ("program",),
        working_directory=Path("~/work"),
    )
    assert stage.working_directory == work.resolve()
    monkeypatch.chdir(work)
    implicit = CommandStage("implicit-cwd", ("program",))
    assert implicit.working_directory == work.resolve()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = []
    run_workflow_plan(
        WorkflowPlan("frozen", (stage,)),
        execute=True,
        printer=None,
        runner=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    assert calls[0][1]["cwd"] == work.resolve()
    assert implicit.working_directory == work.resolve()


def test_run_plan_is_dry_by_default_and_never_uses_a_shell(
    tmp_path: Path,
) -> None:
    stage = CommandStage(
        name="quoted",
        argv=("program", "value with spaces"),
        working_directory=tmp_path,
    )
    plan = WorkflowPlan("one", (stage,))
    calls: list[tuple[list[str], dict[str, object]]] = []
    printed: list[str] = []

    run_workflow_plan(
        plan,
        execute=False,
        printer=printed.append,
        runner=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    assert calls == []
    assert printed == ["program 'value with spaces'"]

    run_workflow_plan(
        plan,
        execute=True,
        printer=None,
        runner=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    assert calls == [
        (
            ["program", "value with spaces"],
            {"cwd": tmp_path, "check": True},
        )
    ]
