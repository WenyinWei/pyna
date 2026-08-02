"""Small, physics-agnostic primitives for reproducible command workflows.

The objects in this module describe orchestration only.  Numerical semantics,
artifact schemas, and admission gates remain owned by the calling package.
Stage working directories are frozen to absolute resolved paths at
construction.  Commands are always passed to ``subprocess`` as argument
vectors; a shell is never involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable, Iterable


CommandRunner = Callable[..., Any]


def _path_key(
    value: Path,
    working_directory: Path | None = None,
) -> str:
    """Return a stage-relative stable key without requiring the path to exist."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        base = (
            Path.cwd()
            if working_directory is None
            else Path(working_directory).expanduser()
        )
        if not base.is_absolute():
            base = Path.cwd() / base
        path = base / path
    return os.path.normcase(str(path.resolve(strict=False)))


@dataclass(frozen=True)
class CommandStage:
    """One immutable command stage with declared file dependencies."""

    name: str
    argv: tuple[str, ...]
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    working_directory: Path | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        argv = tuple(str(value) for value in self.argv)
        inputs = tuple(Path(value) for value in self.inputs)
        outputs = tuple(Path(value) for value in self.outputs)
        if self.working_directory is None:
            working_directory = Path.cwd().resolve()
        else:
            working_directory = Path(self.working_directory).expanduser()
            if not working_directory.is_absolute():
                working_directory = Path.cwd() / working_directory
            working_directory = working_directory.resolve(strict=False)
        if not name:
            raise ValueError("workflow stage name must not be empty")
        if not argv or not argv[0]:
            raise ValueError(f"workflow stage {name!r} has no executable")
        if any("\x00" in value for value in argv):
            raise ValueError(f"workflow stage {name!r} contains a NUL argument")
        if len(
            {_path_key(path, working_directory) for path in outputs}
        ) != len(outputs):
            raise ValueError(f"workflow stage {name!r} repeats an output path")
        overlap = {
            _path_key(path, working_directory) for path in inputs
        } & {
            _path_key(path, working_directory) for path in outputs
        }
        if overlap:
            raise ValueError(
                f"workflow stage {name!r} declares an in-place input/output; "
                "write a distinct artifact instead"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "working_directory", working_directory)

    def shell_display(self) -> str:
        """Render the argument vector for humans without executing a shell."""

        return shlex.join(self.argv)


@dataclass(frozen=True)
class WorkflowPlan:
    """Validated, ordered composition of :class:`CommandStage` objects."""

    name: str
    stages: tuple[CommandStage, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        stages = tuple(self.stages)
        if not name:
            raise ValueError("workflow plan name must not be empty")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError(f"workflow plan {name!r} repeats a stage name")

        producers: dict[str, int] = {}
        for index, stage in enumerate(stages):
            for output in stage.outputs:
                key = _path_key(output, stage.working_directory)
                if key in producers:
                    previous = stages[producers[key]].name
                    raise ValueError(
                        f"workflow output {output} is produced by both "
                        f"{previous!r} and {stage.name!r}"
                    )
                producers[key] = index

        for index, stage in enumerate(stages):
            for input_path in stage.inputs:
                producer = producers.get(
                    _path_key(input_path, stage.working_directory)
                )
                if producer is not None and producer >= index:
                    raise ValueError(
                        f"workflow stage {stage.name!r} consumes {input_path} "
                        "before it is produced"
                    )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "stages", stages)

    def then(
        self,
        *stages: CommandStage,
        name: str | None = None,
    ) -> "WorkflowPlan":
        """Return a new plan with stages appended and the full plan rechecked."""

        return WorkflowPlan(
            name=self.name if name is None else name,
            stages=(*self.stages, *stages),
        )


def compose_workflow_plans(
    name: str,
    plans: Iterable[WorkflowPlan],
) -> WorkflowPlan:
    """Compose ordered plans while retaining all dependency checks."""

    stages = tuple(stage for plan in plans for stage in plan.stages)
    return WorkflowPlan(name=name, stages=stages)


def run_workflow_plan(
    plan: WorkflowPlan,
    *,
    execute: bool,
    printer: Callable[[str], Any] | None = print,
    runner: CommandRunner = subprocess.run,
) -> tuple[str, ...]:
    """Print and optionally execute a plan, returning rendered commands.

    ``runner`` is injectable for tests or an application-owned runtime.  The
    default runner receives ``check=True`` and never receives ``shell=True``.
    """

    rendered: list[str] = []
    for stage in plan.stages:
        display = stage.shell_display()
        rendered.append(display)
        if printer is not None:
            printer(display)
        if execute:
            runner(
                list(stage.argv),
                cwd=stage.working_directory,
                check=True,
            )
    return tuple(rendered)


__all__ = [
    "CommandStage",
    "WorkflowPlan",
    "compose_workflow_plans",
    "run_workflow_plan",
]
