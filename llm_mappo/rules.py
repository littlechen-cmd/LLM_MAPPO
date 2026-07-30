"""Hard task-queue rules for dynamic warehouse execution."""

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence

from llm_mappo.types import LABEL_PATTERN, PriorityAdjustment, Task, TaskStatus


class TaskQueue:
    """FIFO-by-label queue with atomic priority changes and task locking."""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._next_task_index = 1
        self._next_label_number: Dict[str, int] = {}

    @property
    def tasks(self) -> Sequence[Task]:
        return tuple(sorted(self._tasks.values(), key=lambda task: task.task_id))

    @property
    def active_tasks(self) -> Sequence[Task]:
        return tuple(
            task
            for task in self.ordered_tasks()
            if task.status != TaskStatus.COMPLETED
        )

    def ordered_tasks(self) -> Sequence[Task]:
        def order_key(task: Task):
            return task.label[0], int(task.label[1:]), task.arrival_step, task.task_id

        return tuple(sorted(self._tasks.values(), key=order_key))

    def create_batch(
        self, shelf_ids: Iterable[int], batch_id: int, letter: str, arrival_step: int
    ) -> Sequence[Task]:
        if len(letter) != 1 or not letter.isupper():
            raise ValueError("Priority letters must be one uppercase character.")

        created = []
        next_number = self._next_label_number.get(letter, 1)
        for shelf_id in shelf_ids:
            task = Task(
                task_id=f"task-{self._next_task_index:05d}",
                shelf_id=shelf_id,
                batch_id=batch_id,
                label=f"{letter}{next_number}",
                arrival_step=arrival_step,
            )
            self._tasks[task.task_id] = task
            created.append(task)
            self._next_task_index += 1
            next_number += 1
        self._next_label_number[letter] = next_number
        return tuple(created)

    def task_for_shelf(self, shelf_id: int) -> Optional[Task]:
        for task in self._tasks.values():
            if task.shelf_id == shelf_id and task.status != TaskStatus.COMPLETED:
                return task
        return None

    def task_for_agent(self, agent_id: int) -> Optional[Task]:
        for task in self._tasks.values():
            if task.assigned_agent_id == agent_id and task.status == TaskStatus.ASSIGNED:
                return task
        return None

    def assign_next(self, agent_id: int, battery: float) -> Optional[Task]:
        """Lock the next FIFO task, excluding AGVs below the 10% battery limit."""
        if battery < 0.1:
            return None
        existing = self.task_for_agent(agent_id)
        if existing is not None:
            return existing
        for task in self.ordered_tasks():
            if task.status == TaskStatus.PENDING:
                assigned = replace(
                    task, status=TaskStatus.ASSIGNED, assigned_agent_id=agent_id
                )
                self._tasks[task.task_id] = assigned
                return assigned
        return None

    def complete(self, task_id: str, completed_step: int) -> Task:
        task = self._tasks[task_id]
        completed = replace(
            task,
            status=TaskStatus.COMPLETED,
            assigned_agent_id=None,
            completed_step=completed_step,
        )
        self._tasks[task_id] = completed
        return completed

    def release_agent(self, agent_id: int) -> Sequence[Task]:
        """Return tasks locked by an unavailable AGV to the pending FIFO queue."""
        released = []
        for task in self._tasks.values():
            if task.assigned_agent_id == agent_id and task.status == TaskStatus.ASSIGNED:
                pending = replace(
                    task, status=TaskStatus.PENDING, assigned_agent_id=None
                )
                self._tasks[pending.task_id] = pending
                released.append(pending)
        return tuple(released)

    def apply_adjustments(
        self, adjustments: Iterable[PriorityAdjustment]
    ) -> Sequence[Task]:
        """Validate every adjustment first, then update all labels atomically."""
        requested = tuple(adjustments)
        if not requested:
            return ()

        active = {task.label: task for task in self.active_tasks}
        current_letters = {task.label[0] for task in active.values()}
        staged: Dict[str, str] = {}
        for adjustment in requested:
            task = active.get(adjustment.task)
            if task is None:
                raise ValueError(f"Unknown active task label: {adjustment.task}")
            if task.label in staged:
                raise ValueError(f"Task label adjusted more than once: {task.label}")
            if not LABEL_PATTERN.fullmatch(adjustment.new_label):
                raise ValueError("Priority labels must match ^[A-Z][0-9]+$.")
            if adjustment.new_label[0] not in current_letters:
                raise ValueError("Adjusted priority must use an existing letter.")
            if adjustment.new_label[1:] != task.label[1:]:
                raise ValueError(
                    "Priority adjustments must preserve the numeric suffix."
                )
            staged[task.label] = adjustment.new_label

        final_labels = [staged.get(label, label) for label in active]
        if len(final_labels) != len(set(final_labels)):
            raise ValueError(
                "Priority adjustments would create duplicate active labels."
            )

        updated = []
        for old_label, new_label in staged.items():
            task = active[old_label]
            changed = replace(task, label=new_label)
            self._tasks[changed.task_id] = changed
            updated.append(changed)
        return tuple(updated)

    def priority_weight(self, label: str) -> float:
        letters = sorted({task.label[0] for task in self.active_tasks})
        if len(letters) <= 1:
            return 1.0
        rank = letters.index(label[0])
        return 0.5 + 1.5 * (len(letters) - rank - 1) / (len(letters) - 1)

    def as_dict(self) -> List[dict]:
        return [
            {
                "task_id": task.task_id,
                "shelf_id": task.shelf_id,
                "batch_id": task.batch_id,
                "label": task.label,
                "arrival_step": task.arrival_step,
                "status": task.status.value,
                "assigned_agent_id": task.assigned_agent_id,
                "completed_step": task.completed_step,
            }
            for task in self.ordered_tasks()
        ]
