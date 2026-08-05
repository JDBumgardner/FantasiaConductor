"""Command base + CommandBus (undo/redo + coalescing).

A :class:`Command` knows how to apply itself to a ``Project`` and reverse that.
The :class:`CommandBus` owns the current project and the undo/redo stacks, and
notifies listeners (the UI) after every change.

Two patterns matter for correct undo:

* **Capture-once**: a command records the pre-state the *first* time it runs and
  reuses it on redo — so ``do()`` is safe to call again without clobbering the
  original value. Commands use the module ``_UNSET`` sentinel for this.
* **Coalescing**: continuous edits (dragging a volume slider) return a stable
  ``merge_key`` so the bus folds a run of them into a single undo entry.

The bus is deliberately Qt-free; listeners are plain callables.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

_UNSET: Any = object()


class Command:
    """Base class for a reversible edit on a ``Project``."""

    label: str = "Edit"

    def do(self, project) -> None:  # noqa: ANN001
        raise NotImplementedError

    def undo(self, project) -> None:  # noqa: ANN001
        raise NotImplementedError

    def merge_key(self) -> Optional[tuple]:
        """Return a stable key to coalesce consecutive edits, or ``None``."""
        return None

    def merge(self, other: "Command") -> None:
        """Fold ``other`` (already applied) into ``self`` for a single undo."""
        # Default: absorb nothing. Mergeable commands override.


class CommandBus:
    """Executes commands and maintains undo/redo history for one project."""

    def __init__(self, project) -> None:  # noqa: ANN001
        self.project = project
        self._undo: List[Command] = []
        self._redo: List[Command] = []
        self._listeners: List[Callable[[Optional[Command]], None]] = []

    # ---- project / listeners --------------------------------------------
    def set_project(self, project) -> None:  # noqa: ANN001
        """Rebind to a new project (e.g. after Open) and clear history."""
        self.project = project
        self._undo.clear()
        self._redo.clear()
        self._notify(None)

    def add_listener(self, fn: Callable[[Optional[Command]], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, cmd: Optional[Command]) -> None:
        for fn in list(self._listeners):
            fn(cmd)

    # ---- dispatch / undo / redo -----------------------------------------
    def dispatch(self, cmd: Command) -> Command:
        cmd.do(self.project)
        key = cmd.merge_key()
        if key is not None and self._undo and self._undo[-1].merge_key() == key:
            self._undo[-1].merge(cmd)
        else:
            self._undo.append(cmd)
        self._redo.clear()
        self._notify(cmd)
        return cmd

    def undo(self) -> None:
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.undo(self.project)
        self._redo.append(cmd)
        self._notify(cmd)

    def redo(self) -> None:
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.do(self.project)
        self._undo.append(cmd)
        self._notify(cmd)

    # ---- introspection (for menu state) ---------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1].label if self._redo else None
