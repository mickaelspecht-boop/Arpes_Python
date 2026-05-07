"""Pile undo/redo générique pour actions UI réversibles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class UndoFrame:
    action: str
    undo: Callable[[], None]
    redo: Callable[[], None]
    data: dict = field(default_factory=dict)


class UndoStack:
    def __init__(self, max_size: int = 50):
        self.max_size = max(1, int(max_size))
        self._undo: list[UndoFrame] = []
        self._redo: list[UndoFrame] = []
        self._active = False

    def push(self, frame: UndoFrame) -> None:
        if self._active:
            return
        self._undo.append(frame)
        if len(self._undo) > self.max_size:
            del self._undo[: len(self._undo) - self.max_size]
        self._redo.clear()

    def undo(self) -> UndoFrame | None:
        if not self._undo:
            return None
        frame = self._undo.pop()
        self._active = True
        try:
            frame.undo()
        finally:
            self._active = False
        self._redo.append(frame)
        return frame

    def redo(self) -> UndoFrame | None:
        if not self._redo:
            return None
        frame = self._redo.pop()
        self._active = True
        try:
            frame.redo()
        finally:
            self._active = False
        self._undo.append(frame)
        return frame

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
