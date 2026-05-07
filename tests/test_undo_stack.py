from __future__ import annotations

import unittest

from arpes.core.undo import UndoFrame, UndoStack


class TestUndoStack(unittest.TestCase):
    def test_push_undo_redo(self):
        state = {"value": 0}
        stack = UndoStack()
        state["value"] = 1
        stack.push(UndoFrame(
            action="set",
            data={"old": 0, "new": 1},
            undo=lambda: state.update(value=0),
            redo=lambda: state.update(value=1),
        ))

        self.assertTrue(stack.can_undo())
        self.assertFalse(stack.can_redo())
        frame = stack.undo()
        self.assertEqual(frame.action, "set")
        self.assertEqual(state["value"], 0)
        self.assertTrue(stack.can_redo())
        stack.redo()
        self.assertEqual(state["value"], 1)

    def test_max_size_drops_oldest(self):
        values: list[int] = []
        stack = UndoStack(max_size=2)
        for i in range(3):
            stack.push(UndoFrame(
                action=str(i),
                undo=lambda i=i: values.append(i),
                redo=lambda: None,
            ))

        stack.undo()
        stack.undo()
        self.assertEqual(values, [2, 1])
        self.assertIsNone(stack.undo())

    def test_push_during_undo_is_ignored(self):
        stack = UndoStack()

        def undo():
            stack.push(UndoFrame("nested", undo=lambda: None, redo=lambda: None))

        stack.push(UndoFrame("outer", undo=undo, redo=lambda: None))
        stack.undo()
        self.assertFalse(stack.can_undo())
        self.assertTrue(stack.can_redo())


if __name__ == "__main__":
    unittest.main()
