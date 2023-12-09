"""The priority queue is a standard priority queue but also supports updating
element priorities.
"""

from typing import Any


class PriorityQueue:
    """Priority queue.

    All elements to be inserted into the queue must be hashable.

    Attributes:
        buffer: Buffer for the data.
        size: Number of elements in the queue.
        element_to_index_map: Map from the element to the index in the buffer.
    """

    def __init__(self, capacity: int):
        # Index 0 of the buffer is unused.
        self.buffer = [PriorityQueueElement(None, None)] * (
            capacity + 1)  # type: list[PriorityQueueElement]
        self.size = 0
        self.element_to_index_map = {}

    def size(self) -> int:
        """Returns the size of the queue."""
        return self.size

    def empty(self) -> bool:
        """Returns whether the queue is empty."""
        return self.size == 0

    def full(self) -> bool:
        """Returns whether the queue is full."""
        return self.size == len(self.buffer)

    def add(self, element: Any, priority: float) -> None:
        """Adds an element to the queue with the given priority.

        Args:
            element: Element to add.
            priority: Priority of the element.

        Raises:
            RuntimeError: If the queue is full.
        """
        if self.full():
            raise RuntimeError("Queue is full.")

        self.buffer[self.size + 1] = PriorityQueueElement(element, priority)
        self.element_to_index_map[element] = self.size + 1
        self.size += 1
        self._swim(self.size)

    def remove(self) -> tuple[Any, float]:
        """Removes the element with the minimum priority.

        Returns:
            A 2-tuple consisting of the removed element and its priority.

        Raises:
            RuntimeError: If the queue is empty.
        """
        if self.empty():
            raise RuntimeError("Queue is empty.")

        element = self.buffer[1]
        self.element_to_index_map.pop(element.element)
        self.buffer[1] = self.buffer[self.size]
        self.element_to_index_map[self.buffer[1].element] = 1
        self.size -= 1
        if self._in_bounds(1):
            self._sink(1)
        return element.element, element.priority

    def update(self, element: Any, priority: float) -> None:
        """Updates the given element with the given priority.

        Args:
            element: Element to update.
            priority: Updated priority of the element.
        """
        index = self.element_to_index_map[element]
        old_priority = self.buffer[index].priority
        self.buffer[index].priority = priority
        if priority < old_priority:
            self._swim(index)
        elif priority > old_priority:
            self._sink(index)

    def _in_bounds(self, index: int) -> bool:
        """Returns whether the given index is in bounds.

        Args:
            index: Index to check whether in bounds.
        """
        if index > self.size or index < 1:
            return False
        return True

    def _get_element(self, index: int) -> tuple[Any, int]:
        """Returns the element at the given index.

        Args:
            index: Index of the element to return.
        """
        if not self._in_bounds(index):
            return None
        return self.buffer[index]

    def _get_left_index(self, index: int) -> int:
        """Returns the index of the left child.

        Args:
            index: Index for which to find the left child.
        """
        return 2 * index

    def _get_right_index(self, index: int) -> int:
        """Returns the index of the right child.

        Args:
            index: Index for which to find the right child.
        """
        return 2 * index + 1

    def _get_parent_index(self, index: int) -> int:
        """Returns the index of the parent.

        Args:
            index: Index for which to get the parent index.
        """
        return index // 2

    def _min(self, index1: int, index2: int) -> int:
        """Returns the index with the smaller priority.

        Args:
            index1: Index of the element to compare.
            index2: Index of the element to compare.
        """
        element1 = self._get_element(index1)
        element2 = self._get_element(index2)
        if element1 is None:
            return index2
        if element2 is None:
            return index1
        if element1.priority < element2.priority:
            return index1
        return index2

    def _swap(self, index1: int, index2: int) -> None:
        """Swaps the elements at the given indices.

        Args:
            index1: Index to swap.
            index2: Index to swap.
        """
        self.buffer[index1], self.buffer[index2] = self.buffer[
            index2], self.buffer[index1]
        element1 = self.buffer[index1].element
        self.element_to_index_map[element1] = index1
        element2 = self.buffer[index2].element
        self.element_to_index_map[element2] = index2

    def _swim(self, index: int) -> None:
        """Bubbles up the element at the given index.

        Args:
            index: Index of the element to bubble up.
        """
        parent_index = self._get_parent_index(index)
        if self._in_bounds(parent_index) and self._min(index,
                                                       parent_index) == index:
            self._swap(index, parent_index)
            self._swim(parent_index)

    def _sink(self, index: int) -> None:
        """Bubbles down the element at the given index.

        Args:
            index: Index of the element to bubble down.
        """
        min_child_index = self._min(self._get_left_index(index),
                                    self._get_right_index(index))
        if self._in_bounds(min_child_index) and self._min(
                index, min_child_index) == min_child_index:
            self._swap(index, min_child_index)
            self._sink(min_child_index)


class PriorityQueueElement:
    """Priority queue element."""

    def __init__(self, element: Any, priority: int):
        self.element = element
        self.priority = priority
