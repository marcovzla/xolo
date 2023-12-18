from collections.abc import Iterator
from dataclasses import astuple, dataclass
from typing import Optional, Self


@dataclass(frozen=True, order=True)
class Span:
    """
    Represents a range with a start and stop integer.

    Attributes:
        start (int): The starting point of the span. Must be non-negative.
        stop (int): The ending point of the span. Must be non-negative and greater than or equal to start.
    """
    start: int
    stop: int

    def __post_init__(self):
        """
        Validates the start and stop values of the span after object initialization.
        """
        Span.validate(self.start, self.stop)

    def __iter__(self):
        """
        Returns an iterator that allows the Span object to be unpacked like a tuple.

        Yields:
            The start and stop values of the Span.
        """
        return iter(astuple(self))

    @staticmethod
    def validate(start_or_stop: int, stop: Optional[int] = None) -> tuple[int, int]:
        """
        Validates that start and stop values are non-negative and start is less than or equal to stop.

        Args:
            start_or_stop (int): The start value if stop is provided, otherwise the stop value with start
                assumed to be 0.
            stop (Optional[int]): The stop value. If None, start is assumed to be 0 and start_or_stop
                is the stop value.

        Returns:
            tuple[int, int]: The validated start and stop values.

        Raises:
            ValueError: If any of the start or stop values are negative or if start is greater than stop.
        """
        if stop is None:
            start, stop = 0, start_or_stop
        else:
            start = start_or_stop
        if start < 0:
            raise ValueError(f'Start must be non-negative. Received: start={start}')
        if stop < 0:
            raise ValueError(f'Stop must be non-negative. Received: stop={stop}')
        if start > stop:
            raise ValueError(f'Start must be less than or equal to stop. Received: start={start}, stop={stop}')
        return start, stop

    @classmethod
    def generate(
            cls,
            start_or_stop: int,
            stop: Optional[int] = None,
            /, *,
            include_empty: bool = False,
            **kwargs,
    ) -> Iterator[Self]:
        """
        Generates an iterator of Span objects within a specified range.

        Args:
            start_or_stop (int): Start of the range if stop is provided, otherwise end of the range with start
                assumed to be 0.
            stop (Optional[int]): End of the range. If None, start is assumed to be 0 and start_or_stop is the end.
            include_empty (bool): If True, includes spans of length 0.

        Returns:
            Iterator[Span]: An iterator of Span objects.

        """
        start, stop = cls.validate(start_or_stop, stop)
        offset = 0 if include_empty else 1
        # By returning a generator expression, immediate validation errors can be raised
        # when generate() is called. This approach contrasts with yielding, which delays
        # error detection until the first item of the iterator is accessed.
        return (
            cls(i, j, **kwargs)
            for i in range(start, stop)
            for j in range(i + offset, stop + 1)
        )

    @classmethod
    def count(
            cls,
            start_or_stop: int,
            stop: Optional[int] = None,
            /, *,
            include_empty: bool = False,
    ) -> int:
        """
        Counts the number of spans within a specified range.

        Args:
            start_or_stop (int): Start of the range if stop is provided, otherwise end of the range with start
                assumed to be 0.
            stop (Optional[int]): End of the range. If None, start is assumed to be 0 and start_or_stop is the end.
            include_empty (bool): If True, includes counts of spans of length 0.

        Returns:
            int: The number of spans in the specified range.
        """
        start, stop = cls.validate(start_or_stop, stop)
        n = stop - start
        num_spans = n * (n + 1) // 2
        if include_empty:
            num_spans += n
        return num_spans

    def length(self) -> int:
        """
        Calculates the length of the span.

        Returns:
            int: The length of the span, computed as the difference between the stop and start values.
        """
        return self.stop - self.start

    def is_empty(self) -> bool:
        """
        Determines if the span is empty.

        Returns:
            bool: True if the span is empty (start and stop are the same), False otherwise.
        """
        return self.start == self.stop
