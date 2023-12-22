import asyncio
import itertools
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from typing import Generic, Self, TypeVar, overload

A = TypeVar('A')
B = TypeVar('B')


_sentinel = object()


class PeekableIterator(Generic[A]):
    """An iterator that allows peeking at the next element without consuming it."""

    def __init__(self, items: Iterable[A]):
        """Initialize the PeekableIterator with an iterable."""
        self._iter = iter(items)
        self._cache: deque[A] = deque()

    def __iter__(self) -> Self:
        """Return the iterator itself."""
        return self

    def __next__(self) -> A:
        """Return the next item from the iterator or from the cache."""
        return self._cache.popleft() if self._cache else next(self._iter)

    def __bool__(self) -> bool:
        """Return True if there are more items to iterate over."""
        try:
            self.peek()
        except StopIteration:
            return False
        else:
            return True

    @overload
    def peek(self) -> A:
        ...

    @overload
    def peek(self, default: A) -> A:
        ...

    @overload
    def peek(self, default: B) -> A | B:
        ...

    def peek(self, default=_sentinel):
        """
        Peek at the next item without consuming it.

        Args:
            default: A default value to return if no more items are available.

        Returns:
            The next item or the default value if no items are available.

        Raises:
            StopIteration: If no default is provided and no more items are available.
        """
        if not self._cache:
            try:
                self._cache.append(next(self._iter))
            except StopIteration:
                if default is _sentinel:
                    raise
                return default
        return self._cache[0]

    def prepend(self, *items: A):
        """Prepend items to the beginning of the iterator."""
        self._cache.extendleft(reversed(items))

    def append(self, *items: A):
        """Append items to the end of the iterator."""
        self._iter = itertools.chain(self._iter, items)



async def sync_to_async_iterator(items: Iterable[A]) -> AsyncIterator[A]:
    """
    Converts a synchronous iterable into an asynchronous iterator.

    This function takes an iterable (items) as input and yields its elements asynchronously.
    It is useful when you have a synchronous iterable but need to use it in an asynchronous context.

    Args:
        items (Iterable[A]): The iterable to be converted into an asynchronous iterator.

    Yields:
        AsyncIterator[A]: An asynchronous iterator yielding elements from the input iterable.

    Examples:
        async for item in sync_to_async_iterator([1, 2, 3]):
            print(item)
    """
    for item in items:
        # Using asyncio.sleep to yield control back to the event loop
        await asyncio.sleep(0)
        yield item



def async_to_sync_iterator(items: AsyncIterable[A], loop: asyncio.AbstractEventLoop) -> Iterator[A]:
    """
    Converts an asynchronous iterable into a synchronous iterator.

    This function takes an asynchronous iterable (items) and an event loop (loop) as inputs,
    and yields elements from the asynchronous iterable in a synchronous manner. This is particularly
    useful when integrating asynchronous code with synchronous code.

    Args:
        items (AsyncIterable[A]): The asynchronous iterable to be converted into a synchronous iterator.
        loop (asyncio.AbstractEventLoop): The event loop on which the asynchronous operations will be run.

    Yields:
        Iterator[A]: A synchronous iterator yielding elements from the input asynchronous iterable.

    Examples:
        loop = asyncio.get_event_loop()
        for item in async_to_sync_iterator(async_iterable, loop):
            print(item)

    Note:
        This function blocks the current thread until the next item of the async iterable is ready.
        Therefore, it should be used cautiously to avoid blocking the event loop for prolonged periods.
    """
    try:
        ait = aiter(items)
        while True:
            yield loop.run_until_complete(anext(ait))
    except StopAsyncIteration:
        pass
