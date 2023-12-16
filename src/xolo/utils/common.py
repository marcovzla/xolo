from collections.abc import Iterable
from xolo.utils.typing import H



def deduplicate_preserve_order(xs: Iterable[H]) -> list[H]:
    """
    Removes duplicate elements from an iterable while preserving their original order.

    This function takes an iterable and returns a list containing the unique elements from
    the iterable, maintaining their original order. Duplicate elements are removed,
    and the order of the remaining elements is preserved.

    Args:
        iterable (Iterable[H]): The input iterable containing elements to be deduplicated.

    Returns:
        list[H]: A list containing the unique elements from the input iterable in their
        original order.

    Example:
        >>> input_list = [3, 2, 1, 2, 3, 4, 5, 4, 6]
        >>> deduplicated_list = deduplicate_preserve_order(input_list)
        >>> deduplicated_list
        [3, 2, 1, 4, 5, 6]
    """
    return list(dict.fromkeys(xs))
