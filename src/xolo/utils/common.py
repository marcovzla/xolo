import dataclasses
from collections.abc import Iterable
from xolo.utils.typing import H



def is_dataclass(obj) -> bool:
    """
    Determines whether the provided object is both a class and a dataclass.

    This function checks two conditions: whether 'obj' is a class type, 
    and whether it is a dataclass. This is useful for ensuring that 'obj' 
    not only adheres to the structure of a dataclass but is also a class definition 
    rather than an instance of a class.

    Args:
        obj: The object to be checked. It can be any Python object.

    Returns:
        bool: True if 'obj' is a class and is a dataclass, False otherwise.
    """
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)



def is_dataclass_instance(obj) -> bool:
    """
    Determines whether the given object is an instance of a dataclass.

    This function checks if 'obj' is an instance of a dataclass, as opposed to 
    being the dataclass itself (i.e., the class definition). It does this by 
    first checking if 'obj' is a dataclass and then ensuring that it is not 
    a type (class definition). This is useful for differentiating between a 
    dataclass and an instance of that dataclass.

    Args:
        obj: The object to be checked. This can be any Python object.

    Returns:
        bool: True if 'obj' is an instance of a dataclass, False otherwise.
    """
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)



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
