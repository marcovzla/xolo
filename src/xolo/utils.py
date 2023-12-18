import dataclasses
from collections.abc import Iterable

from xolo.typing import H


def is_dataclass_type(obj) -> bool:
    """
    Determines if the provided object is both a class type and a dataclass.

    This function verifies whether the 'obj' is a class (i.e., a type object) and
    whether it is decorated with the dataclass decorator. This ensures that 'obj'
    is not just any class, but specifically a dataclass type, which is important
    for cases where behavior or attributes specific to dataclasses are required.

    Args:
        obj (Any): The object to be checked. It can be any Python object.

    Returns:
        bool: True if 'obj' is a class type and is decorated as a dataclass, False otherwise.
    """
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)



def is_dataclass_instance(obj) -> bool:
    """
    Determines if the given object is an instance of a dataclass.

    This function checks if 'obj' is an instance of a class that is decorated as
    a dataclass. It differentiates between the class itself (the type object) and
    its instances. This is useful when needing to verify that 'obj' is an actual
    instantiated object of a dataclass, rather than just the dataclass definition.

    Args:
        obj (Any): The object to be checked. This can be any Python object.

    Returns:
        bool: True if 'obj' is an instance of a dataclass, False otherwise.
    """
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)



def is_namedtuple_type(obj) -> bool:
    """
    Determines whether the provided object is a namedtuple type.

    This function checks if 'obj' is a subclass of tuple and whether it has
    characteristics specific to namedtuple types, such as the '_fields', '_make',
    and '_asdict' attributes. This distinguishes namedtuple types from other
    tuple-like classes or instances.

    Args:
        obj (Any): The object to be checked. It can be any Python object.

    Returns:
        bool: True if 'obj' is a namedtuple type, False otherwise.
    """
    return (
        isinstance(obj, type)
        and issubclass(obj, tuple)
        and hasattr(obj, '_fields')
        and hasattr(obj, '_make')
        and hasattr(obj, '_asdict')
    )



def is_namedtuple_instance(obj) -> bool:
    """
    Determines whether the provided object is an instance of a namedtuple.

    This function checks if 'obj' is an instance of 'tuple' and possesses the '_fields' attribute,
    which is indicative of namedtuples. Furthermore, it verifies that '_fields' is a tuple
    consisting solely of strings and that 'obj' includes the '_asdict' method, a characteristic
    feature of namedtuples. These checks ensure that the object is not just any tuple, but
    specifically an instance of a namedtuple.

    Args:
        obj (Any): The object to be checked.

    Returns:
        bool: True if 'obj' is an instance of a namedtuple, False otherwise.
    """
    return (
        isinstance(obj, tuple)
        and hasattr(obj, '_fields')
        and isinstance(obj._fields, tuple)
        and all(isinstance(field, str) for field in obj._fields)
        and hasattr(obj, '_asdict')
        and callable(obj._asdict)
    )



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
