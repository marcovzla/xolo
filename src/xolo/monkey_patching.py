import functools
from types import FunctionType, MethodType
from typing import Any, Optional

from xolo.typing import C


def patch_object(obj: Any, func: C, name: Optional[str] = None):
    """
    Patch an object by adding or replacing a method with the given function.

    Args:
        obj (Any): The object to be patched.
        func (C): The function to be added or replaced in the object.
        name (Optional[str]): The name of the method to be patched. Defaults to the name of `func`.
    """
    f_copy = prepare_for_monkey_patching(func, obj, name)
    setattr(obj, f_copy.__name__, MethodType(f_copy, obj))


def patch_class(
        cls: type,
        func: C,
        name: Optional[str] = None,
        *,
        as_classmethod: bool = False,
        as_property: bool = False,
):
    """
    Patch a class by adding or replacing a method, class method, or property.

    Args:
        cls (type): The class to be patched.
        func (C): The function to be added or replaced in the class.
        name (Optional[str]): The name of the method/property to be patched. Defaults to the name of `func`.
        as_classmethod (bool): Set to True to add `func` as a class method.
        as_property (bool): Set to True to add `func` as a property.
    """
    if as_classmethod and as_property:
        raise ValueError('as_classmethod and as_property cannot be both set to True')

    f_copy = prepare_for_monkey_patching(func, cls, name)

    if as_classmethod:
        setattr(cls, f_copy.__name__, MethodType(f_copy, cls))
    elif as_property:
        setattr(cls, f_copy.__name__, property(f_copy))
    else:
        setattr(cls, f_copy.__name__, f_copy)


def patch_dunder_methods(obj: Any, **kwargs: C):
    """
    Patch the dunder methods of an object with provided functions.

    Args:
        obj (Any): The object whose dunder methods are to be patched.
        **kwargs (C): The dunder methods to be patched, provided as name=function pairs.

    Raises:
        ValueError: If any provided method name is not a dunder method.
    """
    cls = type(obj)
    patched_cls = type(f'Patched_{cls.__name__}', (cls, MonkeyPatchedDunderMethods), {})

    for name, f in kwargs.items():
        if not name.startswith('__') or not name.endswith('__'):
            raise ValueError(f'Method {name} is not a dunder method.')
        patch_class(patched_cls, f, name)

    obj.__class__ = patched_cls


def restore_dunder_methods(obj: Any, *, recursive: bool = False):
    """
    Restore the original class of an object by removing monkey-patched dunder methods.

    Args:
        obj (Any): The object to restore.
        recursive (bool): If True, recursively restore the base classes.
    """
    while has_patched_dunder_methods(obj):
        obj.__class__ = obj.__class__.__base__
        if not recursive:
            break


def has_patched_dunder_methods(obj: Any) -> bool:
    """
    Check if an object has monkey-patched dunder methods.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object has monkey-patched dunder methods, False otherwise.
    """
    return isinstance(obj, MonkeyPatchedDunderMethods)


class MonkeyPatchedDunderMethods:
    """A base class for objects with monkey-patched dunder methods."""


def prepare_for_monkey_patching(f: C, target: Any, name: Optional[str] = None) -> C:
    """
    Prepares a function for monkey patching by copying it and updating its name and qualname.

    Args:
        f (C): The function to prepare.
        target (Any): The target object or class to patch.
        name (Optional[str]): Optional new name for the function.

    Returns:
        C: A copy of the function with updated name and qualname.
    """
    # Ensure we have a name
    name = name or f.__name__

    # Get target class name
    class_name = target.__qualname__ if isinstance(target, type) else type(target).__qualname__

    # Get function that we can modify safely
    f_copy = copy_function(f)

    # Overwrite name and qualname
    f_copy.__name__ = name
    f_copy.__qualname__ = f'{class_name}.{name}'

    return f_copy


def copy_function(f: C) -> C:
    """
    Creates a copy of a function or method, allowing for safe modifications without altering the original.

    This function is particularly useful in scenarios such as monkey patching, where there's a need to modify
    or extend the behavior of existing functions or methods at runtime. By creating a copy, it ensures that
    any changes made to the function or method do not affect the original code, thereby maintaining code integrity
    and preventing side effects in other parts of the program that rely on the original behavior.

    The copied function or method retains the original's code, globals, name, defaults, and closures. In the case
    of methods (MethodType), a wrapper is created to preserve the method's characteristics.

    Args:
        f (C): The function or method to copy. This can be a regular function or a bound/unbound method.

    Returns:
        C: A copy of the provided function or method. The copy is a new, independent entity that can be modified
           or extended without impacting the original function or method.

    Raises:
        TypeError: If the provided argument 'f' is neither a FunctionType nor a MethodType, indicating that the
                   function cannot process the given input.
    """
    if isinstance(f, FunctionType):
        f_copy = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
        f_copy.__kwdefaults__ = f.__kwdefaults__
        return functools.update_wrapper(f_copy, f)

    if isinstance(f, MethodType):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper.__qualname__ = f.__func__.__name__
        wrapper.__defaults__ = f.__func__.__defaults__
        wrapper.__kwdefaults__ = f.__func__.__kwdefaults__
        return wrapper

    raise TypeError(f'Expected FunctionType or MethodType, got {type(f)}')
