import os
from collections.abc import Callable, Hashable
from numbers import Number
from typing import Any, TypeVar

Args = tuple[Any, ...]
"""
Defines a type for representing positional arguments in a function call,
allowing for any number of arguments of any type.
"""


KwArgs = dict[str, Any]
"""
Defines a type for representing keyword arguments in a function call,
as a dictionary with string keys and values of any type.
"""


PathLike = str | os.PathLike[str]
"""
Type alias for representing file system path information.

This type can be used to annotate variables that accept either a string representing a file path
or an instance of `os.PathLike` which provides a path-like interface. It is useful for functions
that need to handle both traditional file path strings and path objects in a flexible manner.
"""


C = TypeVar('C', bound=Callable)
"""
Type variable bound to the `Callable` type.

This type variable `C` is used to represent a type that is a subtype of `Callable`. It allows
for type annotations where you specifically want to restrict a type parameter to be a callable
object, such as a function or a method. It provides flexibility while ensuring the type adheres
to the callable protocol.
"""


H = TypeVar('H', bound=Hashable)
"""
Type variable bound to the `Hashable` base class.

This type variable `H` is intended for use in situations where a type parameter needs to
be restricted to types that are hashable. By bounding it to `Hashable`, `H` can be any
type that is a subclass of the `Hashable` class. This ensures that the type parameter
accepts only hashable types, providing type safety in contexts where hashability is required,
such as keys in dictionaries.
"""


N = TypeVar('N', bound=Number)
"""
Type variable bound to the `Number` base class.

This type variable `N` is intended for use in situations where a type parameter needs to
be restricted to types that represent numerical values. By bounding it to `Number`,
`N` can be any type that is a subclass of the `Number` class, such as `int`, `float`,
or `complex`. This ensures that the type parameter accepts a wide range of numerical types,
providing both flexibility and type safety in numerical computations.
"""
