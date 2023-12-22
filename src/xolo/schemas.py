import dataclasses
import inspect
import re
from collections.abc import Callable
from types import GenericAlias
from typing import Any, Optional, Union, get_args, get_origin

import docstring_parser
import jsonref
from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.fields import Field

from xolo.symbols import prepare_symbol
from xolo.utils import is_dataclass_type


def new_schema(
        *types: type[Any],
        array: bool = False,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    """
    Creates a new schema based on the provided type arguments. This function can generate
    a schema for a single type, a union of multiple types, or an array of these types.

    The function operates in the following manner:
    1. Constructs a type or a union of types based on the provided arguments.
    2. If the 'array' flag is set to True, it wraps the type(s) in a list, indicating an array of those types.
    3. Utilizes a TypeAdapter to adapt the type(s) into a schema format.
    4. Employs the 'prepare_schema' function to simplify and prepare the final schema.

    Args:
        *types (type[Any]): Variable length argument list where each argument is a type.
        These types are used to define the elements in the schema.
        array (bool, optional): If set to True, the schema will represent an array of the specified type(s).
            Defaults to False.
        replace_refs (bool, optional): If True, resolves JSON references in the schema. Defaults to True.
        keep_titles (bool, optional): If True, retains 'title' entries in the schema. Defaults to False.
        flatten (bool, optional): If True, flattens 'allOf' entries in the schema if they contain a single clause.
            Defaults to True.

    Returns:
        dict[str, Any]: A dictionary representing the prepared schema. If 'array' is True, this
        will be a schema for an array of the specified type(s). Otherwise, it will represent the
        specified type or a union of types. The schema is simplified and ready for use, with
        resolved JSON references and unnecessary entries removed.

    Raises:
        ValueError: If no type arguments are provided, a ValueError is raised, indicating that
        at least one type argument is required to create the schema.

    Examples:
        - new_schema(int) returns a schema for integers.
        - new_schema(int, str) returns a schema for elements that can be either integers or strings.
        - new_schema(int, array=True) returns a schema for an array of integers.
        - new_schema(int, str, array=True) returns a schema for an array of elements that can be
          either integers or strings.
    """
    if not types:
        raise ValueError('At least one type argument is required to create a schema.')

    # prepare types
    types = [prepare_type_for_pydantic(t) for t in types]

    # merge types
    single_type = types[0] if len(types) == 1 else Union[*types]  # type: ignore

    # maybe wrap single_type in a list
    if array:
        single_type = list[single_type]

    # maybe wrap single_type with a TypeAdapter
    if not isinstance(single_type, type) or not issubclass(single_type, BaseModel):
        single_type = TypeAdapter(single_type)

    # return schema
    return prepare_schema(
        single_type,
        replace_refs=replace_refs,
        keep_titles=keep_titles,
        flatten=flatten,
    )


def schema_from_callable(
        f: Callable,
        name: Optional[str] = None,
        *,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    """
    Generates a JSON schema from a callable (function or method).

    This function creates a Pydantic model from the callable, simplifies the schema,
    and structures it into a dictionary format suitable for JSON serialization.

    Args:
        f (Callable): The callable (function or method) to generate the schema from.
        name (Optional[str]): An optional name for the generated schema. Defaults to the callable's name.
        replace_refs (bool, optional): If True, resolves JSON references in the schema. Defaults to True.
        keep_titles (bool, optional): If True, retains 'title' entries in the schema. Defaults to False.
        flatten (bool, optional): If True, flattens 'allOf' entries in the schema if they contain a single clause.
            Defaults to True.

    Returns:
        dict[str, Any]: The generated JSON schema as a dictionary.
    """
    model = new_model_from_callable(f, name)

    parameters = prepare_schema(
        model,
        replace_refs=replace_refs,
        keep_titles=keep_titles,
        flatten=flatten,
    )

    # assemble schema
    schema = {'name': model.__name__}
    if 'description' in parameters:
        schema['description'] = parameters.pop('description')
    schema['parameters'] = parameters

    return schema


def prepare_schema(
        schema: dict[str, Any] | TypeAdapter | type[BaseModel],
        *,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    """
    Prepares a given schema, which can be in the form of a dictionary, a Pydantic model,
    or a TypeAdapter object. The preparation process is customizable and involves the following steps:

    1. Converting the input into a schema dictionary, if it is a TypeAdapter or Pydantic model.
    2. Optionally resolving any JSON references (like $ref) present in the schema.
    3. Optionally deleting 'title' entries, depending on the 'keep_titles' flag.
    4. Optionally flattening 'allOf' entries in the schema, provided they contain a single clause.

    Args:
        schema (dict[str, Any] | TypeAdapter | type[BaseModel]): The schema to prepare. This can be a dictionary
            representing a JSON schema, a Pydantic model class, or a TypeAdapter object. The function will handle
            these different types to produce a prepared schema dictionary.
        replace_refs (bool, optional): If True, resolves JSON references in the schema. Defaults to True.
        keep_titles (bool, optional): If True, retains 'title' entries in the schema. Defaults to False.
        flatten (bool, optional): If True, flattens 'allOf' entries in the schema if they contain a single clause.
            Defaults to True.

    Returns:
        dict[str, Any]: A prepared version of the input schema. The resulting dictionary will have resolved
            JSON references (if 'replace_refs' is True), and unnecessary entries like 'title'(if 'keep_titles'
            is False) will be removed or flattened for easier interpretation and use.

    Raises:
        TypeError: If the schema is not a dictionary, TypeAdapter, or Pydantic model class, a TypeError is raised.
        Other exceptions may also be raised during the processing of the schema, such as during JSON
        reference resolution.

    Note:
        - This function does not modify the original schema object but returns a new dictionary.
        - If the input schema is already in a prepared form, the function will return it without modifications.
        - The behavior of the schema preparation can be tailored using the optional boolean flags.
    """
    # Ensure schema dictionary
    if isinstance(schema, TypeAdapter):
        schema = schema.json_schema()
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    elif not isinstance(schema, dict):
        raise TypeError('Invalid schema')

    # Resolve JSON references
    if replace_refs:
        schema = jsonref.replace_refs(schema, proxies=False)
        delete_entry(schema, '$defs')

    # Remove title entries
    if not keep_titles:
        delete_entry(schema, 'title')

    # flatten allOf when possible
    if flatten:
        flatten_entry(schema, 'allOf')

    return schema


def delete_entry(obj: Any, name: str):
    """
    Recursively deletes an entry with a given name from a dictionary or a list.

    This function iterates through the object (dict or list) and removes any occurrence
    of the entry with the specified name. If the entry is in a nested structure, it is
    also removed.

    Args:
        obj (Any): The object (dictionary or list) from which to delete the entry.
        name (str): The name of the entry to delete.

    Returns:
        None: This function modifies the object in place and does not return a value.
    """
    if isinstance(obj, dict):
        if name in obj:
            del obj[name]
        for x in obj.values():
            delete_entry(x, name)
    elif isinstance(obj, list):
        for x in obj:
            delete_entry(x, name)


def flatten_entry(obj: Any, name: str):
    """
    Recursively flattens dictionary entries that contain a single clause.

    This function iterates through a nested dictionary structure and merges dictionary entries
    with the specified 'name' if they contain only a single clause.

    Args:
        obj (Any): The dictionary or nested structure to flatten.
        name (str): The name of the key to check and merge.

    Returns:
        None: This function modifies the input 'obj' in place and does not return a value.
    """
    if isinstance(obj, dict):
        clauses = obj.get(name, [])
        if len(clauses) == 1:
            obj.pop(name)
            obj.update(clauses[0])
        for x in obj.values():
            flatten_entry(x, name)
    elif isinstance(obj, list):
        for x in obj:
            flatten_entry(x, name)


def new_model(
        model_name: str,
        fields: list[str] | dict[str, Optional[dict[str, Any]]],
        *,
        model_description: Optional[str] = None,
        default_annotation: type[Any] = str,
        default_value: Any = Ellipsis,
) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided field definitions.

    This function constructs a Pydantic model with fields defined in the `fields` argument.
    Fields can be specified as a list of field names or a dictionary with additional metadata.

    Args:
        model_name (str): The name of the model to be created.
        fields (list[str] | dict[str, Optional[dict[str, Any]]]): Field definitions.
        model_description (Optional[str]): A description of the model, used as the model's docstring.
        default_annotation (type[Any]): Default type annotation for fields. Defaults to `str`.
        default_value (Any): Default value for fields. Defaults to `Ellipsis` as a placeholder.

    Returns:
        type[BaseModel]: The dynamically created Pydantic model class.
    """
    field_definitions = {}

    for name, meta in (fields.items() if isinstance(fields, dict) else zip(fields, [None] * len(fields))):
        safe_name = prepare_symbol(name)
        field_args = meta if isinstance(meta, dict) else {}
        annotation = field_args.pop('annotation', default_annotation)
        default = field_args.pop('default', default_value)
        field_definitions[safe_name] = (annotation, Field(default, **field_args))

    return create_model(
        prepare_symbol(model_name, style='pascal'),
        __doc__=model_description,
        **field_definitions,
    )


def new_model_from_callable(f: Callable, name: Optional[str] = None) -> type[BaseModel]:
    """
    Creates a Pydantic model from a callable (function or method).

    This function utilizes the callable's signature and docstring to generate the model's field definitions.
    The model's name defaults to the callable's name if not provided.

    Args:
        f (Callable): The callable to create the model from.
        name (Optional[str]): Optional custom name for the model. Defaults to the callable's name.

    Returns:
        type[BaseModel]: The dynamically created Pydantic model class.

    Raises:
        TypeError: If 'f' is not a Callable.
    """
    if not callable(f):
        raise TypeError('The provided object is not a Callable')

    if name is None:
        name = f.__name__

    description, param_descriptions = parse_docstring(f)

    fields = {
        p.name: {
            'annotation': prepare_type_for_pydantic(p.annotation),
            'default': p.default if p.default != inspect.Parameter.empty else Ellipsis,
            'description': param_descriptions.get(p.name),
        }
        for p in inspect.signature(f).parameters.values()
    }

    return new_model(name, fields, model_description=description)


def new_model_from_dataclass(c: type[Any], name: Optional[str] = None) -> type[BaseModel]:
    """
    Creates a Pydantic model from a given dataclass.

    This function converts a dataclass into a Pydantic model, preserving the
    field definitions, annotations, and defaults. It checks if the provided class 'c'
    is a dataclass and raises a ValueError if it is not. The function also supports
    nested dataclasses, converting them into nested Pydantic models.

    Args:
        c (type[Any]): The dataclass from which the Pydantic model will be created.
                       It must be a valid dataclass.

    Returns:
        type[BaseModel]: A Pydantic model class dynamically created based on
                         the structure of the provided dataclass.

    Raises:
        ValueError: If 'c' is not a dataclass.
    """
    if not is_dataclass_type(c):
        raise ValueError('The provided class is not a dataclass')

    if name is None:
        name = c.__name__

    description, param_descriptions = parse_docstring(c)

    fields = {
        f.name: {
            'annotation': prepare_type_for_pydantic(f.type),
            'default': handle_dataclass_field_default(f.default, f.default_factory),
            'description': param_descriptions.get(f.name),
        }
        for f in dataclasses.fields(c)
    }

    return new_model(name, fields, model_description=description)



def prepare_type_for_pydantic(c: type[Any]) -> type[Any]:
    """
    Prepares a given type for compatibility with Pydantic models.

    This function is designed to adapt a variety of types, making them suitable for use
    in Pydantic models. It effectively handles not only dataclass field types but also
    other complex types, including simple types, nested structures, and parameterized
    types (e.g., List[int]). For types that are dataclasses, it employs the
    'new_model_from_dataclass' function for conversion. When dealing with parameterized
    types, it ensures the integrity of their structure is preserved during the adaptation
    process.

    Args:
        c (type[Any]): The type to be prepared. This can include simple types, nested
                       structures, dataclasses, or parameterized types.

    Returns:
        type[Any]: The prepared type, optimized for integration with Pydantic models.
                   This ensures compatibility with Pydantic's requirements, while
                   retaining the original type's structure and characteristics.
    """
    if is_dataclass_type(c):
        return new_model_from_dataclass(c)

    if isinstance(c, GenericAlias):
        tpe = get_origin(c)
        args = tuple(prepare_type_for_pydantic(arg) for arg in get_args(c))
        return tpe[args]

    return c



def handle_dataclass_field_default(default: Any, default_factory: Any) -> Any:
    """
    Determines the appropriate default value for a dataclass field.

    This function is designed to handle the default values of fields in a dataclass. It
    decides which default value should be applied based on the provided 'default' and
    'default_factory' parameters. The function supports both explicitly set default values
    and default values generated by a factory function.

    Args:
        default: The explicitly set default value for the field, if any.
        default_factory: A factory function used to generate a default value for the field.

    Returns:
        The determined default value for the field. This will be the explicitly set default
        value if available; otherwise, the value generated by the default factory. If neither
        is available, it returns an ellipsis (...), indicating the absence of a default value.

    Note:
        This function assists in accurately mapping default values of dataclass fields to
        corresponding fields in Pydantic models, ensuring consistency and correctness in
        default value handling.
    """
    if default != dataclasses.MISSING:
        return default

    if default_factory != dataclasses.MISSING:
        return default_factory()

    return ...



WS = re.compile(r'\s+')
"""Precompiled whitespace regular expression."""


def parse_docstring(obj: Any) -> tuple[Optional[str], dict[str, str]]:
    """
    Parses the docstring of a given Python object and extracts its short description and
    parameter descriptions.

    This function employs the `docstring_parser` library to parse the docstring of the
    provided object. It then cleans up the extracted text by replacing multiple
    whitespace characters with a single space. The function returns a tuple containing
    the short description of the object and a dictionary of its parameter descriptions.
    If a description is not provided for a parameter, that parameter is excluded from
    the dictionary. In case of a parsing error or if the object has no docstring, the
    function returns None for the description and an empty dictionary for parameter
    descriptions.

    Args:
        obj (Any): The object whose docstring is to be parsed. This can be any Python
                   object with a docstring, such as a function, class, or method.

    Returns:
        tuple[Optional[str], dict[str, str]]: A tuple where the first element is the
        short description of the object or None if not available. The second element is
        a dictionary where keys are parameter names and values are their descriptions.
        If there are no parameters or the docstring is unavailable, this dictionary
        will be empty.

    Raises:
        This function suppresses all exceptions and instead returns None and an empty
        dictionary in case of any error during parsing.

    Note:
        This function assumes that the docstrings are formatted in a way that is
        compatible with the `docstring_parser` library. Non-standard docstring formats
        may not be parsed correctly.
    """
    try:
        doc = docstring_parser.parse(inspect.getdoc(obj))
        description = WS.sub(' ', doc.short_description) if doc.short_description else None
        param_descriptions = {
            p.arg_name: WS.sub(' ', p.description)
            for p in doc.params
            if p.description
        }
        return description, param_descriptions
    except:
        return None, {}
