import inspect
import dataclasses
from typing import Any, Optional, Union
from collections.abc import Callable
from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.fields import Field
import docstring_parser
import jsonref
from xolo.utils.symbols import prepare_symbol
from xolo.utils.common import is_dataclass


def new_schema(*args: type[Any], array: bool = False) -> dict[str, Any]:
    """
    Creates a new schema based on the provided type arguments. This function can generate
    a schema for a single type, a union of multiple types, or an array of these types.

    The function operates in the following manner:
    1. Constructs a type or a union of types based on the provided arguments.
    2. If the 'array' flag is set to True, it wraps the type(s) in a list, indicating an array of those types.
    3. Utilizes a TypeAdapter to adapt the type(s) into a schema format.
    4. Employs the 'prepare_schema' function to simplify and prepare the final schema.

    Args:
        *args (type[Any]): Variable length argument list where each argument is a type. 
        These types are used to define the elements in the schema.
        array (bool, optional): If set to True, the schema will represent an array of the specified type(s). 
        Defaults to False.

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
    if not args: raise ValueError('At least one type argument is required to create a schema.')
    t = args[0] if len(args) == 1 else Union[*args]  # type: ignore
    if array: t = list[t]
    type_adapter = TypeAdapter(t)
    return prepare_schema(type_adapter)


def schema_from_callable(f: Callable, name: Optional[str] = None) -> dict[str, Any]:
    """
    Generates a JSON schema from a callable (function or method).

    This function creates a Pydantic model from the callable, simplifies the schema,
    and structures it into a dictionary format suitable for JSON serialization.

    Args:
        f (Callable): The callable (function or method) to generate the schema from.
        name (Optional[str]): An optional name for the generated schema. Defaults to the callable's name.

    Returns:
        dict[str, Any]: The generated JSON schema as a dictionary.
    """
    model = new_model_from_callable(f, name)
    schema = dict(name=model.__name__)

    parameters = prepare_schema(model)
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
            JSON references (if 'replace_refs' is True), and unnecessary entries like 'title' (if 'keep_titles' is False) 
            will be removed or flattened for easier interpretation and use.

    Raises:
        TypeError: If the schema is not a dictionary, TypeAdapter, or Pydantic model class, a TypeError is raised.
        Other exceptions may also be raised during the processing of the schema, such as during JSON reference resolution.

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
    """
    if name is None:
        name = f.__name__

    doc = docstring_parser.parse(inspect.getdoc(f))
    description = doc.short_description or doc.long_description
    param_descriptions = {p.arg_name: p.description for p in doc.params}

    fields = {
        p.name: {
            'annotation': p.annotation,
            'default': p.default if p.default != inspect.Parameter.empty else Ellipsis,
            'description': param_descriptions.get(p.name),
        }
        for p in inspect.signature(f).parameters.values()
    }

    return new_model(name, fields, model_description=description)


def new_model_from_dataclass(c: type[Any]) -> type[BaseModel]:
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
    if not is_dataclass(c):
        raise ValueError('The provided class is not a dataclass.')

    model_name = c.__name__
    doc = docstring_parser.parse(inspect.getdoc(c))
    description = doc.short_description or doc.long_description
    field_descriptions = {p.arg_name: p.description for p in doc.params}

    field_definitions = {}
    for name, field in c.__dataclass_fields__.items():
        annotation = new_model_from_dataclass(field.type) if is_dataclass(field.type) else field.type
        default = field.default if field.default != dataclasses.MISSING else Ellipsis
        field_definitions[name] = (annotation, Field(default, description=field_descriptions.get(name)))

    return create_model(
        model_name,
        __doc__=description,
        **field_definitions,
    )
