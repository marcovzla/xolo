import re
from collections.abc import Iterable
from typing import Literal, assert_never

from slugify import slugify

CaseStyle = Literal['snake', 'camel', 'pascal', 'kebab']


def prepare_symbol(
        text: str,
        *,
        style: CaseStyle = 'snake',
        lowercase: bool = False,
        allow_unicode: bool = True,
        **kwargs,
) -> str:
    """
    Transforms a given text string into a safe and standardized symbol, typically for use as an identifier,
    file name, or any other context where a clean, readable string is required.

    This function utilizes 'python-slugify' to normalize the text. The customization options include setting
    the case style (camel, pascal, snake, kebab), determining if the returned symbol should be in lowercase,
    and deciding whether to allow Unicode characters.

    Args:
        text (str): The text to be transformed into a safe symbol.
        style (Literal['camel', 'pascal', 'snake', 'kebab'], optional): Specifies the naming convention
            to be applied to the symbol. Defaults to 'snake'.
        lowercase (bool, optional): Determines if the returned symbol should be in lowercase. Applies before
            converting to the specified case style. Defaults to False.
        allow_unicode (bool, optional): If True, Unicode characters are allowed in the symbol. Defaults to True.
        **kwargs: Additional keyword arguments that are passed to the 'slugify' function.

    Returns:
        str: The transformed symbol, which is a normalized version of the input text according to the specified
        parameters. The symbol is in the specified case style.

    Examples:
        - prepare_symbol("Hello World!") returns "hello_world".
        - prepare_symbol("Hello World!", case_style='pascal') returns "HelloWorld".
        - prepare_symbol("Hello World!", case_style='camel') returns "helloWorld".
        - prepare_symbol("Hello World!", case_style='kebab') returns "hello-world".
        - prepare_symbol("Привет мир!", allow_unicode=False, case_style='snake') returns "privet_mir".
        - prepare_symbol("Привет мир!", allow_unicode=True, case_style='pascal') returns "ПриветМир".

    Note:
        The behavior of this function can be further customized by additional keyword arguments that are specific
        to the underlying 'slugify' function. The 'lowercase' option applies before the case style conversion.
    """
    if not text or not isinstance(text, str):
        raise ValueError('Input must be a non-empty string.')

    separator = '_' if style == 'snake' else '-'

    symbol = slugify(
        text=text,
        separator=separator,
        lowercase=lowercase,
        allow_unicode=allow_unicode,
        **kwargs,
    )

    if style in ('camel', 'pascal'):
        symbol = join_symbol(symbol.split(separator), style)

    return symbol


def convert_case_style(symbol: str, from_style: CaseStyle, to_style: CaseStyle) -> str:
    """
    Converts a symbol from one case style to another.

    Args:
        symbol (str): The symbol to be converted.
        from_style (CaseStyle): The current case style of the symbol.
        to_style (CaseStyle): The target case style to convert the symbol to.

    Returns:
        str: The symbol converted to the target case style.

    Raises:
        ValueError: If the input string is empty or not a valid string.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError('Input must be a non-empty string.')

    return join_symbol(split_symbol(symbol, from_style), to_style)


def split_symbol(symbol: str, style: CaseStyle) -> list[str]:
    """
    Splits a symbol into its constituent parts based on the specified case style.

    Args:
        symbol (str): The symbol to be split.
        style (CaseStyle): The case style used in the symbol.

    Returns:
        list[str]: A list of parts obtained from splitting the symbol.

    Raises:
        ValueError: If the input string is empty or not a valid string.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError('Input must be a non-empty string.')

    match style:
        case 'snake':
            return symbol.split('_')

        case 'kebab':
            return symbol.split('-')

        case 'camel' | 'pascal':
            return re.findall(r'.+?(?:(?<=[a-zA-Z])(?=[0-9])|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', symbol)

        case _:
            assert_never(style)


def join_symbol(parts: Iterable[str], style: CaseStyle, *, preserve_case: bool = False) -> str:
    """
    Joins the parts of a symbol into a single string based on the specified case style.

    Args:
        parts (Iterable[str]): The parts to be joined.
        style (CaseStyle): The case style to be applied to the joined symbol.
        preserve_case (bool, optional): If True, preserves the original case of the parts. Defaults to False.

    Returns:
        str: The symbol formed by joining the parts in the specified case style.

    Raises:
        ValueError: If the parts are not a valid iterable of strings.
    """
    if not isinstance(parts, Iterable) or not all(isinstance(part, str) for part in parts):
        raise ValueError('Parts must be an iterable of strings.')

    match style:
        case 'snake':
            if not preserve_case:
                parts = [p.lower() for p in parts]
            return '_'.join(parts)

        case 'kebab':
            if not preserve_case:
                parts = [p.lower() for p in parts]
            return '-'.join(parts)

        case 'camel':
            parts = list(parts)
            parts = [parts[0].lower()] + [p.capitalize() for p in parts[1:]]
            return ''.join(parts)

        case 'pascal':
            return ''.join(p.capitalize() for p in parts)

        case _:
            assert_never(style)
