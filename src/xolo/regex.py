import re
from typing import Optional


def delimited_string(
        delimiters: str,
        close_delimiters: Optional[str] = None,
        *,
        escape_chars: Optional[str] = '\\',
        return_string: bool = False,
) -> str | re.Pattern[str]:
    """
    Create a regex pattern to capture strings between matching delimiters.

    Parameters:
        delimiters (str): Opening delimiters.
        close_delimiters (Optional[str]): Closing delimiters. If None, defaults to the same as delimiters.
        escape_chars (Optional[str]): Characters to treat as escape characters. If None, no escape character is used.
        return_string (bool): If True, returns the regex pattern as a string; else, a compiled regex.

    Returns:
        Union[str, re.Pattern[str]]: The regex pattern, either as a string or compiled pattern.

    Raises:
        ValueError: If delimiters are empty, lengths of delimiters and close_delimiters do not match,
            or if escape_chars length does not match delimiters and is not a single character.
    """
    # Ensure we have at least one delimiter
    if not delimiters:
        raise ValueError('At least one delimiter is required.')

    # Default to using the same delimiter for closing if none is provided
    close_delimiters = close_delimiters or delimiters

    # Check for matching lengths of delimiters
    if len(delimiters) != len(close_delimiters):
        raise ValueError('Open and close delimiters must have the same length.')

    # Duplicate a single escape character for all delimiters if necessary
    if escape_chars and len(escape_chars) == 1:
        escape_chars *= len(delimiters)

    # Check for matching lengths of delimiters and escape characters
    if escape_chars and len(escape_chars) != len(delimiters):
        raise ValueError('Escape characters length must match delimiters or be a single character.')

    # Construct patterns
    patterns = []
    for i in range(len(delimiters)):
        open_delim = re.escape(delimiters[i])
        close_delim = re.escape(close_delimiters[i])
        esc_char = re.escape(escape_chars[i]) if escape_chars else None

        if not esc_char:
            pattern = f'{open_delim}[^{close_delim}]*{close_delim}'
        elif esc_char == close_delim:
            pattern = f'{open_delim}[^{close_delim}]*(?:{close_delim}{close_delim}[^{close_delim}]*)*{close_delim}'
        else:
            pattern = f'{open_delim}[^{esc_char}{close_delim}]*(?:{esc_char}.[^{esc_char}{close_delim}]*)*{close_delim}'

        patterns.append(pattern)

    # Combine patterns using a non-capturing group
    combined_pattern = '(?:' + '|'.join(patterns) + ')'

    # Return the pattern either as a string or as a compiled regex
    return combined_pattern if return_string else re.compile(combined_pattern)



def integer(
        *,
        base: int = 10,
        sep: Optional[str] = None,
        group: Optional[int | tuple[int, int]] = 3,
        places: Optional[int | tuple[int, int]] = None,
        sign: Optional[str] = '[-+]?',
        return_string: bool = False,
) -> str | re.Pattern[str]:
    """
    Returns a regex pattern that matches integers in a specific numeral base
    with optional formatting.

    Args:
        base (int, optional): The numeral base for the integer, between 2 and 36. Default is 10.
        sep (str, optional): The character used to separate groups of digits. Default is None.
        group (int or tuple[int, int], optional): The number of digits in each group,
            either as a fixed number or a range. This is ignored unless `sep` is provided. Default is 3.
        places (int or tuple[int, int], optional): Specifies the total number of places/digits
            the integer should have, either as a fixed number or a range. This is ignored if `sep`
            is provided. Default is None.
        sign (str, optional): Regular expression pattern to match the sign of the integer.
            Default is '[-+]?', which matches optional minus or plus signs.
        return_string (bool, optional): If True, returns the pattern as a string. Otherwise,
            returns a compiled regex pattern. Default is False.

    Returns:
        str | regex.Pattern[str]: The regex pattern, either as a string or compiled pattern.

    Raises:
        ValueError: If the base is not between 2 and 36.
    """
    # Get valid characters for the base
    valid_chars = valid_digits_for_base(base)

    # Separator handling
    if sep:
        sep = re.escape(sep)
        group_quant = make_quantifier(group, default='+')
        prefix_quant = make_quantifier(group, default='+', min_val=1)
        core_pattern = f'{valid_chars}{prefix_quant}(?:{sep}{valid_chars}{group_quant})*'
    else:
        quant = make_quantifier(places, default='+')
        core_pattern = f'{valid_chars}{quant}'

    # Add sign pattern if required
    pattern = f'{sign}{core_pattern}' if sign else core_pattern

    # Case sensitivity
    if base > 10:
        pattern = f'(?i:{pattern})'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else re.compile(pattern)



def floating_point(
        *,
        base: int = 10,
        radix: str = r'\.',
        places: Optional[int | tuple[int, int]] = None,
        sep: Optional[str] = None,
        group: Optional[int | tuple[int, int]] = 3,
        expon: Optional[str] = '[Ee]',
        sign: Optional[str] = '[-+]?',
        return_string: bool = False,
) -> str | re.Pattern[str]:
    """
    Returns a regex pattern that matches floating-point numbers in a specific numeral base with optional formatting.

    Args:
        base (int, optional): The numeral base for the number, between 2 and 36. Default is 10.
        radix (str, optional): The character or pattern representing the radix point. Default is '\\.'.
        places (int or tuple[int, int], optional): Specifies the number of digits after the radix point.
            Default is None, allowing any number of digits.
        sep (str, optional): The character used to separate groups of digits before the radix point.
            Default is None, indicating no grouping.
        group (int or tuple[int, int], optional): The number of digits in each group before the radix point.
            Relevant only if `sep` is provided. Default is 3.
        expon (str or None, optional): The character or pattern for the exponential part.
            Default is '[Ee]', matching 'E' or 'e'.
        sign (str, optional): Pattern to match the sign of the number or its exponent.
            Default is '[-+]?'
        return_string (bool, optional): If True, returns the pattern as a string;
            otherwise, returns a compiled regex pattern.

    Returns:
        str | regex.Pattern[str]: The regex pattern, either as a string or compiled pattern.

    Raises:
        ValueError: If the base is not between 2 and 36.
    """
    # Get valid characters for the base
    valid_chars = valid_digits_for_base(base)

    # Integral part
    int_pattern = integer(base=base, sep=sep, group=group, sign=None, return_string=True)

    valid_chars = valid_digits_for_base(base)

    # Fractional part
    frac_pattern = f'{radix}{valid_chars}{make_quantifier(places, default="*")}'

    # Decimal pattern
    pattern = f'{sign}(?:{int_pattern}(?:{frac_pattern})?|{frac_pattern})'

    # Exponential part
    if expon is not None:
        pattern += f'(?:{expon}{sign}{valid_chars}+)?'

    # Return the pattern either as a string or as a compiled regex
    return pattern if return_string else re.compile(pattern)



def make_quantifier(
        x: Optional[int | tuple[int, int]],
        default: str,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
) -> str:
    """
    Generate a regex quantifier string based on provided values or a range.

    Args:
        x (Optional[int | tuple[int, int]]): An integer or a tuple specifying a range for the quantifier.
            If None, the default quantifier is used.
        default (str): The default quantifier to use if x is None.
        min_val (Optional[int]): The minimum value for the quantifier if x is an integer.
            Defaults to the value of x or x[0] if x is a tuple.
        max_val (Optional[int]): The maximum value for the quantifier if x is an integer.
            Defaults to the value of x or x[1] if x is a tuple.

    Returns:
        str: A regex quantifier string.

    Raises:
        ValueError: If the range is invalid (min_val > max_val) or if x is not an int or a tuple of two ints.
    """
    if x is None:
        return default

    # Validate the type of x
    if not (isinstance(x, int) or (isinstance(x, tuple) and len(x) == 2 and all(isinstance(val, int) for val in x))):
        raise ValueError("x must be an integer or a tuple of two integers")

    # Determine min and max values
    if min_val is None:
        min_val = x if isinstance(x, int) else x[0]
    if max_val is None:
        max_val = x if isinstance(x, int) else x[1]

    # Validate the range
    if min_val > max_val:
        raise ValueError(f'Invalid range: min_val={min_val} max_val={max_val}')

    # Construct the quantifier string
    return f'{{{min_val}}}' if min_val == max_val else f'{{{min_val},{max_val}}}'



def valid_digits_for_base(base: int) -> str:
    """
    Generate a regex pattern for valid digits in the specified base.

    This function creates a regex pattern that matches valid digits for a given base, up to base 36.
    For example, for base 16 (hexadecimal), it returns '[0-9a-f]'.

    Args:
        base (int): The base for which to generate the regex pattern. Must be between 2 and 36.

    Returns:
        str: A regex pattern string for valid digits in the specified base.

    Raises:
        ValueError: If the base is not between 2 and 36.
    """
    if not (2 <= base <= 36):
        raise ValueError('Base should be between 2 and 36')

    if base == 2:
        return '[01]'

    # Generate pattern for numeric digits
    numeric_range = f'0-{min(base, 10)-1}'

    # Generate pattern for alphabetic digits if base is greater than 10
    alphabetic_range = ''
    if base > 10:
        alphabetic_range = f'a-{chr(ord("a") + base - 11)}'

    # Combine numeric and alphabetic ranges
    pattern = f'[{numeric_range}{alphabetic_range}]'

    return pattern
