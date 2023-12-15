from collections.abc import Sequence
import regex


def merge_tokens(tokens: Sequence[str]) -> str:
    """
    Merge a sequence of tokens into a single string, correcting spacing around punctuation and quotes.

    This function handles special cases such as unpaired quotation marks, contractions, and various
    punctuation marks, ensuring that the resultant string is formatted correctly.

    Args:
        tokens (Sequence[str]): A sequence of string tokens to merge.

    Returns:
        str: The merged string with corrected punctuation and spacing.
    """
    tokens = fix_quotes(tokens)
    text = ' '.join(tokens)
    text = correct_spacing(text)
    text = replace_ptb_style_quotes(text)
    return text


def fix_quotes(tokens: Sequence[str]) -> list[str]:
    """
    Correct the pairing of quotation marks in a sequence of tokens.

    This function ensures that each pair of quotation marks is correctly opened and closed.

    Args:
        tokens (Sequence[str]): A sequence of string tokens.

    Returns:
        list[str]: The sequence with corrected quotation marks.
    """
    corrected_tokens = list(tokens)
    counter = 0

    for i, token in enumerate(corrected_tokens):
        if token in ("``", "''"):
            corrected_tokens[i] = "``" if counter % 2 == 0 else "''"
            counter += 1

    return corrected_tokens


def correct_spacing(text: str) -> str:
    """
    Correct the spacing around various punctuation marks in a string.

    Args:
        text (str): The string to correct.

    Returns:
        str: The string with corrected spacing.
    """
    # Correct space for PTB-style quotes
    text = regex.sub(r'((?:^|\s)``)\s', r'\1', text)
    text = regex.sub(r"\s(''(?:\s|$))", r'\1', text)

    # Remove right space from open brackets and initial quotation characters
    text = regex.sub(r'(\p{Ps}|\p{Pi})\s', r'\1', text)

    # Remove left space from close brackets and final quotation characters
    text = regex.sub(r'\s(\p{Pe}|\p{Pf})', r'\1', text)

    # Remove left space from contractions
    text = regex.sub(r"\s('s|'m|'d|'re|'ve|'ll|n't)", r'\1', text, flags=regex.IGNORECASE)

    # Remove surrounding spaces from connectors
    text = regex.sub(r'\s?(\p{Pc}|\p{Pd})\s?', r'\1', text)

    # Correct space for standard punctuation
    text = regex.sub(r"\s([.,:;!?'])", r'\1', text)

    # Correct space for currency and other symbols
    text = regex.sub(r'([#$])\s', r'\1', text)

    return text


def replace_ptb_style_quotes(text: str) -> str:
    """
    Replace Penn Treebank-style quotes in a string with standard quotation marks.

    Args:
        text (str): The string with PTB-style quotes.

    Returns:
        str: The string with standard quotation marks.
    """
    return regex.sub(r"``|''", '"', text)
