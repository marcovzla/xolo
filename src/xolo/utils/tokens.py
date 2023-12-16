from typing import Optional, assert_never
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import regex
from xolo.utils.spans import Span



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



VALID_TOKEN_TAGS = set('OBILUES')
"""All the valid tags used in the supported token annotation schemas."""



class TaggingSchema(StrEnum):
    IO = 'io'
    IOB1 = 'iob1'
    IOB2 = 'iob2'
    BILOU = 'bilou'
    IOBES = 'iobes'



@dataclass(frozen=True, order=True)
class TokenSpan(Span):
    label: Optional[str] = None



def parse_token_label(label: str, sep: str = '-') -> tuple[Optional[str], Optional[str]]:
    """
    Parse a token label into its constituent parts (tag and entity) based on a given separator.

    This function handles various token-based labeling formats used in NER and similar tasks. It splits a given token
    label into two parts: the tag (like 'O', 'B', 'I', etc.) and the entity type (like 'Person', 'Location', etc.),
    based on the specified separator. If the label does not follow the expected format, it defaults to treating it as
    an 'Inside' (I) tag with the entire label as the entity.

    Args:
        label (str): The token label to be parsed.
        sep (str, optional): The separator used to split the label into tag and entity. Defaults to '-'.

    Returns:
        tuple[Optional[str], Optional[str]]: A tuple where the first element is the tag and the second element 
        is the entity type. If the label does not contain a separator, the function defaults to ('I', label).

    Notes:
        - If the label is empty, both the tag and entity are returned as None.
        - If the label contains the separator, it's split into tag and entity.
        - If the label does not contain the separator but is a valid token tag, it's assumed to be a tag with no entity.
        - If the label is not a valid tag and does not contain the separator, it's treated as an entity with an 'I' tag.
    """
    if not label:
        return None, None
    elif sep in label:
        return tuple(label.split(sep=sep, maxsplit=1))
    elif label in VALID_TOKEN_TAGS:
        return label, None
    else:
        return 'I', label




def valid_label(label: str, schema: TaggingSchema) -> bool:
    """
    Determine if a given label is valid according to a specified tagging schema.

    This function evaluates whether a label conforms to the rules of the tagging schema provided. Different tagging schemas
    like IO, IOB1, IOB2, BILOU, IOBES, and UNTAGGED have specific formats and rules for labels. This function delegates the
    validation to the corresponding schema-specific function based on the provided schema.

    Args:
        label (str): The label to be validated.
        schema (TaggingSchema): The tagging schema to be used for validating the label.

    Returns:
        bool: True if the label is valid according to the specified tagging schema; otherwise, False.

    Raises:
        AssertionError: If an unsupported tagging schema is provided.

    Notes:
        - The function uses a `match` statement to delegate to the appropriate schema-specific validation function.
          An `assert_never` is used to ensure all possible schema cases are covered.
        - The UNTAGGED schema is included to handle cases where labels are untagged or do not follow a structured format.
    """
    match schema:
        case TaggingSchema.IO:
            return io_valid_label(label)
        case TaggingSchema.IOB1:
            return iob1_valid_label(label)
        case TaggingSchema.IOB2:
            return iob2_valid_label(label)
        case TaggingSchema.BILOU:
            return bilou_valid_label(label)
        case TaggingSchema.IOBES:
            return iobes_valid_label(label)
        case _:
            assert_never(schema)



def valid_transition(from_label: Optional[str], to_label: Optional[str], schema: TaggingSchema) -> bool:
    """
    Determine if a transition between two labels is valid according to a specified tagging schema.

    This function evaluates the validity of transitioning from `from_label` to `to_label` based on the tagging schema
    provided. Different tagging schemas like IO, IOB1, IOB2, BILOU, IOBES, and UNTAGGED have specific rules for label transitions.
    This function delegates the validation to the corresponding schema-specific function based on the provided schema.

    Args:
        from_label (Optional[str]): The label from which the transition starts. If `None`, it implies the beginning of a sequence.
        to_label (Optional[str]): The label to which the transition goes. If `None`, it implies the end of a sequence.
        schema (TaggingSchema): The tagging schema to be used for validating the transition.

    Returns:
        bool: True if the transition is valid according to the specified tagging schema; otherwise, False.

    Raises:
        AssertionError: If an unsupported tagging schema is provided.

    Notes:
        - The function uses a `match` statement to delegate to the appropriate schema-specific validation function.
          An `assert_never` is used to ensure all possible schema cases are covered.
    """
    match schema:
        case TaggingSchema.IO:
            return io_valid_transition(from_label, to_label)
        case TaggingSchema.IOB1:
            return iob1_valid_transition(from_label, to_label)
        case TaggingSchema.IOB2:
            return iob2_valid_transition(from_label, to_label)
        case TaggingSchema.BILOU:
            return bilou_valid_transition(from_label, to_label)
        case TaggingSchema.IOBES:
            return iobes_valid_transition(from_label, to_label)
        case _:
            assert_never(schema)



def load_spans(labels: Sequence[str], scheme: TaggingSchema) -> list[TokenSpan]:
    """
    Convert a sequence of labels into a list of TokenSpan objects according to a specified tagging schema.

    This function delegates the conversion of labels to TokenSpans based on the provided tagging schema, such as IO, IOB1, IOB2, 
    BILOU, or IOBES. Each tagging schema has specific rules and formats for representing entity spans in labeled sequences, 
    commonly used in tasks like Named Entity Recognition (NER).

    Args:
        labels (Sequence[str]): A sequence of labels to be converted into TokenSpans.
        scheme (TaggingSchema): The tagging schema to use for the conversion.

    Returns:
        list[TokenSpan]: A list of TokenSpan objects, each representing a span of tokens with a specific entity type,
                         as defined by the specified tagging schema.

    Raises:
        AssertionError: If an unsupported tagging schema is provided.

    Notes:
        - TokenSpan is assumed to be a named tuple or similar object with fields `start`, `end`, and `entity`.
        - This function is capable of handling different tagging schemas, making it versatile for various NER and
          related tasks.
        - The `match` statement efficiently directs to the corresponding function for each tagging schema.
    """
    match scheme:
        case TaggingSchema.IO:
            return io_to_spans(labels)
        case TaggingSchema.IOB1:
            return iob1_to_spans(labels)
        case TaggingSchema.IOB2:
            return iob2_to_spans(labels)
        case TaggingSchema.BILOU:
            return bilou_to_spans(labels)
        case TaggingSchema.IOBES:
            return iobes_to_spans(labels)
        case _:
            assert_never(scheme)



def io_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    """
    Determine if a transition between two labels is valid within the IO token annotation schema.

    This method specifically applies to the IO (Inside-Outside) token annotation schema. It assesses
    the validity of a transition from `from_label` to `to_label` in sequences annotated according to the IO schema.

    Args:
        from_label (Optional[str]): The label from which the transition starts. If `None`, it is 
            interpreted as the start of the sequence, meaning that `to_label` is the first label in the sequence.
            This is specific to the IO schema, where sequences start without a prior context.
        to_label (Optional[str]): The label to which the transition goes. If `None`, it is interpreted 
            as the end of the sequence, indicating that `from_label` is the last token in the sequence.
            In the IO schema, this signifies the end of the current annotation.

    Returns:
        bool: True if the transition is valid within the IO schema, i.e., at least one of the labels 
        (`from_label` or `to_label`) is not `None`. False is returned if both are `None`, as it implies a 
        transition from nowhere to nowhere, which is invalid in the IO context.

    Examples:
        - Calling `io_valid_transition("I", "O")` returns `True`, as it's a valid IO transition from "I" to "O".
        - Calling `io_valid_transition(None, "I")` returns `True`, indicating the start of a sequence with "I".
        - Calling `io_valid_transition("O", None)` returns `True`, indicating the end of a sequence after "O".
        - Calling `io_valid_transition(None, None)` returns `False`, as it's an invalid transition in IO schema.
    """
    return from_label is not None or to_label is not None



def iob1_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    """
    Determine if a transition between two labels is valid within the IOB1 (Inside-Outside-Beginning) token annotation schema.

    This function assesses the validity of transitioning from `from_label` to `to_label` in sequences annotated according to the IOB1 schema,
    which is commonly used in tasks like Named Entity Recognition (NER). It enforces the rules of the IOB1 format, ensuring that transitions
    between labels follow the correct sequence pattern.

    Args:
        from_label (Optional[str]): The label from which the transition starts. If `None`, it represents the start of a sequence.
        to_label (Optional[str]): The label to which the transition goes. If `None`, it represents the end of a sequence.

    Returns:
        bool: True if the transition is valid within the IOB1 schema; otherwise, False.

    Rules:
        - Transition from `None` to a valid 'I' or 'O' tag is valid (start of a sequence).
        - Transition to `None` from any tag is valid (end of a sequence).
        - 'O' can transition to 'I' or 'O'.
        - 'B' can transition to 'I' with the same entity or to any 'B' or 'O'.
        - 'I' can transition to 'I' with the same entity or to any 'B' or 'O'.
    """
    if from_label is None:
        return to_label is not None and to_label[0] in 'IO'
    if to_label is None:
        return True

    from_tag, from_entity = parse_token_label(from_label)
    to_tag, to_entity = parse_token_label(to_label)

    if from_tag == 'O':
        return to_tag in 'IO'
    if from_tag in 'BI':
        if to_tag == 'I':
            return from_entity == to_entity
        if to_tag in 'BO':
            return True
    return False



def iob2_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    """
    Determine if a transition between two labels is valid within the IOB2 (Inside-Outside-Beginning 2) token annotation schema.

    This function assesses the validity of transitioning from `from_label` to `to_label` in sequences annotated according to the IOB2 schema,
    which is widely used in Named Entity Recognition (NER) and similar tasks. It enforces the rules of the IOB2 format, ensuring that transitions
    between labels adhere to the correct sequence pattern.

    Args:
        from_label (Optional[str]): The label from which the transition starts. If `None`, it represents the start of a sequence.
        to_label (Optional[str]): The label to which the transition goes. If `None`, it represents the end of a sequence.

    Returns:
        bool: True if the transition is valid within the IOB2 schema; otherwise, False.

    Rules:
        - Transition from `None` to a valid 'B' or 'O' tag is valid (start of a sequence).
        - Transition to `None` from any tag is valid (end of a sequence).
        - 'O' can transition to 'B' or 'O'.
        - 'B' can transition to 'I' with the same entity or to any 'B' or 'O'.
        - 'I' can transition to 'I' with the same entity or to any 'B' or 'O'.
    """
    if from_label is None:
        return to_label is not None and to_label[0] in 'BO'
    if to_label is None:
        return True

    from_tag, from_entity = parse_token_label(from_label)
    to_tag, to_entity = parse_token_label(to_label)

    if from_tag == 'O':
        return to_tag in 'BO'
    if from_tag in 'BI':
        if to_tag == 'I':
            return from_entity == to_entity
        if to_tag in 'BO':
            return True
    return False



def bilou_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    """
    Determine if a transition between two labels is valid within the BILOU (Beginning-Inside-Last-Outside-Unit) token annotation schema.

    This function evaluates the validity of transitioning from `from_label` to `to_label` in sequences annotated according to the BILOU schema.
    The BILOU schema is a detailed labeling system used in NER and similar tasks, which provides more specific information about the position of a token within an entity.

    Args:
        from_label (Optional[str]): The label from which the transition starts. If `None`, it represents the start of a sequence.
        to_label (Optional[str]): The label to which the transition goes. If `None`, it represents the end of a sequence.

    Returns:
        bool: True if the transition is valid within the BILOU schema; otherwise, False.

    Rules:
        - Transition from `None` to a valid 'B', 'U', or 'O' tag is valid (start of a sequence).
        - Transition to `None` from 'L', 'U', or 'O' tag is valid (end of a sequence).
        - 'O' can transition to 'B', 'U', or 'O'.
        - 'B' can transition to 'I' or 'L', but only with the same entity type.
        - 'I' can transition to 'I' or 'L', but only with the same entity type.
        - 'L' can transition to 'B', 'U', or 'O'.
        - 'U' can transition to 'B', 'U', or 'O'.
    """
    if from_label is None:
        return to_label is not None and to_label[0] in 'BUO'
    if to_label is None:
        return from_tag in 'LUO'

    from_tag, from_entity = parse_token_label(from_label)
    to_tag, to_entity = parse_token_label(to_label)

    if from_tag == 'O':
        return to_tag in 'BUO'
    if from_tag == 'B':
        return to_tag in 'IL' and from_entity == to_entity
    if from_tag == 'I':
        return to_tag in 'IL' and from_entity == to_entity
    if from_tag == 'L':
        return to_tag in 'BUO'
    if from_tag == 'U':
        return to_tag in 'BUO'
    return False



def iobes_valid_transition(from_label: Optional[str], to_label: Optional[str]) -> bool:
    """
    Determine if a transition between two labels is valid within the IOBES (Inside-Outside-Beginning-End-Single) token annotation schema.

    This function evaluates the validity of transitioning from `from_label` to `to_label` in sequences annotated according to the IOBES schema.
    The IOBES schema is a comprehensive labeling system used in NER and similar tasks, offering specific information about the position of a token
    within an entity as well as the boundaries of the entity.

    Args:
        from_label (Optional[str]): The label from which the transition starts. If `None`, it represents the start of a sequence.
        to_label (Optional[str]): The label to which the transition goes. If `None`, it represents the end of a sequence.

    Returns:
        bool: True if the transition is valid within the IOBES schema; otherwise, False.

    Rules:
        - Transition from `None` to a valid 'O', 'B', or 'S' tag is valid (start of a sequence).
        - Transition to `None` from 'E', 'U', or 'O' tag is valid (end of a sequence).
        - 'O' can transition to 'O', 'B', or 'S'.
        - 'B' can transition to 'I' or 'E', but only with the same entity type.
        - 'I' can transition to 'I' or 'E', but only with the same entity type.
        - 'E' can transition to 'O', 'B', or 'S'.
        - 'S' can transition to 'O', 'B', or 'S'.
    """
    if from_label is None:
        return to_label is not None and to_label[0] in 'BSO'
    if to_label is None:
        return from_tag in 'EUO'

    from_tag, from_entity = parse_token_label(from_label)
    to_tag, to_entity = parse_token_label(to_label)

    if from_tag == 'O':
        return to_tag in 'BSO'
    if from_tag == 'B':
        return to_tag in 'IE' and from_entity == to_entity
    if from_tag == 'I':
        return to_tag in 'IE' and from_entity == to_entity
    if from_tag == 'E':
        return to_tag in 'BSO'
    if from_tag == 'S':
        return to_tag in 'BSO'
    return False



def io_valid_label(label: str) -> bool:
    """
    Check if a given label is valid within the IO (Inside-Outside) token annotation schema.

    This function evaluates whether a given label conforms to the rules of the IO schema, which is a simple
    binary labeling system used in NER and similar tasks.

    Args:
        label (str): The label to be validated.

    Returns:
        bool: True if the label is valid within the IO schema; otherwise, False.

    Notes:
        - Valid IO tags are 'I' (Inside) and 'O' (Outside).
    """
    tag, _ = parse_token_label(label)
    return tag in 'IO'



def iob1_valid_label(label: str) -> bool:
    """
    Check if a given label is valid within the IOB1 (Inside-Outside-Beginning) token annotation schema.

    This function evaluates whether a given label conforms to the rules of the IOB1 schema, commonly used
    in NER and similar tasks for more detailed entity recognition.

    Args:
        label (str): The label to be validated.

    Returns:
        bool: True if the label is valid within the IOB1 schema; otherwise, False.

    Notes:
        - Valid IOB1 tags are 'I' (Inside), 'O' (Outside), and 'B' (Beginning).
    """
    tag, _ = parse_token_label(label)
    return tag in 'IOB'



def iob2_valid_label(label: str) -> bool:
    """
    Check if a given label is valid within the IOB2 (Inside-Outside-Beginning 2) token annotation schema.

    This function evaluates whether a given label conforms to the rules of the IOB2 schema, an enhancement
    of the IOB1 schema for NER and related tasks, offering clearer boundaries of entities.

    Args:
        label (str): The label to be validated.

    Returns:
        bool: True if the label is valid within the IOB2 schema; otherwise, False.

    Notes:
        - Valid IOB2 tags are 'I' (Inside), 'O' (Outside), and 'B' (Beginning).
    """
    tag, _ = parse_token_label(label)
    return tag in 'IOB'



def bilou_valid_label(label: str) -> bool:
    """
    Check if a given label is valid within the BILOU (Beginning-Inside-Last-Outside-Unit) token annotation schema.

    This function evaluates whether a given label conforms to the rules of the BILOU schema, a detailed
    labeling system used in NER and similar tasks for precise entity boundary and position identification.

    Args:
        label (str): The label to be validated.

    Returns:
        bool: True if the label is valid within the BILOU schema; otherwise, False.

    Notes:
        - Valid BILOU tags are 'B' (Beginning), 'I' (Inside), 'L' (Last), 'O' (Outside), and 'U' (Unit).
    """
    tag, _ = parse_token_label(label)
    return tag in 'BILOU'



def iobes_valid_label(label: str) -> bool:
    """
    Check if a given label is valid within the IOBES (Inside-Outside-Beginning-End-Single) token annotation schema.

    This function evaluates whether a given label conforms to the rules of the IOBES schema, an advanced
    labeling system used in NER and similar tasks for detailed entity boundary and position identification.

    Args:
        label (str): The label to be validated.

    Returns:
        bool: True if the label is valid within the IOBES schema; otherwise, False.

    Notes:
        - Valid IOBES tags are 'I' (Inside), 'O' (Outside), 'B' (Beginning), 'E' (End), and 'S' (Single).
    """
    tag, _ = parse_token_label(label)
    return tag in 'IOBES'



def io_to_spans(labels: Sequence[str]) -> list[TokenSpan]:
    """
    Convert a list of IO (Inside-Outside) formatted labels into a list of TokenSpan objects.

    This function processes a sequence of labels formatted according to the IO schema and groups contiguous 
    tokens with the same entity type into TokenSpans. Each TokenSpan represents a span of tokens associated 
    with a particular entity type.

    Args:
        labels (Sequence[str]): A sequence of labels formatted according to the IO schema.

    Returns:
        list[TokenSpan]: A list of TokenSpan objects, each representing a span of tokens with a specific entity type.

    Raises:
        ValueError: If any label in the list is invalid according to the IO schema or if there is an invalid transition 
                    between two consecutive labels.

    Notes:
        - A TokenSpan is defined here as a named tuple or similar object with fields `start`, `end`, and `entity`.
          `start` and `end` denote the start and end indices of the span in the list of labels, and `entity` is the 
          entity type associated with the span.
        - This function is specific to the IO tagging schema and verifies both the validity of individual labels 
          and the transitions between them.
    """
    spans = []
    start = span_entity = previous_label = None
    for i, label in enumerate(labels):
        if not io_valid_label(label):
            raise ValueError(f'Invalid label: {label}')
        if not io_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'Invalid transition from {prev!r} to {label!r}')

        tag, entity = parse_token_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'I':
            if entity != span_entity:
                if start is not None:
                    spans.append(TokenSpan(start, i, span_entity))
                start = i
                span_entity = entity

        previous_label = label

    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))

    return spans



def iob1_to_spans(labels: Sequence[str]) -> list[TokenSpan]:
    """
    Convert a list of IOB1 (Inside-Outside-Beginning) formatted labels into a list of TokenSpan objects.

    This function processes a sequence of labels formatted according to the IOB1 schema and groups contiguous 
    tokens with the same entity type into TokenSpans. Each TokenSpan represents a span of tokens associated 
    with a particular entity type.

    Args:
        labels (Sequence[str]): A sequence of labels formatted according to the IOB1 schema.

    Returns:
        list[TokenSpan]: A list of TokenSpan objects, each representing a span of tokens with a specific entity type.

    Raises:
        ValueError: If any label in the list is invalid according to the IOB1 schema or if there is an invalid transition 
                    between two consecutive labels.

    Notes:
        - A TokenSpan is defined here as a named tuple or similar object with fields `start`, `end`, and `entity`.
          `start` and `end` denote the start and end indices of the span in the list of labels, and `entity` is the 
          entity type associated with the span.
        - This function is specific to the IOB1 tagging schema and verifies both the validity of individual labels 
          and the transitions between them.
    """
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not iob1_valid_label(label):
            raise ValueError(f'Invalid label: {label}')
        if not iob1_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'Invalid transition from {prev!r} to {label!r}')

        tag, entity = parse_token_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'B':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
            start = i
            span_entity = entity
        elif tag == 'I':
            if start is None:
                start = i
                span_entity = entity

        previous_label = label

    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))

    return spans



def iob2_to_spans(labels: Sequence[str]) -> list[TokenSpan]:
    """
    Convert a list of IOB2 (Inside-Outside-Beginning 2) formatted labels into a list of TokenSpan objects.

    This function processes a sequence of labels formatted according to the IOB2 schema, a commonly used format in 
    Named Entity Recognition (NER) tasks. It groups contiguous tokens with the same entity type into TokenSpans, 
    where each TokenSpan represents a span of tokens associated with a particular entity type.

    Args:
        labels (Sequence[str]): A sequence of labels formatted according to the IOB2 schema.

    Returns:
        list[TokenSpan]: A list of TokenSpan objects, each representing a span of tokens with a specific entity type.

    Raises:
        ValueError: If any label in the list is invalid according to the IOB2 schema or if there is an invalid transition 
                    between two consecutive labels.

    Notes:
        - A TokenSpan is defined here as a named tuple or similar object with fields `start`, `end`, and `entity`.
          `start` and `end` denote the start and end indices of the span in the list of labels, and `entity` is the 
          entity type associated with the span.
        - The IOB2 schema enhances the IOB1 format by enforcing that every new entity span must start with a 'B' tag, 
          making it easier to delineate entities.
    """
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not iob2_valid_label(label):
            raise ValueError(f'Invalid label: {label}')
        if not iob2_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'Invalid transition from {prev!r} to {label!r}')

        tag, entity = parse_token_label(label)
        if tag == 'O':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
                start = span_entity = None
        elif tag == 'B':
            if start is not None:
                spans.append(TokenSpan(start, i, span_entity))
            start = i
            span_entity = entity

        previous_label = label

    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))

    return spans



def bilou_to_spans(labels: Sequence[str]) -> list[TokenSpan]:
    """
    Convert a list of BILOU (Beginning-Inside-Last-Outside-Unit) formatted labels into a list of TokenSpan objects.

    This function processes a sequence of labels formatted according to the BILOU schema, which is a detailed
    tagging format used in Named Entity Recognition (NER) tasks. It groups tokens into TokenSpans, each representing
    a contiguous sequence of tokens associated with a particular entity type. The BILOU format explicitly marks the
    beginning, inside, last, and single-token (unit) entities, providing clear boundaries for each entity span.

    Args:
        labels (Sequence[str]): A sequence of labels formatted according to the BILOU schema.

    Returns:
        list[TokenSpan]: A list of TokenSpan objects, each representing a span of tokens with a specific entity type.

    Raises:
        ValueError: If any label in the list is invalid according to the BILOU schema or if there is an invalid
                    transition between two consecutive labels.

    Notes:
        - A TokenSpan is defined as a named tuple or similar object with fields `start`, `end`, and `entity`.
          `start` and `end` denote the start and end indices of the span in the list of labels, and `entity` is the
          entity type associated with the span.
        - The function also validates the transitions between labels, including the transition from the last label
          to the end of the sequence.
    """
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not bilou_valid_label(label):
            raise ValueError(f'Invalid label: {label}')
        if not bilou_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'Invalid transition from {prev!r} to {label!r}')

        tag, entity = parse_token_label(label)
        if tag == 'B':
            start = i
            span_entity = entity
        elif tag == 'L':
            spans.append(TokenSpan(start, i + 1, span_entity))
            start = span_entity = None
        elif tag == 'U':
            spans.append(TokenSpan(i, i + 1, entity))

        previous_label = label

    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))

    if not bilou_valid_transition(labels[-1], None):
        raise ValueError(f'Invalid transition from {labels[-1]!r} to END')

    return spans



def iobes_to_spans(labels: Sequence[str]) -> list[TokenSpan]:
    """
    Convert a list of IOBES (Inside-Outside-Beginning-End-Single) formatted labels into a list of TokenSpan objects.

    This function processes a sequence of labels formatted according to the IOBES schema, which is used in Named Entity 
    Recognition (NER) tasks. The IOBES format is more granular than IOB, identifying the beginning, inside, end, and 
    single-token entities. This function groups tokens into TokenSpans, where each TokenSpan represents a contiguous 
    sequence of tokens associated with a particular entity type.

    Args:
        labels (Sequence[str]): A sequence of labels formatted according to the IOBES schema.

    Returns:
        list[TokenSpan]: A list of TokenSpan objects, each representing a span of tokens with a specific entity type.

    Raises:
        ValueError: If any label in the list is invalid according to the IOBES schema or if there is an invalid 
                    transition between two consecutive labels.

    Notes:
        - A TokenSpan is defined as a named tuple or similar object with fields `start`, `end`, and `entity`.
          `start` and `end` denote the start and end indices of the span in the list of labels, and `entity` is the
          entity type associated with the span.
        - The function also validates the transitions between labels, including the transition from the last label
          to the end of the sequence.
    """
    spans = []
    start = span_entity = None
    previous_label = None
    for i, label in enumerate(labels):
        if not iobes_valid_label(label):
            raise ValueError(f'Invalid label: {label}')
        if not iobes_valid_transition(previous_label, label):
            prev = 'START' if previous_label is None else previous_label
            raise ValueError(f'Invalid transition from {prev!r} to {label!r}')

        tag, entity = parse_token_label(label)
        if tag == 'B':
            start = i
            span_entity = entity
        elif tag == 'E':
            spans.append(TokenSpan(start, i + 1, span_entity))
            start = span_entity = None
        elif tag == 'S':
            spans.append(TokenSpan(i, i + 1, entity))

        previous_label = label

    if start is not None:
        spans.append(TokenSpan(start, len(labels), span_entity))

    if not iobes_valid_transition(labels[-1], None):
        raise ValueError(f'Invalid transition from {labels[-1]!r} to END')

    return spans
