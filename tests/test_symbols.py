from xolo.symbols import convert_case_style


def test_convert_pascal_case_to_kebab_case():
    """
    Test if the convert_case_style function correctly converts from pascal-case to kebab-case.
    """
    symbol = 'XMLHttpRequest'
    result = convert_case_style(symbol, from_style='pascal', to_style='kebab')
    assert result == 'xml-http-request', 'The case style conversion is incorrect'
