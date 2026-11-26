import pandas as pd

from hygeos_core.ascii_table import ascii_table


def _lines(s):
    return [l for l in s.split("\n") if l]


def test_v_padding():
    df = pd.DataFrame({"a": [1, 22], "bb": ["x", "y"]})
    s = ascii_table(df, style=ascii_table.style(v_padding=1)).to_string()
    lines = _lines(s)
    assert len(lines) == 4 + 2 * 3
    assert len({len(l) for l in lines}) == 1  # all lines aligned


def test_empty_dataframe():
    ascii_table(pd.DataFrame()).to_string()
    s = ascii_table(pd.DataFrame({"a": []})).to_string()
    assert "A" in s


def test_integer_column_names():
    s = ascii_table(pd.DataFrame({0: [1], 1: [2]})).to_string()
    assert len({len(l) for l in _lines(s)}) == 1
