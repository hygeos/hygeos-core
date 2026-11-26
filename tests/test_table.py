from hygeos_core.table import *
from hygeos_core.static.Exceptions import InterfaceException
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def dataset():
    # Generation of test dataset
    shape = (20,5)
    array = np.arange(shape[0]*shape[1],dtype=int).reshape(shape)
    cols  = [f'col_{i+1}' for i in range(shape[1])] 
    return pd.DataFrame(array, columns=cols)


def test_read_xml():
    xml_file = Path(__file__).parent/'inputs'/'test.xml'
    d = read_xml(xml_file)
    print(d)    

def test_read_csv():
    csv_file = Path(__file__).parent/'inputs'/'test.csv'
    assert csv_file.is_file()
    df = read_csv(csv_file)
    print(df)

@pytest.mark.parametrize("cols", [
    "col_1",
    ['col_1','col_5']
])
@pytest.mark.parametrize("conditions", [("col_2", '=', 6)])
def test_select(dataset, conditions, cols):
    out = select(dataset, cols=cols, where=conditions)
    assert isinstance(out, pd.Series|pd.DataFrame)

@pytest.mark.parametrize("conditions", [("col_2", '=', 6)])
def test_select_one(dataset, conditions):
    out = select_cell(dataset, col='col_1', where=conditions)
    assert isinstance(out, np.int64)

def test_read_csv_strips_trailing_whitespace(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_text('name , value\n foo  , 1\n')
    df = read_csv(p)
    assert list(df.columns) == ['name', 'value']
    assert df['name'].iloc[0] == 'foo'


def test_read_csv_header_none(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_text(' foo  , 1\n')
    df = read_csv(p, header=None)
    assert list(df.columns) == [0, 1]
    assert df[0].iloc[0] == 'foo'


def test_read_xml_falsy_values(tmp_path):
    p = tmp_path / 'a.xml'
    p.write_text('<root><a>0</a><b>1</b><c>false</c><d></d></root>')
    d = read_xml(p)
    assert d['a'] == 0
    assert d['b'] == 1
    assert d['c'] is False
    assert d['d'] == {}


def test_read_xml_default_namespace(tmp_path):
    p = tmp_path / 'a.xml'
    p.write_text('<root xmlns="http://ex.com/ns" xmlns:p="http://ex.com/p">'
                 '<a>1</a><p:b>2</p:b></root>')
    d = read_xml(p)
    assert d == {'a': 1, 'b': 2}
