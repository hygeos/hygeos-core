from lxml import objectify
from pathlib import Path
import pandas as pd


def read_xml(path: str|Path) -> dict:
    """
    Function to read xml file

    Args:
        path (str | Path): Path of xml file
    """
    return _XML_parser().parse(path)

def read_csv(path: str|Path, **kwargs) -> pd.DataFrame:
    """
    Function to read csv file without taking care of tabulation and whitespaces

    Args:
        path (str | Path): Path of csv file
        kwargs: Keyword arguments of `read_csv` function from pandas

    Returns:
        DataFrame: Output table in pandas DataFrame format
    """
    # verify that the csv_file exists
    csv_file = Path(path)
    assert csv_file.is_file()
    
    # open csv file
    df = pd.read_csv(csv_file, skipinitialspace=True, **kwargs)
    
    # remove trailing whitespaces
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) 
    df.columns = [c.strip() for c in df.columns]
    return df

def select(table, where:tuple, cols:str|list = None):
    """
    Selection function in a pandas DataFrame with a condition 

    Args:
        dataframe (pd.DataFrame): Input table from which to select
        where (tuple): Condition to use for the selection
        cols (str | list): Name of the columns to return
    
    Example:
        select(df, ('col_1','=',20), ['col_2','col_3'])
    """ 
    condition = op_map[where[1]](table[where[0]], where[2])
    
    result = table[condition]
    if cols is not None: 
        result = result[cols]
    return result

def select_cell(table, where:tuple, col:str):
    """
    Function for selecting a single cell value in a pandas DataFrame with a condition

    Args:
        dataframe (pd.DataFrame): Input table from which to select
        where (tuple): Condition to use for the selection
        col (str | list): Name of the column to return
    
    Example:
        select_cell(df, ('col_1','=',20), 'col_2')
    """  
    df = select(table, where, col)
    assert len(df) == 1, f'Expected to return only one values (got {len(df)})'
    return df.values[0]


op_map = {"=": lambda a, b: a == b,
          "==": lambda a, b: a == b,
          ">": lambda a, b: a > b,
          "<": lambda a, b: a < b,
          ">=": lambda a, b: a >= b,
          "<=": lambda a, b: a <= b,
          "!=": lambda a, b: a != b,}

class _XML_parser:
    
    def __init__(self):
        pass
    
    def parse(self, xmlpath):
        root = objectify.parse(xmlpath).getroot()
        return self.recurse(root)
    
    def get_tag(self, node):
        if 'name' not in node.attrib: 
            tag = node.tag
            if node.prefix: 
                prefix = '{'+node.nsmap[node.prefix]+'}'
                tag = tag.replace(prefix,'')
            return tag.strip()
        else: return node.attrib['name']

    def recurse(self, node):
        """Recursively convert an lxml.objectify tree to a dictionary"""
        result = {}
        
        # Handle attributes
        if node.attrib:
            result['attributes'] = dict(node.attrib)
        
        # Handle child elements
        for child in node.getchildren():
            
            tag = self.get_tag(child)
            child_res = self.recurse(child)
            if ['values'] == list(child_res.keys()): 
                child_res = child_res['values']
            
            # If this tag already exists, we need to make it a list
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(child_res)
                else:
                    result[tag] = [result[tag], child_res]
            else:
                result[tag] = child_res
        
        # Handle text content if no children
        if not node.getchildren() and node.pyval:
            result['values'] = node.pyval
        
        return result