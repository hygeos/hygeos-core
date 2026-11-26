from hygeos_core.table import read_csv, select
from pathlib import Path
from hygeos_core import log
import re


def retrieve_product(product_id: str, fields: dict, pattern: tuple[str] = None) -> str:
    """
    Function retrieve a product base on another one.
    For example, retrieve Level-2 product based on Level-1

    Args:
        product_id (str): Product considered as base
        fields (dict): Fields to change
        pattern (tuple[str], optional): Pattern of <product_id>. Defaults to None.

    Returns:
        str: New product id
    """
    
    # Check fields input
    if pattern is None: pattern = get_pattern(product_id)
    valid_fields = get_fields(pattern['pattern'])
    assert all(k in valid_fields for k in fields.keys()), \
    f'Invalid fields: keys of fields should be in {valid_fields}, got {fields.keys()}'
    
    # Transform giving fields
    retrieve = []
    rules = {valid_fields[i]: r for i,r in enumerate(pattern['regexp'].split(' '))}
    decompo = decompose(product_id, pattern['regexp'].split(' '))
    for i, field in enumerate(valid_fields):
        if field not in fields: retrieve.append(decompo[i])
        elif check(fields[field], rules[field]):
            retrieve.append(fields[field]) 
        else:
            log.error(f'Value for field {field} ({fields[field]}) '
                      f'does not satisfy regexp {rules[field]}', 
                      e=ValueError)
    
    # Join pieces to create new product id
    new = '_'.join(retrieve)
    ext = product_id.split('.')[1:]
    if len(ext) != 0: new = '.'.join([new]+ext)
    return new

def get_pattern(name: str) -> dict:
    """
    Get pattern that match product name

    Args:
        name (str): Product name or id

    Returns:
        dict: Dictionary containing sensor name, pattern and regular expression
    """
    
    # Find patterns
    db = read_csv(Path(__file__).parent/'patterns.csv')
    sensors = _find_pattern(name, db)
    
    # Check patterns contentpattern
    if len(sensors) == 0: log.error(f'No pattern matches product ID : {name}', e=Exception)
    assert len(sensors) == 1, f'Multiple matches, got {sensors}'
    
    return select(db, ('Name','=',sensors[0])).iloc[0].to_dict() 

def _find_pattern(name: str, database) -> list:
    """
    Hidden function that returns the list of sensor that match product id

    Args:
        name (str): Product name and id
        database (pd.DataFrame): Table with all patterns

    Returns:
        list: List of name of sensor
    """
    # Product names use both '_' and '.' as separators between tokens.
    # A trailing dot-separated chunk may be a file extension (e.g. '.SEN3',
    # '.N1'), so try the full name first, then without its last chunk.
    candidates = [name]
    if '.' in name: candidates.append(name.rsplit('.', 1)[0])
    for candidate in candidates:
        out = []
        for _, row in database.iterrows():
            regexp = row['regexp'].strip().split(' ')
            regexp = '_'.join(regexp)
            if check(candidate.replace('.', '_'), regexp): out.append(row['Name'])
        if len(out) != 0: return out
    return []

def check(name: str, regexp: str) -> bool:
    """
    Check if name satisfies regular expression
    """
    return bool(re.fullmatch(regexp, name))

def decompose(name: str, regexps: list[str], sep: str = '_') -> dict:
    """
    Decompose a string according to the list of regexp, 
    assumming that every regexp block are splitted by `sep` (or by '.';
    product names use both as separators between tokens)
    """
    l = []
    sep_re = f'[{re.escape(sep)}.]'
    seps = [sep_re for i in range(len(regexps[:-1]))] + ['']
    for i in range(100): # Prefer using for loop rather than while loop
        if i == len(seps): break
        check = re.match(f'({regexps[i]}){seps[i]}', name)
        if check: 
            l.append(check.group(1))
            name = name[check.span()[1]:]
        else: 
            log.error('name cannot be decompose by regexp list. '
                      f'{name} does not match {regexps[i]}')
    return l

def get_fields(name: str, out: list = None):
    """
    Returns every field name that can be change in product name
    """
    if out is None: out = []
    if len(name) == 0: return out
    check = re.match('[{_]*', name)
    name = name[check.span()[1]:]
    if check: 
        check = re.match('[0-9a-zA-Z_]*}', name)
        if not check: raise Exception
        out.append(name[:check.span()[1]-1])
        name = name[check.span()[1]:]
    return get_fields(name, out)

def get_level(name: str, pattern: dict):
    """
    Returns the level of a product based on its name
    """
    decomp = decompose(name, pattern['regexp'].split(' '))
    fields = [f.strip('{}') for f in pattern['pattern'].split('_')]
    level = dict(zip(fields, decomp))['level']
    level = re.search('[123]', level)
    assert level is not None
    return int(level[0])