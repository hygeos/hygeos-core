from netrc import netrc


def get_auth(name: str) -> dict:
    """
    Returns a dictionary with credentials, using .netrc

    `name` is the identifier (= `machine` in .netrc). This allows for several accounts
    on a single machine.
    The url is returned as `account`
    """
    ret = netrc().authenticators(name)
    if ret is None:
        raise ValueError(
            f'Please provide entry "{name}" in ~/.netrc ; '
            f'example: machine {name} login <login> password <passwd> account <url>')
    (login, account, password) = ret

    return {'user': login,
            'password': password,
            'url': account}


def get_auth_dhus(name):
    auth = get_auth(name)
    api_url = auth['url'] or {
        'scihub': 'https://scihub.copernicus.eu/dhus/',
        'coda': 'https://coda.eumetsat.int',
    }[name]
    return {'user': auth['user'],
            'password': auth['password'],
            'api_url': api_url}