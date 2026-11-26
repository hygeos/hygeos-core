from ftplib import FTP, error_perm
from typing import Union, Dict
from pathlib import Path
import threading
import fnmatch

from tqdm import tqdm
from hygeos_core.files.fileutils import filegen
from hygeos_core.network.auth import get_auth


def get_auth_ftp(name) -> Dict:
    """
    get netrc credentials for use with pyfilesystem's
    FTPFS or ftplib's FTP
    
    Ex: FTP(**get_auth_ftp(<name>))
    """
    auth = get_auth(name)
    return {'host': auth['url'],
            'user': auth['user'],
            'passwd': auth['password']}

def get_url_ftpfs(name):
    """
    get netrc credentials for use with pyfilesystem's fs.open_fs

    Ex: fs.open_fs(get_url_ftpfs(<name>))
    """
    auth = get_auth(name)
    user = auth['user']
    password = auth['password']
    machine = auth['url']
    return f"ftp://{user}:{password}@{machine}/"



@filegen(1)
def ftp_download(ftp: FTP,
                 file_local: Path,
                 dir_server: Union[str, Path],
                 verbose=True):
    """
    Downloads `file_local` on ftp, from server directory `dir_server`

    The file name on the server is determined by `file_local.name`

    Example:
        ftp = FTP(**get_auth_ftp('my_server'))
        local_file = Path('./data.txt')
        ftp_download(ftp, local_file, '/remote/folder')
        ftp.quit()

    Refs:
        https://stackoverflow.com/questions/19692739/
        https://stackoverflow.com/questions/73534659/
    """
    fname = file_local.name
    path_server = str(Path('/')/str(dir_server)/fname)
    size = ftp.size(path_server)
    pbar = tqdm(desc=f'Downloading {path_server}',
                total=size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                disable=not verbose)
    with open(file_local, 'wb') as fp, ftp.transfercmd('RETR ' + path_server) as sock:
        def background():
            while True:
                block = sock.recv(1024*1024)
                if not block:
                    break
                fp.write(block)
                pbar.update(len(block))
        t = threading.Thread(target=background)
        t.start()
        noops_sent = 0
        while t.is_alive():
            t.join(60)
            ftp.putcmd('NOOP')
            noops_sent += 1
    pbar.close()
    ftp.voidresp()
    for _ in range(noops_sent):
        ftp.voidresp()
    assert file_local.exists()


def ftp_file_exists(ftp: FTP, path_server: Union[Path, str]) -> bool:
    try:  # test file existence
        ftp.size(str(path_server))
        return True
    except error_perm:
        return False

def ftp_create_dir(ftp: FTP, path_server: Union[Path, str]):
    try:
        ftp.cwd(str(path_server))
    except error_perm:
        # if path_server does not exist
        assert Path(path_server).parent != Path(path_server)
        ftp_create_dir(ftp, Path(path_server).parent)
        ftp.mkd(str(path_server))
    ftp.cwd('/')


def ftp_upload(ftp: FTP,
               file_local: Path,
               dir_server: str,
               if_exists='skip',
               blocksize=8192,
               verbose=True,
               ):
    """
    FTP upload function
    
    - Use temporary files
    - Create remote directories
    - if_exists:
        'skip': skip the existing file
        'error': raise an error on existing file
        'overwrite': overwrite existing file
    """
    path_server = f'{dir_server}/{file_local.name}'
    if ftp_file_exists(ftp, path_server):
        if if_exists == 'skip':
            # check size consistency
            assert file_local.stat().st_size == ftp.size(path_server)
            return
        elif if_exists == 'overwrite':
            ftp.delete(path_server)
        elif if_exists == 'error':
            raise FileExistsError
        else:
            raise ValueError(f'skip = {if_exists}')

    assert not ftp_file_exists(ftp, path_server)
    
    # cleanup tmp file
    path_server_tmp = f'{dir_server}/{file_local.name}.tmp'
    if ftp_file_exists(ftp, path_server_tmp):
        ftp.delete(path_server_tmp)

    # create directory
    ftp_create_dir(ftp, dir_server)
        
    cmd = f'STOR {path_server_tmp}'
    pbar = tqdm(desc=f'Uploading {path_server}',
                total=file_local.stat().st_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                disable=not verbose)
    errors = []
    with ftp.transfercmd(cmd) as sock, open(file_local, 'rb') as fp:
        def background():
            try:
                while True:
                    buf = fp.read(blocksize)
                    if not buf:
                        break
                    sock.sendall(buf)
                    pbar.update(len(buf))
            except BaseException as e:
                errors.append(e)
        t = threading.Thread(target=background)
        t.start()
        noops_sent = 0
        while t.is_alive():
            t.join(60)
            ftp.putcmd('NOOP')
            noops_sent += 1
    pbar.close()
    if errors:
        # failed transfer: the temporary file is not renamed to its final name
        # (it is removed at the next upload)
        raise errors[0]
    ftp.voidresp()
    for _ in range(noops_sent):
        ftp.voidresp()
    ftp.rename(path_server_tmp, path_server)


def ftp_list(ftp: FTP, dir_server: str, pattern: str='*'):
    '''
    Returns the list of fles matching `pattern` on `dir_server`
    '''
    ftp.cwd(dir_server)
    ls = ftp.nlst()
    return fnmatch.filter(ls, pattern)
