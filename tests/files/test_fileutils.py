
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from hygeos_core.files.fileutils import LockFile, filegen, get_git_commit, mdir


def test_lockfile():
    with TemporaryDirectory() as tmpdir:
        lock_file = Path(tmpdir)/'test'
        with LockFile(lock_file) as lf:
            assert lf.exists()
            with pytest.raises(TimeoutError):
                with LockFile(lock_file):
                    pass
        with LockFile(lock_file) as lf:
            pass


@pytest.mark.parametrize('if_exists', ['skip', 'overwrite', 'backup'])
def test_filegen(if_exists):
    @filegen(arg=0, if_exists=if_exists)
    def f(path, target_path):
        with open(path, 'w') as fd:
            fd.write('test')
        
        # check that target file is currently locked
        with pytest.raises(TimeoutError):
            with LockFile(target_path):
                pass

    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/'test'
        f(target, target)
        f(target, target)  # second call skips existing file


@pytest.mark.parametrize('if_exists', ['skip', 'overwrite', 'backup'])
def test_filegen_class(if_exists):
    class MyClass:
        def __init__(self) -> None:
            self.ncalls = 0
        @filegen(1, if_exists=if_exists)
        def method(self, path):
            self.ncalls += 1
            with open(path, 'w') as fd:
                fd.write('test')
    
    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)/'test'
        M = MyClass()
        M.method(target)
        assert M.ncalls == 1
        M.method(target)  # second call skips existing file
        assert M.ncalls == (1 if if_exists == 'skip' else 2)


def test_dirgen():
    """
    Test filegen on function that create directories
    """
    @filegen()
    def create_dir(directory: Path):
        mdir(directory)

    with TemporaryDirectory() as tmpdir:
        target_dir = Path(tmpdir)/'test'
        create_dir(target_dir)
        assert target_dir.exists()


def test_mdir():
    with TemporaryDirectory() as tmpdir:
        for _ in range(2):
            mdir(Path(tmpdir)/'managed_directory',
                 description='A sample managed directory')


def test_missing_mdir():
    """
    Check that mdir fails when a mdir.json does not exist
    in an existing directory
    """
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            mdir(Path(tmpdir), strict=True)


def test_get_git_commit():
    gc = get_git_commit()
    print(gc)
                    

def test_mdir_keeps_data(tmp_path):
    """Regression: on modification, mdir wrote the initial data instead of the
    updated data (creation_date reset, keys lost)"""
    import json
    d = tmp_path / 'managed'
    mdir(d, project='A')
    data1 = json.loads((d / 'mdir.json').read_text())
    data1['custom_key'] = 'kept'
    (d / 'mdir.json').write_text(json.dumps(data1))

    mdir(d, project='B')  # modification
    data2 = json.loads((d / 'mdir.json').read_text())
    assert data2['project'] == 'B'
    assert data2['custom_key'] == 'kept'
    assert data2['creation_date'] == data1['creation_date']


def test_filegen_overwrite_directory(tmp_path):
    """Regression: if_exists='overwrite' failed on directory outputs"""
    @filegen(if_exists='overwrite')
    def create_dir(directory: Path):
        directory.mkdir()
        (directory / 'content.txt').write_text('new')

    target = tmp_path / 'store.zarr'
    target.mkdir()
    (target / 'old.txt').write_text('old')
    create_dir(target)
    assert sorted(p.name for p in target.iterdir()) == ['content.txt']


def test_persistent_list_reload(tmp_path):
    """Regression: item access did not reload the file in concurrent mode"""
    from hygeos_core.files.fileutils import PersistentList
    fname = tmp_path / 'list.json'
    a = PersistentList(fname)
    b = PersistentList(fname)
    a.append('x')
    assert b[0] == 'x'
    a.clear()
    a.append('y')
    assert b[0] == 'y'
    assert b[0:1] == ['y']
