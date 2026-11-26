from contextlib import contextmanager
from ftplib import error_perm
from unittest import mock

import pytest

from hygeos_core.network.ftp import ftp_upload


class FakeFTP:
    """Minimal in-memory FTP server mock (no network)"""
    def __init__(self, fail_upload=False):
        self.files = {}
        self.fail_upload = fail_upload
        self.size = mock.Mock(side_effect=self._size)
        self.delete = mock.Mock(side_effect=lambda p: self.files.pop(p))
        self.rename = mock.Mock(side_effect=self._rename)
        self.cwd = mock.Mock()
        self.mkd = mock.Mock()
        self.putcmd = mock.Mock()
        self.voidresp = mock.Mock()

    def _size(self, path):
        if path not in self.files:
            raise error_perm('550 not found')
        return len(self.files[path])

    def _rename(self, src, dst):
        self.files[dst] = self.files.pop(src)

    @contextmanager
    def transfercmd(self, cmd):
        assert cmd.startswith('STOR ')
        path = cmd[len('STOR '):]
        self.files[path] = b''
        sock = mock.Mock()

        def sendall(buf):
            if self.fail_upload:
                self.files[path] += buf[:1]  # partial write
                raise ConnectionError('interrupted')
            self.files[path] += buf
        sock.sendall.side_effect = sendall
        yield sock


def test_ftp_upload_uses_tmp_file(tmp_path):
    """Regression: upload was done directly under the final name"""
    f = tmp_path / 'data.bin'
    f.write_bytes(b'0123456789')
    ftp = FakeFTP()
    ftp_upload(ftp, f, '/remote', verbose=False)
    assert ftp.files == {'/remote/data.bin': b'0123456789'}
    src, dst = ftp.rename.call_args.args
    assert dst == '/remote/data.bin'
    assert src != dst


def test_ftp_upload_interrupted(tmp_path):
    """Regression: an interrupted upload left a partial file under the final
    name, and a later upload with if_exists='skip' failed on an assert"""
    f = tmp_path / 'data.bin'
    f.write_bytes(b'0123456789')
    ftp = FakeFTP(fail_upload=True)
    # the transfer fails: the error is raised, and the partial file is not
    # renamed to its final name
    with pytest.raises(ConnectionError):
        ftp_upload(ftp, f, '/remote', verbose=False)
    assert '/remote/data.bin' not in ftp.files
    ftp.rename.assert_not_called()

    # retry: the leftover temporary file is cleaned up and the upload succeeds
    ftp.fail_upload = False
    ftp_upload(ftp, f, '/remote', if_exists='skip', verbose=False)
    assert ftp.files == {'/remote/data.bin': b'0123456789'}
