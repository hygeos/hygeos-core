#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gzip
import bz2
import tarfile
import io
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from hygeos_core.files.uncompress import uncompress_decorator, uncompress, uncompress_single
from hygeos_core.dates import duration
import pytest


        

def test_duration():
    assert duration('2w') == timedelta(weeks=2)
    assert duration('2d') == timedelta(days=2)
    assert duration('2h') == timedelta(hours=2)


def test_uncompress_decorator():
    """Test the uncompress_decorator with a custom target name function"""
    
    def url_to_filename(url):
        """Extract filename from URL, removing extension for uncompressed target"""
        filename = Path(url).name
        # Remove compression extensions to get the target name
        if filename.endswith('.zip'):
            return filename[:-4]
        return filename
    
    @uncompress_decorator(target_name_func=url_to_filename, verbose=False)
    def mock_download(identifier, tmpdir):
        """Mock function that simulates downloading a compressed file"""
        # Create a sample zip file in tmpdir
        sample_content = b"This is sample content from a compressed file."
        zip_path = Path(tmpdir) / f"downloaded_{Path(identifier).stem}.zip"
        
        with ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("sample_data1.txt", sample_content)
            zipf.writestr("sample_data2.txt", sample_content)
        
        return zip_path
    
    with TemporaryDirectory(prefix='test_decorator_') as test_dir:
        # Test with a mock URL identifier
        test_url = "https://example.com/dataset.zip"
        
        # First call should uncompress and create the target
        result1 = mock_download(test_url, test_dir)
        expected_target = Path(test_dir) / "dataset"
        
        assert result1 == expected_target
        assert result1.exists()
        assert result1.is_dir()  # zip extracts to a directory
        
        # Check that the content was extracted correctly
        extracted_file = result1 / "sample_data1.txt"
        assert extracted_file.exists()
        assert extracted_file.read_text() == "This is sample content from a compressed file."
        
        # Second call should return the same path without re-extracting
        result2 = mock_download(test_url, test_dir)
        assert result2 == result1
        assert result2 == expected_target
        
        # Test with a different identifier
        test_url2 = "https://example.com/other_data.zip"
        result3 = mock_download(test_url2, test_dir)
        expected_target2 = Path(test_dir) / "other_data"
        
        assert result3 == expected_target2
        assert result3.exists()
        assert result3 != result1  # Different targets for different identifiers


@pytest.mark.parametrize("format_ext,compress_func", [
    ('.gz', gzip.open),
    ('.bz2', bz2.BZ2File),
])
def test_uncompress_single_formats(format_ext, compress_func):
    """Test uncompress_single with different compression formats"""
    with TemporaryDirectory(prefix='test_single_') as tmpdir:
        test_content = b"Test content for compression."
        compressed_file = Path(tmpdir) / f"test.txt{format_ext}"
        
        # Create compressed file
        with compress_func(compressed_file, 'wb') as f:
            f.write(test_content)
        
        # Test decompression
        output_file = Path(tmpdir) / "decompressed.txt"
        result = uncompress_single(compressed_file, output_file)
        
        assert result == output_file
        assert output_file.exists()
        assert output_file.read_bytes() == test_content


@pytest.mark.parametrize("archive_format,create_func", [
    ('zip', lambda path: _create_zip_multi(path)),
    ('tar.gz', lambda path: _create_tar_gz_multi(path)),
    ('tar.bz2', lambda path: _create_tar_bz2_multi(path)),
])
@pytest.mark.parametrize("extract_to,expected_files", [
    ('subdir', lambda tmpdir: [(Path(tmpdir) / "test" / "file1.txt", "content"), 
                               (Path(tmpdir) / "test" / "subdir" / "file2.txt", "content")]),
    ('target_dir', lambda tmpdir: [(Path(tmpdir) / "file1.txt", "content"), 
                                   (Path(tmpdir) / "subdir" / "file2.txt", "content")]),
    ('auto', lambda tmpdir: [(Path(tmpdir) / "test" / "file1.txt", "content"), 
                             (Path(tmpdir) / "test" / "subdir" / "file2.txt", "content")]),
])
def test_uncompress_archive_formats_and_modes(archive_format, create_func, extract_to, expected_files):
    """Test uncompress with different archive formats and extract modes"""
    with TemporaryDirectory(prefix='test_archive_') as tmpdir:
        archive_file = Path(tmpdir) / f"test.{archive_format}"
        
        # Create archive with multiple files
        create_func(archive_file)
        
        # Test extraction
        uncompress(archive_file, tmpdir, extract_to=extract_to)
        
        # Check expected files exist
        for file_path, content in expected_files(tmpdir):
            assert file_path.exists()
            assert file_path.read_text() == content


def _create_zip_multi(path):
    """Helper to create a zip file with multiple files"""
    with ZipFile(path, 'w') as zf:
        zf.writestr("file1.txt", "content")
        zf.writestr("subdir/file2.txt", "content")


def _create_tar_gz_multi(path):
    """Helper to create a tar.gz file with multiple files"""
    with tarfile.open(path, 'w:gz') as tf:
        # Add file1.txt
        info1 = tarfile.TarInfo(name="file1.txt")
        info1.size = len(b"content")
        tf.addfile(info1, fileobj=io.BytesIO(b"content"))
        # Add subdir/file2.txt
        info2 = tarfile.TarInfo(name="subdir/file2.txt")
        info2.size = len(b"content")
        tf.addfile(info2, fileobj=io.BytesIO(b"content"))


def _create_tar_bz2_multi(path):
    """Helper to create a tar.bz2 file with multiple files"""
    with tarfile.open(path, 'w:bz2') as tf:
        # Add file1.txt
        info1 = tarfile.TarInfo(name="file1.txt")
        info1.size = len(b"content")
        tf.addfile(info1, fileobj=io.BytesIO(b"content"))
        # Add subdir/file2.txt
        info2 = tarfile.TarInfo(name="subdir/file2.txt")
        info2.size = len(b"content")
        tf.addfile(info2, fileobj=io.BytesIO(b"content"))


@pytest.mark.parametrize('path', [
    "file1.txt",
    "subdir/file2.txt",
])
def test_uncompress_single_dir(path: str):
    """
    Uncompressing a directory with a single file/directory, check appropriate returned
    path
    """
    with TemporaryDirectory(prefix='test_archive_') as tmpdir:
        archive_file = Path(tmpdir) / "test.zip"
        with ZipFile(archive_file, 'w') as zf:
            zf.writestr(path, "content")

        ret = uncompress(archive_file, tmpdir)
        assert ret.name != Path(tmpdir).name

def test_uncompress_single_unsupported():
    """Test uncompress_single with unsupported format"""
    with TemporaryDirectory(prefix='test_single_') as tmpdir:
        unsupported_file = Path(tmpdir) / "test.txt"
        unsupported_file.write_text("plain text")
        output_file = Path(tmpdir) / "output.txt"
        
        with pytest.raises(ValueError, match="Unsupported single file compression format"):
            uncompress_single(unsupported_file, output_file)
