from hygeos_core.network.download import download_url


def test_download_url(tmp_path):
    """
    Test the download of a sample file
    """
    path = download_url('https://www.google.com/favicon.ico', tmp_path)
    assert path.exists()
    assert path.name == 'favicon.ico'