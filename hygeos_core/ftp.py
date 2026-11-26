import warnings
from hygeos_core.network.ftp import (
    ftp_download,
    get_auth_ftp,
    get_url_ftpfs,
    ftp_file_exists,
    ftp_create_dir,
    ftp_upload,
    ftp_list,
)

warnings.warn("Please import from hygeos_core.network.ftp", DeprecationWarning)
