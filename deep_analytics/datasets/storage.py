import os
import sys
import re
import errno
import threading
import hashlib
import boto3
import tarfile
import zipfile
from urllib.parse import urlparse
from pathlib import Path
from botocore.exceptions import NoCredentialsError

import torch
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, TypeVar, Union
from torch.hub import download_url_to_file

default_cache_dir = os.path.join(Path(torch.hub.get_dir()).parent, "data")

# Set the default AWS profile and endpoint
os.environ['AWS_PROFILE'] = 'wasabi'
os.environ['S3_ENDPOINT_URL'] = 'https://s3.wasabisys.com'

__all__ = ['download_if_needed', 'extract_archive', 'download_file', 
           'download_from_url', 'download_from_s3', 'calculate_file_hash']

def download_if_needed(url, cache_dir=default_cache_dir, progress=True, check_hash=False,
                       delete_archive=True):
    # Parse the URL to extract the filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Extract basename excluding file extensions
    extensions = "".join(Path(filename).suffixes)
    basename = Path(filename).name.replace(extensions, "")

    # Determine the directory where the extracted contents will be stored
    extract_dir = os.path.join(cache_dir, basename)

    # Check if the directory already exists
    if os.path.exists(extract_dir):
        # The directory exists, no need to download and extract
        return extract_dir

    # Download the file as the directory does not exist    
    downloaded_file = download_file(url, cache_dir, progress, check_hash)
    
    # extract archive
    print(f"Extracting file {downloaded_file} to folder {extract_dir}")
    extract_archive(downloaded_file, extract_dir, delete_archive=delete_archive, show_progress=progress)
    
    return extract_dir

def extract_archive(archive_path, extract_to, delete_archive=True, show_progress=True, progress_interval=0.005):
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"No archive file found at {archive_path}")

    def _report_progress(members, total):
        last_reported = 0
        interval_count = max(1, int(total * progress_interval))  # Calculate the interval count
        for i, member in enumerate(members):
            if show_progress and (i - last_reported >= interval_count or i == total - 1):
                print(f"Extracting file {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
                last_reported = i
            yield member

    # Determine the archive type and extract
    if archive_path.endswith(('.tar', '.tar.gz', '.tgz', '.gz')):
        with tarfile.open(archive_path, 'r:*') as archive:
            members = archive.getmembers()
            archive.extractall(path=extract_to, members=_report_progress(members, len(members)))
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as archive:
            members = archive.infolist()
            for i, member in enumerate(_report_progress(members, len(members))):
                archive.extract(member, path=extract_to)
    else:
        raise ValueError(f"Unsupported archive type for file {archive_path}")

    print()  # Print newline at end
    # Optionally delete the archive after extraction
    if delete_archive:
        os.remove(archive_path)

def download_file(url, cache_dir=default_cache_dir, progress=True, check_hash=True):
    
    if url.startswith("s3://"):
        cached_filename = download_from_s3(url, cache_dir=cache_dir, progress=progress, check_hash=check_hash)
    elif url.startswith("http://") or url.startswith("https://"):
        cached_filename = download_from_url(url, data_dir=cache_dir, progress=progress, check_hash=check_hash)
    else:
        raise ValueError(f"URL should be a bucket object s3://<buckname>/<objectkey>, or valid web url http/https://example.com/<filename>: {url}")
    
    return cached_filename

def download_from_url(url: str, data_dir: Optional[str] = None, progress: bool = True, check_hash: bool = False,
                      file_name: Optional[str] = None) -> Dict[str, Any]:
    r"""Downloads the object at the given URL.

    If downloaded file is a .tar file or .tar.gz file, it will be automatically
    decompressed.

    If the object is already present in `data_dir`, it's deserialized and
    returned.
    
    The default value of ``data_dir`` is ``<hub_dir>/../data`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        data_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')
    
    if data_dir is None:
        hub_dir = torch.hub.get_dir()
        data_dir = default_cache_dir
    
    HASH_REGEX = re.compile(r'-([a-f0-9]{4,64})\.')
    
    try:
        os.makedirs(data_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(data_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            #r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            #hash_prefix = r.group(1) if r else None
            matches = HASH_REGEX.findall(filename) # matches is Optional[Match[str]]
            hash_prefix = matches[-1] if matches else None

        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    
    return cached_file

def download_from_s3(s3_url, cache_dir=default_cache_dir, progress=True, check_hash=True) -> str:
    """
    Download a file from an S3 bucket using AWS credentials and check its hash if required.

    Args:
        s3_url (str): S3 URL of the file to download (s3://bucket-name/path/to/file).
        cache_dir (str): Directory where the file will be stored.
        progress (bool): Whether to display download progress.
        check_hash (bool): Whether to check the hash of the file.

    Returns:
        str: Path to the downloaded file.
    """

    # Parse the S3 URL
    s3_bucket, s3_key = parse_s3_url(s3_url)

    # Prepare the cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cached_filename = os.path.join(cache_dir, os.path.basename(s3_key))

    # Check if the file is already cached
    if os.path.exists(cached_filename):
        print(f"File already exists: {cached_filename}")
        return cached_filename                    

    # Explicitly get the endpoint URL and profile from the environment
    endpoint_url = os.environ.get('S3_ENDPOINT_URL', 'https://s3.amazonaws.com')
    profile_name = os.environ.get('AWS_PROFILE', 'default')

    # Create a boto3 session with the specified profile
    session = boto3.Session(profile_name=profile_name)

    # Create an S3 client with the specified endpoint URL
    s3 = session.client('s3', endpoint_url=endpoint_url)

    try:
        # Download the file from S3
        print(f"Downloading {s3_key} from bucket {s3_bucket}...")
        total_size = get_file_size(s3, s3_bucket, s3_key)
        progress = DownloadProgressBar(cached_filename, total_size)
        s3.download_file(s3_bucket, s3_key, cached_filename, Callback=progress)

        if check_hash and not is_hash_matching(cached_filename, s3_key):
            print(f"Hash mismatch for file: {cached_filename}. Removing file from cache_dir...")
            os.remove(cached_filename)
        #else:
        #    print(f"\nDownloaded to {cached_filename}")
    except NoCredentialsError:
        print("Error: AWS credentials not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return cached_filename

def parse_s3_url(s3_url):
    """
    Parse the S3 URL into bucket name and key.

    Args:
        s3_url (str): S3 URL of the file (s3://bucket-name/path/to/file).

    Returns:
        tuple: A tuple containing the bucket name and the key.
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("URL must start with 's3://'")

    parts = s3_url[5:].split('/', 1)
    if len(parts) < 2:
        raise ValueError("URL must include both bucket name and key")

    return parts[0], parts[1]

def is_hash_matching(file_path, expected_hash):
    """
    Check if the hash of the file at file_path matches the expected_hash.

    Args:
        file_path (str): Path to the file to check.
        expected_hash (str): Expected hash value.

    Returns:
        bool: True if hashes match, False otherwise.
    """
    # HASH_REGEX = re.compile(r'-([a-f0-9]{8,})\.')
    HASH_REGEX = re.compile(r'-([a-f0-9]{4,64})\.')
    
    # Extract expected hash from file_path
    match = HASH_REGEX.search(file_path)
    if not match:
        print("No hash found in filename. Cannot verify hash.")
        return False

    expected_hash = match.group(1)
    actual_hash = calculate_file_hash(file_path)

    return actual_hash.startswith(expected_hash)

def calculate_file_hash(file_path):
    """
    Calculate the SHA256 hash of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: SHA256 hash of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def list_bucket(bucket_name, subfolder=None):
    """
    List objects in an S3 bucket, optionally filtered by a subfolder.

    Args:
        bucket_name (str): The name of the S3 bucket.
        subfolder (str, optional): The subfolder path.

    Returns:
        list: A list of object keys in the specified bucket and subfolder.
    """

    # Explicitly get the endpoint URL and profile from the environment
    endpoint_url = os.environ.get('S3_ENDPOINT_URL', 'https://s3.amazonaws.com')
    profile_name = os.environ.get('AWS_PROFILE', 'default')

    # Create a boto3 session with the specified profile
    session = boto3.Session(profile_name=profile_name)

    # Create an S3 client with the specified endpoint URL
    s3 = session.client('s3', endpoint_url=endpoint_url)
    object_keys = []

    # If a subfolder is specified, ensure it ends with a '/'
    if subfolder and not subfolder.endswith('/'):
        subfolder += '/'

    # List objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=subfolder)

    for page in page_iterator:
        if "Contents" in page:
            for obj in page['Contents']:
                object_keys.append(obj['Key'])

    return object_keys

class DownloadProgressBar:
    def __init__(self, filename, total_size, update_interval=.5):
        self.filename = filename
        self.total_size = total_size
        self._seen_so_far = 0
        self._last_reported_percentage = 0
        self._update_interval = update_interval  # Update every 0.5%
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            current_percentage = (self._seen_so_far / self.total_size) * 100

            # Check if update is needed based on the interval
            if (current_percentage - self._last_reported_percentage) >= self._update_interval or current_percentage >= 100:
                sys.stdout.write(f"\r{self.filename}: {current_percentage:.2f}% downloaded")
                sys.stdout.flush()
                self._last_reported_percentage = current_percentage

    def finish(self):
        # This method will be called when the download is complete
        print("")  # Print a newline character
        
def get_file_size(s3_client, bucket, key):
    response = s3_client.head_object(Bucket=bucket, Key=key)
    return response['ContentLength']

def get_filename_without_suffixes(file_path):
    extensions = "".join(Path(file_path).suffixes)
    filename_without_suffixes = Path(file_path).name.replace(extensions, "")
    return filename_without_suffixes