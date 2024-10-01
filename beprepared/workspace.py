import os 
import sqlite3
import pickle
from urllib.parse import urlparse
from beprepared.utils import copy_or_hardlink
from beprepared.logging import configure_default_logger
import hashlib
from tqdm import tqdm
import requests

from typing import List, Dict, Callable, Any, TypeVar, Generic

class Database:
    def __init__(self, path) -> None:
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(os.path.join(path, 'db.sqlite3'))

        # Make things faster
        cursor = self.db.cursor()
        cursor.execute('PRAGMA synchronous = NORMAL')
        cursor.execute('PRAGMA journal_mode = WAL')

        self.initialize_schema()

    def initialize_schema(self):
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS property_cache (
                key          TEXT PRIMARY KEY,
                value        BLOB,
                last_updated TIMESTAMP
            )
        ''')
        self.db.commit()

    # get image path for <hash>.{jpg,png,...} where path is <first 2 chars>/<second 2 chars>/<hash.{jpg,png,...}
    def get_object_path(self, hash: str) -> str:
        return os.path.join(self.path, 'objects', hash[:2], hash[2:4], hash)

    def put_object(self, bytes_or_path: bytes | str) -> str:
        if isinstance(bytes_or_path, str):
            bytes = open(bytes_or_path, 'rb').read()
            path = bytes_or_path
        else:
            bytes = bytes_or_path
            path = None
        hash = hashlib.sha256(bytes).hexdigest()
        dst_path = self.get_object_path(hash)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if path:
            copy_or_hardlink(path, dst_path)
        else:
            with open(dst_path, 'wb') as f:
                f.write(bytes)
        return hash

    def get_object(self, hash: str) -> bytes:
        path = self.get_object_path(hash)
        with open(path, 'rb') as f:
            return f.read()

    def get_prop(self, cachekey: str) -> Any | None:
        cursor = self.db.cursor()
        val = cursor.execute('SELECT value FROM property_cache WHERE key = ?', (cachekey,)).fetchone()
        if val is None:
            return None
        return pickle.loads(val[0])

    def put_prop(self, cachekey: str, value: Any) -> None:
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO property_cache (key, value, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (cachekey, pickle.dumps(value)))
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class DownloadCacheDir:
    def __init__(self, cache, dir):
        self.cache = cache
        self.dir = dir

    def download(self, url: str, relpath: str) -> str:
        path = os.path.join(self.dir, relpath)
        if os.path.exists(path):
            return path
        download_to_path(url, path)

def download_to_path(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    tmp_path = path + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with open(tmp_path, 'wb') as file, tqdm(
        desc=f"Downloading {filename}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    os.rename(tmp_path, path)

class DownloadCache:
    _cachedir = os.path.join(os.path.expanduser('~'), '.beprepared', 'cache')

    def __init__(self) -> None:
        pass

    def dir(self, name) -> str:
        return DownloadCacheDir(self, os.path.join(DownloadCache._cachedir, name))

    def download(self, url: str) -> str:
        '''returns a local path that corresponds to the given URL, only downloading if needed'''
        os.makedirs(DownloadCache._cachedir, exist_ok=True)
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        local_filename = os.path.join(DownloadCache._cachedir, url_hash)
        download_to_path(url, local_filename)
        return local_filename

class Workspace:
    _active_workspaces = []
    current = None

    def __init__(self, dir=None, logger=None) -> None:
        self.dir = dir or os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        self.db = Database(os.path.join(self.dir, '_beprepared'))
        self.nodes = []
        self.cache = DownloadCache()

        if logger:
            self.log = logger
        else:
            self.log = configure_default_logger()

    def __enter__(self):
        Workspace._active_workspaces.append(self)
        Workspace.current = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if len(Workspace._active_workspaces) == 0:
            raise Exception('No workspace active')
        if Workspace._active_workspaces[-1] != self:
            raise Exception('Workspace stack mismatch')
        Workspace._active_workspaces.pop()
        if len(Workspace._active_workspaces) > 0:
            Workspace.current = Workspace._active_workspaces[-1]
        else:
            Workspace.current = None

    def get_object_path(self, hash: str) -> str:
        return self.db.get_object_path(hash)

    def get_path(self, obj: Any): 
        return self.db.get_object_path(obj.objectid.value)

    def get_object(self, obj: Any):
        return self.db.get_object(obj.objectid.value)

    def put_object(self, bytes_or_path: bytes | str) -> str:
        return self.db.put_object(bytes_or_path)

    def run(self):
        for node in self.nodes:
            if len(node.sinks) == 0:
                node()

