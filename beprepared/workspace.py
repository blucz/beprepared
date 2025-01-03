import os
import sqlite3
import pickle
import logging
from fnmatch import fnmatch
from urllib.parse import urlparse
from beprepared.utils import copy_or_hardlink
from beprepared.logging import configure_default_logger
import hashlib
from beprepared.web import WebInterface
import requests
import threading
from tqdm import tqdm
from beprepared.web import WebInterface

from typing import List, Dict, Callable, Any, TypeVar, Generic

class Database:
    def __init__(self, log, path) -> None:
        self.log = log
        os.makedirs(path, exist_ok=True)
        self.tls_db = threading.local()
        self.path = path

        self.initialize_schema()

    @property
    def db(self):
        if not hasattr(self.tls_db, 'db'):
            self.tls_db.db = sqlite3.connect(os.path.join(self.path, 'db.sqlite3'))

            # Make things faster
            cursor = self.tls_db.db.cursor()
            cursor.execute('PRAGMA synchronous = NORMAL')
            cursor.execute('PRAGMA journal_mode = WAL')
            self.tls_db.db.commit()
        return self.tls_db.db

    def initialize_schema(self):
        self.log.info('Initializing database schema')
        cursor = None
        try:
            cursor = self.db.cursor()
            
            # Create migrations table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Get current schema version
            cursor.execute('SELECT MAX(version) FROM migrations')
            result = cursor.fetchone()
            current_version = result[0] if result and result[0] is not None else 0
            self.log.info(f'Current database schema version: {current_version}')
            
            # Apply migrations in order
            if current_version < 1:
                self.log.info('Applying migration to version 1: Creating property_cache table')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS property_cache (
                        key          TEXT PRIMARY KEY,
                        value        BLOB,
                        last_updated TIMESTAMP
                    )
                ''')
                cursor.execute('INSERT INTO migrations (version) VALUES (1)')
                current_version = 1
                self.log.info('Migration to version 1 complete')

            if current_version < 2:
                self.log.info('Applying migration to version 2: Adding domain column and migrating keys')

                cursor.execute('ALTER TABLE property_cache RENAME TO property_cache_v1')

                # Create a new table with value+domain as primary key
                cursor.execute('''
                    CREATE TABLE property_cache (
                        key          TEXT,
                        domain       TEXT,
                        value        BLOB,
                        last_updated TIMESTAMP,
                        PRIMARY KEY (key, domain)
                    )
                ''')

                # Copy data from old table to new table
                cursor.execute('''
                    INSERT INTO property_cache (key, domain, value, last_updated)
                    SELECT key, NULL, value, last_updated FROM property_cache_v1
                ''')
                
                # Migrate humanfilter and humantag keys
                keys_to_delete = []
                for prefix in ['humanfilter', 'humantag']:
                    self.log.info(f'Migrating {prefix} keys to include domain')
                    # Get all matching keys
                    cursor.execute(f"SELECT key, value, last_updated FROM property_cache WHERE key LIKE '{prefix}(%'")
                    rows = cursor.fetchall()
                    
                    for (old_key, value, last_updated) in rows:
                        # Extract parts from old format: prefix("domain","hash")
                        if not old_key.startswith(prefix + '('):
                            continue
                            
                        parts = old_key[len(prefix)+1:-1].split(',')
                        if len(parts) != 2:
                            continue
                            
                        domain = parts[0].strip('"')
                        hash_part = parts[1].strip('"')
                        
                        # Create new key format: prefix("hash")
                        new_key = f'{prefix}("{hash_part}")'

                        keys_to_delete.append(old_key)
                        
                        # Insert new key with domain, handling potential conflicts
                        cursor.execute('''
                            INSERT OR REPLACE INTO property_cache (key, value, domain, last_updated)
                            VALUES (?, ?, ?, ?)
                        ''', (new_key, value, domain, last_updated))
                
                # Delete old keys
                for key in keys_to_delete:
                    cursor.execute('DELETE FROM property_cache WHERE key = ? AND DOMAIN IS NULL', (key,))

                # Drop the old table 
                cursor.execute('DROP TABLE property_cache_v1')
                
                cursor.execute('INSERT INTO migrations (version) VALUES (2)')
                current_version = 2
                self.log.info('Migration to version 2 complete')
            
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logging.error(f'Error during schema initialization: {str(e)}')
            raise
        finally:
            if cursor:
                cursor.close()

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

    def get_prop(self, cachekey: str, domain: str | None = None) -> Any | None:
        try:
            cursor = self.db.cursor()
            val = cursor.execute('SELECT value FROM property_cache WHERE key = ? AND (domain IS ? OR ? IS NULL)', 
                               (cachekey, domain, domain)).fetchone()
            if val is None:
                return None
            return pickle.loads(val[0])
        finally:
            cursor.close()

    def has_prop(self, cachekey: str, domain: str | None = None) -> bool:
        try:
            cursor = self.db.cursor()
            return cursor.execute('SELECT 1 FROM property_cache WHERE key = ? AND (domain IS ? OR ? IS NULL)', 
                                (cachekey, domain, domain)).fetchone() is not None
        finally:
            cursor.close()

    def put_prop(self, cachekey: str, value: Any, domain: str | None = None) -> None:
        try:
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO property_cache (key, value, domain, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (cachekey, pickle.dumps(value), domain))
            self.db.commit()
        finally:
            cursor.close()

    def list_props(self, pattern: str = None, domain: str = None) -> List[tuple]:
        """List properties in cache, optionally filtered by glob pattern and domain
        
        Args:
            pattern: Optional glob pattern to filter properties by (e.g. "human*", "*.jpg")
            domain: Optional domain to filter properties by
        """
        try:
            cursor = self.db.cursor()
            query = 'SELECT key, domain, value FROM property_cache'
            params = []
            
            conditions = []
            if domain is not None:
                conditions.append('domain IS ?')
                params.append(domain)
                
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
                
            cursor.execute(query, params)
            results = [(key, domain, pickle.loads(value)) for key, domain, value in cursor.fetchall()]
            
            # Apply glob pattern filtering if specified
            if pattern:
                results = [(key, domain, value) for key, domain, value in results if fnmatch(key, pattern)]
                
            return results
        finally:
            cursor.close()
            
    def count_props(self, pattern: str, domain: str = None) -> int:
        """Count properties matching glob pattern and optional domain
        
        Returns:
            int: Number of matching properties
        """
        try:
            cursor = self.db.cursor()
            query = 'SELECT key FROM property_cache'
            params = []
            
            conditions = []
            if domain is not None:
                conditions.append('domain IS ?')
                params.append(domain)
                
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
                
            cursor.execute(query, params)
            keys = [row[0] for row in cursor.fetchall()]
            
            # Count keys matching the glob pattern
            return sum(1 for key in keys if fnmatch(key, pattern))
        finally:
            cursor.close()

    def clear_props(self, pattern: str, domain: str = None) -> int:
        """Delete properties matching glob pattern and optional domain
        
        Returns:
            int: Number of properties deleted
        """
        try:
            # First get matching keys
            cursor = self.db.cursor()
            query = 'SELECT key FROM property_cache'
            params = []
            
            conditions = []
            if domain is not None:
                conditions.append('domain IS ?')
                params.append(domain)
                
            if conditions:
                query += ' WHERE ' + ' and '.join(conditions)
                
            cursor.execute(query, params)
            keys = [row[0] for row in cursor.fetchall()]
            
            # Filter keys by pattern
            keys_to_delete = [key for key in keys if fnmatch(key, pattern)]
            
            if not keys_to_delete:
                return 0
                
            # Delete matching keys
            placeholders = ','.join('?' * len(keys_to_delete))
            delete_query = f'DELETE FROM property_cache WHERE key IN ({placeholders})'
            if domain is not None:
                delete_query += ' AND domain IS ?'
                keys_to_delete.append(domain)
                
            cursor.execute(delete_query, keys_to_delete)
            deleted = cursor.rowcount
            self.db.commit()
            return deleted
        finally:
            cursor.close()

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

class Abort(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)

class Workspace:
    _active_workspaces = []
    current = None

    def __init__(self, dir=None, logger=None, port=8989) -> None:
        if logger:
            self.log = logger
        else:
            self.log = configure_default_logger()

        self.dir = dir or os.getcwd()
        os.makedirs(self.dir, exist_ok=True)
        self.db = Database(self.log, os.path.join(self.dir, '_beprepared'))
        self.nodes = []
        self.cache = DownloadCache()
        self.tmp_dir = os.path.join(self.dir, 'tmp')

        os.makedirs(self.tmp_dir, exist_ok=True)

        self.web = WebInterface(self.log, debug=False, port=port)

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
        
        # Clean up web interface
        if hasattr(self, 'web'):
            self.web.stop()

    def get_object_path(self, hash: str) -> str:
        return self.db.get_object_path(hash)

    def get_path(self, obj: Any): 
        return self.db.get_object_path(obj.objectid.value)

    def get_object(self, obj: Any):
        return self.db.get_object(obj.objectid.value)

    def put_object(self, bytes_or_path: bytes | str) -> str:
        return self.db.put_object(bytes_or_path)

    def list_props(self, prefix: str = None, domain: str = None) -> List[tuple]:
        """List properties in cache, optionally filtered by prefix and domain
        
        Args:
            prefix: Optional prefix to filter properties by
            domain: Optional domain to filter properties by
            
        Returns:
            List of tuples containing (key, domain, value) for each property
        """
        return self.db.list_props(prefix, domain)

    def count_props(self, prefix: str, domain: str = None) -> int:
        """Count properties matching prefix and optional domain
        
        Args:
            prefix: String prefix to match properties to count
            domain: Optional domain to restrict counting to
            
        Returns:
            int: Number of matching properties
        """
        return self.db.count_props(prefix, domain)

    def clear_props(self, prefix: str, domain: str = None) -> int:
        """Delete properties matching prefix and optional domain
        
        Args:
            prefix: String prefix to match properties to delete
            domain: Optional domain to restrict deletion to
            
        Returns:
            int: Number of properties deleted
        """
        return self.db.clear_props(prefix, domain)

    @classmethod
    def clear_database(cls, workspace_dir: str) -> bool:
        """Delete the database for the given workspace directory
        
        Args:
            workspace_dir: Path to workspace directory
            
        Returns:
            bool: True if database was deleted, False if it didn't exist
        """
        import shutil
        db_path = os.path.join(workspace_dir or os.getcwd(), '_beprepared')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            return True
        return False

    def run(self):
        # start web interface
        self.web.start()
        try:
            for node in self.nodes:
                if len(node.sinks) == 0:
                    node()
        except KeyboardInterrupt:
            self.log.info("\n\nStopped by user..Exiting.\n")
        except Abort as e:
            self.log.error(str(e))
        self.log.info("Workspace finished")

