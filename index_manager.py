"""
FAISS-based Face Index Manager
Manages master index (all scanned photos) and search index (current directory)
"""

import faiss
import numpy as np
import sqlite3
import json
import os
import hashlib
from perceptual_hash import compute_perceptual_hash
from pathlib import Path
from datetime import datetime
import pickle
import threading


class IndexManager:
    def __init__(self, index_dir='face_indices'):
        """
        Initialize the index manager
        
        Args:
            index_dir: Directory to store index files and metadata
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # File paths
        self.master_index_path = self.index_dir / 'master_index.faiss'
        self.metadata_db_path = self.index_dir / 'metadata.db'
        
        # FAISS index (512-dimensional face embeddings)
        self.embedding_dim = 512
        self.master_index = None
        self.search_index = None
        
        # Thread-safe database connection
        self.metadata_db = None
        self._db_lock = threading.Lock()
        
        # Initialize
        self._init_master_index()
        self._init_metadata_db()
    
    def _init_master_index(self):
        """Initialize or load the master FAISS index"""
        if self.master_index_path.exists():
            # Load existing index
            self.master_index = faiss.read_index(str(self.master_index_path))
            print(f"✅ Loaded master index with {self.master_index.ntotal} faces")
        else:
            # Create new index using IndexFlatIP (Inner Product for cosine similarity)
            self.master_index = faiss.IndexFlatIP(self.embedding_dim)
            print("✅ Created new master index")
    
    def _init_metadata_db(self):
        """Initialize SQLite database for file metadata"""
        self.metadata_db = sqlite3.connect(str(self.metadata_db_path), check_same_thread=False)
        cursor = self.metadata_db.cursor()
        
        # Check if we need to migrate schema
        cursor.execute("PRAGMA table_info(file_metadata)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Create metadata table (now supports multiple faces per file)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                face_index INTEGER NOT NULL DEFAULT 0,
                total_faces INTEGER NOT NULL DEFAULT 1,
                file_hash TEXT NOT NULL,
                modified_time REAL NOT NULL,
                indexed_time REAL NOT NULL,
                embedding_idx INTEGER NOT NULL,
                has_face BOOLEAN NOT NULL,
                face_bbox TEXT,
                file_size INTEGER,
                UNIQUE(file_path, face_index)
            )
        ''')
        
        # Create index on file_path for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_path ON file_metadata(file_path)
        ''')
        
        # Create index on file_hash for deduplication
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_hash ON file_metadata(file_hash)
        ''')
        
        self.metadata_db.commit()
        print("✅ Metadata database initialized")
    
    def _compute_file_hash(self, file_path):
        """Compute MD5 hash of file for change detection"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return None

    def _compute_perceptual_hash(self, file_path):
        """Compute perceptual hash (phash) for image file"""
        return compute_perceptual_hash(file_path)
    
    def is_file_indexed(self, file_path):
        """Check if file is already indexed and unchanged"""
        with self._db_lock:
            cursor = self.metadata_db.cursor()
            cursor.execute('SELECT file_hash, modified_time FROM file_metadata WHERE file_path = ?', 
                          (str(file_path),))
            result = cursor.fetchone()
            
            if result is None:
                return False, None
            
            stored_hash, stored_mtime = result
            
            # Check if file was modified
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime != stored_mtime:
                    # File modified - need to re-index
                    return False, stored_hash
                
                return True, stored_hash
            except:
                return False, None
    
    def add_to_index(self, file_path, embedding, face_index=0, total_faces=1, bbox=None):
        """
        Add a face embedding to the master index
        
        Args:
            file_path: Path to the image file
            embedding: Face embedding (512-dim numpy array)
            face_index: Which face in the image (0-indexed)
            total_faces: Total number of faces in the image
            bbox: Face bounding box coordinates (optional)
        
        Returns:
            Index position in FAISS
        """
        # Normalize embedding for cosine similarity
        embedding = embedding.astype('float32')
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1)

        # Add to FAISS index (thread-safe)
        with self._db_lock:
            self.master_index.add(embedding)
            embedding_idx = self.master_index.ntotal - 1

        # Compute file metadata
        file_hash = self._compute_file_hash(file_path)
        perceptual_hash = self._compute_perceptual_hash(file_path)
        modified_time = os.path.getmtime(file_path)
        indexed_time = datetime.now().timestamp()
        file_size = os.path.getsize(file_path)

        # Store metadata in database (thread-safe)
        with self._db_lock:
            cursor = self.metadata_db.cursor()
            # Add perceptual_hash column if not exists
            try:
                cursor.execute('ALTER TABLE file_metadata ADD COLUMN perceptual_hash TEXT')
            except sqlite3.OperationalError:
                pass  # Already exists
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (file_path, face_index, total_faces, file_hash, perceptual_hash, modified_time, indexed_time, embedding_idx, has_face, face_bbox, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(file_path),
                face_index,
                total_faces,
                file_hash,
                perceptual_hash,
                modified_time,
                indexed_time,
                embedding_idx,
                True,
                json.dumps(bbox) if bbox else None,
                file_size
            ))
            self.metadata_db.commit()

        return embedding_idx
    
    def add_no_face_entry(self, file_path):
        """Record that a file was processed but no face was found"""
        file_hash = self._compute_file_hash(file_path)
        modified_time = os.path.getmtime(file_path)
        indexed_time = datetime.now().timestamp()
        file_size = os.path.getsize(file_path)
        
        with self._db_lock:
            cursor = self.metadata_db.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (file_path, face_index, total_faces, file_hash, modified_time, indexed_time, embedding_idx, has_face, face_bbox, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(file_path),
                0,  # Default face index
                0,  # No faces found
                file_hash,
                modified_time,
                indexed_time,
                -1,  # No embedding index
                False,
                None,
                file_size
            ))
            self.metadata_db.commit()
    
    def build_search_index(self, directory_paths):
        """
        Build a search index for specific directories
        Only includes embeddings from files in these directories
        
        Args:
            directory_paths: List of directory paths to include
        
        Returns:
            FAISS index, list of file paths, list of face metadata dicts
        """
        # Build search index from directory files
        search_embeddings = []
        search_file_paths = []
        search_face_metadata = []
        
        with self._db_lock:
            cursor = self.metadata_db.cursor()
            
            for dir_path in directory_paths:
                # Get all indexed faces in this directory
                cursor.execute('''
                    SELECT file_path, embedding_idx, face_index, total_faces FROM file_metadata 
                    WHERE file_path LIKE ? AND has_face = 1
                ''', (f'{dir_path}%',))
                
                results = cursor.fetchall()
                
                for file_path, emb_idx, face_idx, total_faces in results:
                    # Get embedding from master index
                    if emb_idx >= 0 and emb_idx < self.master_index.ntotal:
                        embedding = self.master_index.reconstruct(int(emb_idx))
                        search_embeddings.append(embedding)
                        search_file_paths.append(file_path)
                        search_face_metadata.append({
                            'face_index': face_idx,
                            'total_faces': total_faces
                        })
        
        if len(search_embeddings) == 0:
            # Create empty index
            self.search_index = faiss.IndexFlatIP(self.embedding_dim)
            return self.search_index, [], []
        
        # Create search index
        search_embeddings = np.array(search_embeddings).astype('float32')
        self.search_index = faiss.IndexFlatIP(self.embedding_dim)
        self.search_index.add(search_embeddings)
        
        return self.search_index, search_file_paths, search_face_metadata
    
    def search(self, query_embedding, k=1000, threshold=0.3):
        """
        Search for similar faces in the search index
        
        Args:
            query_embedding: Query face embedding
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
        
        Returns:
            List of (file_path, similarity_score) tuples
        """
        if self.search_index is None or self.search_index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(k, self.search_index.ntotal)
        distances, indices = self.search_index.search(query_embedding, k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= threshold:  # Cosine similarity threshold
                results.append((idx, float(dist)))
        
        return results
    
    def get_files_to_index(self, directory):
        """
        Get list of files that need to be indexed in a directory
        
        Args:
            directory: Directory path to scan
        
        Returns:
            new_files, modified_files, total_files
        """
        from pathlib import Path
        
        SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in SUPPORTED_FORMATS:
                    all_files.append(os.path.join(root, file))
        
        new_files = []
        modified_files = []
        
        for file_path in all_files:
            is_indexed, stored_hash = self.is_file_indexed(file_path)
            if not is_indexed:
                if stored_hash is None:
                    new_files.append(file_path)
                else:
                    modified_files.append(file_path)
        
        return new_files, modified_files, len(all_files)
    
    def get_embedding(self, index):
        """
        Get embedding vector by index from search index
        
        Args:
            index: Index position in search index
        
        Returns:
            Embedding vector or None if not found
        """
        if self.search_index is None or index >= self.search_index.ntotal:
            return None
        
        # Reconstruct the embedding from FAISS index
        embedding = self.search_index.reconstruct(int(index))
        return embedding
    
    def save_master_index(self):
        """Save the master index to disk"""
        faiss.write_index(self.master_index, str(self.master_index_path))
        print(f"✅ Saved master index ({self.master_index.ntotal} faces)")
    
    def get_stats(self):
        """Get index statistics"""
        with self._db_lock:
            cursor = self.metadata_db.cursor()
            # Count unique files for each stat
            cursor.execute('SELECT COUNT(DISTINCT file_path) FROM file_metadata WHERE has_face = 1')
            faces_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(DISTINCT file_path) FROM file_metadata WHERE has_face = 0')
            no_face_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(DISTINCT file_path) FROM file_metadata')
            total_files = cursor.fetchone()[0]
            return {
                'total_files': total_files,
                'faces_found': faces_count,
                'no_faces': no_face_count,
                'index_size': self.master_index.ntotal if self.master_index else 0
            }
    
    def close(self):
        """Close database connection"""
        if self.metadata_db:
            self.metadata_db.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
