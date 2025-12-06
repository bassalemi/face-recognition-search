import sqlite3
import os
from perceptual_hash import compute_perceptual_hash

def add_perceptual_hash_to_index(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Add perceptual_hash column if not exists
    try:
        cursor.execute('ALTER TABLE file_metadata ADD COLUMN perceptual_hash TEXT')
    except sqlite3.OperationalError:
        pass  # Already exists
    cursor.execute('SELECT DISTINCT file_path FROM file_metadata')
    rows = cursor.fetchall()
    # Filter only files without perceptual hash
    files_to_hash = []
    for row in rows:
        file_path = row[0]
        if not file_path or not os.path.exists(file_path):
            continue
        cursor.execute('SELECT perceptual_hash FROM file_metadata WHERE file_path = ? LIMIT 1', (file_path,))
        existing_hash = cursor.fetchone()
        if existing_hash and existing_hash[0]:
            print(f'Skipped (already hashed): {file_path}')
            continue
        files_to_hash.append(file_path)

    # Multiprocessing for perceptual hash computation
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    print(f'Computing perceptual hashes for {len(files_to_hash)} images...')
    results = {}
    with ProcessPoolExecutor() as executor:
        future_to_path = {executor.submit(compute_perceptual_hash, fp): fp for fp in files_to_hash}
        for future in tqdm(as_completed(future_to_path), total=len(files_to_hash), desc='Hashing images'):
            file_path = future_to_path[future]
            try:
                phash = future.result()
                results[file_path] = phash
                # tqdm handles progress, so only print errors
            except Exception as e:
                print(f'Error hashing {file_path}: {e}')

    # Update database in batch
    for file_path, phash in results.items():
        cursor.execute('UPDATE file_metadata SET perceptual_hash = ? WHERE file_path = ?', (phash, file_path))
    conn.commit()
    conn.commit()
    conn.close()

if __name__ == '__main__':
    db_path = os.path.join('face_indices', 'metadata.db')
    add_perceptual_hash_to_index(db_path)
