"""
GPU-Accelerated Face Recognition Search with FAISS Indexing
Built with InsightFace, CUDA, and FAISS
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import time
import atexit
import signal
import sys
from index_manager import IndexManager
warnings.filterwarnings('ignore')

def imread_unicode(filename):
    """
    Read image file with Unicode path support (for Arabic/Chinese filenames)
    OpenCV's imread() fails with non-ASCII paths, so we use numpy workaround
    """
    try:
        # Read file as numpy array
        with open(filename, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        # Decode image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# Configure ONNX Runtime for GPU
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    GPU_AVAILABLE = 'CUDAExecutionProvider' in providers
    if GPU_AVAILABLE:
        print(f"✅ GPU ENABLED: ONNX Runtime {ort.__version__} with CUDA")
        print(f"   Providers: {providers}")
        
        # Set GPU options for RTX 3080
        import onnxruntime.backend
        os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'  # Enable FP16 for speed
        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'  # Cache engines
except:
    GPU_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable template caching in development
app.config['TEMPLATES_AUTO_RELOAD'] = True

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Global flag to control indexing
stop_indexing_flag = False

# Initialize InsightFace
try:
    from insightface.app import FaceAnalysis
    
    # Initialize with GPU - CUDA optimized for RTX 3080
    providers_list = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if GPU_AVAILABLE else ['CPUExecutionProvider']
    face_analyzer = FaceAnalysis(providers=providers_list)
    # Larger det_size and very low threshold so small/soft faces pass detection
    face_analyzer.prepare(ctx_id=0 if GPU_AVAILABLE else -1, det_size=(1280, 1280), det_thresh=0.15)
    
    # Separate analyzer for query images with ultra-low threshold for cropped faces
    face_analyzer_query = FaceAnalysis(providers=providers_list)
    face_analyzer_query.prepare(ctx_id=0 if GPU_AVAILABLE else -1, det_size=(640, 640), det_thresh=0.01)
    
    print(f"✅ InsightFace initialized with {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")
    INSIGHTFACE_READY = True
except Exception as e:
    print(f"❌ InsightFace initialization failed: {e}")
    INSIGHTFACE_READY = False

# Initialize Index Manager
try:
    index_manager = IndexManager()
    print(f"✅ Index Manager initialized")
    stats = index_manager.get_stats()
    print(f"   Files indexed: {stats['total_files']} (faces found: {stats['faces_found']}, files without faces: {stats['no_faces']})")
except Exception as e:
    print(f"❌ Index Manager initialization failed: {e}")
    index_manager = None

def extract_face_embedding(image_path, is_query=False):
    """Extract face embeddings using InsightFace with optimized preprocessing
    
    Args:
        image_path: Path to the image
        is_query: If True, optimizes for query images (returns single best face)
                 If False, returns all detected faces for indexing
    
    Returns:
        For query images (is_query=True): Single embedding or None
        For indexing (is_query=False): List of embeddings (can be empty)
    """
    try:
        img = imread_unicode(image_path)
        if img is None:
            return None if is_query else []
        
        h, w = img.shape[:2]
        
        # OPTIMIZED PREPROCESSING for query images
        if is_query:
            # For very small images, upscale first
            if w < 400 or h < 400:
                scale = 400 / min(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                h, w = img.shape[:2]
            
            # Add 20% padding (optimized from batch testing)
            pad = int(min(h, w) * 0.2)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        # For very small images, also apply to directory images
        elif w < 300 or h < 300:
            pad = int(min(h, w) * 0.2)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        # Use optimized analyzer for query images (threshold=0.01)
        analyzer = face_analyzer_query if is_query else face_analyzer
        faces = analyzer.get(img)
        
        if len(faces) == 0:
            return None if is_query else []
        
        if is_query:
            # For query images, return only the most confident face
            face = max(
                faces,
                key=lambda x: (
                    getattr(x, 'det_score', 0.0),
                    (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
                )
            )
            return face.embedding
        else:
            # For indexing, return ALL faces sorted by confidence
            sorted_faces = sorted(
                faces,
                key=lambda x: (
                    getattr(x, 'det_score', 0.0),
                    (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
                ),
                reverse=True
            )
            return [face.embedding for face in sorted_faces]
        
    except Exception as e:
        return None if is_query else []

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between embeddings"""
    try:
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity * 100)  # Convert to percentage
    except:
        return 0


def deduplicate_faiss_results(results, file_paths, duplicate_threshold=0.97):
    """Remove duplicate FAISS matches using filename, size, and embedding similarity"""
    deduplicated = []
    seen_hashes = {}
    duplicates_log = []
    duplicate_count = 0

    import sqlite3
    from index_manager import IndexManager
    # Get perceptual hash for each file from the database
    index_manager_instance = index_manager if 'index_manager' in globals() else IndexManager()
    with index_manager_instance._db_lock:
        cursor = index_manager_instance.metadata_db.cursor()
        for idx, similarity in results:
            file_path = file_paths[int(idx)]
            cursor.execute('SELECT perceptual_hash FROM file_metadata WHERE file_path = ?', (file_path,))
            row = cursor.fetchone()
            perceptual_hash = row[0] if row else None
            is_duplicate = False
            reason = ""
            if perceptual_hash and perceptual_hash in seen_hashes:
                is_duplicate = True
                reason = f"same perceptual hash (matches {seen_hashes[perceptual_hash]})"
            if is_duplicate:
                duplicate_count += 1
                duplicates_log.append({'path': file_path, 'reason': reason})
            else:
                deduplicated.append((idx, similarity))
                seen_hashes[perceptual_hash] = file_path
    return deduplicated, duplicate_count, duplicates_log


def deduplicate_match_list(match_list):
    """Deduplicate list of match dictionaries (used by non-stream search)"""
    deduplicated = []
    seen_hashes = {}
    duplicates_removed = 0
    import sqlite3
    from index_manager import IndexManager
    index_manager_instance = index_manager if 'index_manager' in globals() else IndexManager()
    with index_manager_instance._db_lock:
        cursor = index_manager_instance.metadata_db.cursor()
        for match in match_list:
            file_path = match.get('path') or match.get('file')
            if not file_path:
                deduplicated.append(match)
                continue
            cursor.execute('SELECT perceptual_hash FROM file_metadata WHERE file_path = ?', (file_path,))
            row = cursor.fetchone()
            perceptual_hash = row[0] if row else None
            if perceptual_hash and perceptual_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes[perceptual_hash] = file_path
            deduplicated.append(match)
    return deduplicated, duplicates_removed

def process_batch_insightface(batch_args):
    """Process a batch of images with InsightFace"""
    query_embedding, image_paths, threshold = batch_args
    results = []
    
    for img_path in image_paths:
        try:
            target_embedding = extract_face_embedding(img_path)
            if target_embedding is not None:
                similarity = compute_similarity(query_embedding, target_embedding)
                
                if similarity >= threshold:
                    results.append({
                        'path': img_path,
                        'similarity': similarity
                    })
        except Exception as e:
            continue
    
    return results

def find_matching_faces_insightface(reference_image_path, search_directory, threshold=70):
    """
    GPU-accelerated face search using InsightFace
    threshold: Similarity threshold 0-100 (default 70)
    """
    matches = []
    
    if not INSIGHTFACE_READY:
        return [{'error': 'InsightFace not ready'}]
    
    # Get all images
    image_files = []
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in SUPPORTED_FORMATS:
                image_files.append(os.path.join(root, file))
    
    total = len(image_files)
    if total == 0:
        return []
    
    # Extract query embedding once
    query_embedding = extract_face_embedding(reference_image_path, is_query=True)
    
    if query_embedding is None:
        return [{'error': 'No face detected in query image'}]
    
    # Process in batches with GPU
    batch_size = 1500 if GPU_AVAILABLE else 100
    total_batches = (total + batch_size - 1) // batch_size
    max_workers = 12 if GPU_AVAILABLE else 4
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i in range(0, total, batch_size):
            batch = image_files[i:i + batch_size]
            future = executor.submit(
                process_batch_insightface,
                (query_embedding, batch, threshold)
            )
            futures.append(future)
        
        for future in as_completed(futures):
            batch_results = future.result()
            matches.extend(batch_results)
    
    # Sort by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<path:filepath>')
def serve_image(filepath):
    """Serve image files from absolute paths"""
    try:
        from urllib.parse import unquote
        filepath = unquote(filepath)
        
        if os.path.exists(filepath) and os.path.isfile(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        else:
            return "Image not found", 404
    except Exception as e:
        return f"Error loading image: {str(e)}", 500

@app.route('/search_stream', methods=['POST'])
def search_faces_stream():
    """Stream search results as they are found with FAISS indexing"""
    
    # Extract all request data BEFORE the generator (must be in request context)
    if 'image' not in request.files:
        return jsonify({'type': 'error', 'message': 'No image provided'}), 400
    
    file = request.files['image']
    search_dir = request.form.get('directory', '')
    threshold = float(request.form.get('threshold', 30))
    
    if file.filename == '':
        return jsonify({'type': 'error', 'message': 'No image selected'}), 400
    
    if not search_dir or not os.path.exists(search_dir):
        return jsonify({'type': 'error', 'message': 'Invalid directory path'}), 400
    
    # Save file immediately (still in request context)
    temp_path = 'temp_upload_stream.jpg'
    file.save(temp_path)
    
    def generate():
        try:
            # Verify file was saved
            if not os.path.exists(temp_path):
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to save uploaded image'})}\n\n"
                return
            
            # Extract query face embedding
            query_embedding = extract_face_embedding(temp_path, is_query=True)
            
            if query_embedding is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No face detected in the uploaded image'})}\n\n"
                return
            
            # Send confirmation that face was detected
            yield f"data: {json.dumps({'type': 'info', 'message': '✅ Face detected! Starting indexing...'})}\n\n"
            
            # PHASE 1: INDEXING
            # Get files that need to be indexed
            new_files, modified_files, total_files = index_manager.get_files_to_index(search_dir)
            files_to_index = new_files + modified_files
            
            yield f"data: {json.dumps({'type': 'info', 'message': f'📂 Found {total_files} images ({len(files_to_index)} need indexing)'})}\n\n"
            
            # Index new/modified files
            if len(files_to_index) > 0:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'indexing', 'total': len(files_to_index)})}\n\n"
                
                indexed_count = 0
                faces_found_count = 0
                no_face_count = 0
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=12) as executor:
                    future_to_path = {
                        executor.submit(index_single_file, filepath): filepath
                        for filepath in files_to_index
                    }
                    
                    for future in as_completed(future_to_path):
                        # Check stop flag
                        global stop_indexing_flag
                        if stop_indexing_flag:
                            stop_indexing_flag = False  # Reset flag
                            executor.shutdown(wait=False, cancel_futures=True)
                            yield f"data: {json.dumps({'type': 'info', 'message': '⚠️ Indexing stopped by user'})}\n\n"
                            break
                        
                        result = future.result()
                        indexed_count += 1
                        if result:
                            faces_found_count += 1
                        else:
                            no_face_count += 1
                        
                        # Calculate speed
                        elapsed = time.time() - start_time
                        speed = indexed_count / elapsed if elapsed > 0 else 0
                        
                        # Send detailed indexing stats every 5 files or on completion
                        if indexed_count % 5 == 0 or indexed_count == len(files_to_index):
                            stats_data = {
                                'type': 'indexing_stats', 
                                'indexed': indexed_count, 
                                'total': len(files_to_index),
                                'faces_found': faces_found_count,
                                'no_face': no_face_count,
                                'speed': round(speed, 1)
                            }
                            yield f"data: {json.dumps(stats_data)}\n\n"
                        
                        # Save index every 100 files to prevent data loss
                        if indexed_count % 100 == 0:
                            index_manager.save_master_index()
                
                # Final save after all files indexed
                index_manager.save_master_index()
                yield f"data: {json.dumps({'type': 'info', 'message': f'✅ Indexed {indexed_count} new/modified files'})}\n\n"
            
            # PHASE 2: BUILD SEARCH INDEX
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'preparing'})}\n\n"
            search_index, file_paths, face_metadata = index_manager.build_search_index([search_dir])
            
            if len(file_paths) == 0:
                yield f"data: {json.dumps({'type': 'info', 'message': 'No faces found in indexed images'})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'total_processed': 0})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'info', 'message': f'🔍 Searching {len(file_paths)} indexed faces...'})}\n\n"
            
            # PHASE 3: SEARCH
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'matching'})}\n\n"
            
            # Convert threshold from 0-100 to 0-1 for cosine similarity
            threshold_normalized = threshold / 100.0
            
            # Search using FAISS
            results = index_manager.search(query_embedding, k=len(file_paths), threshold=threshold_normalized)
            
            # Stream results
            for idx, similarity in results:
                metadata = face_metadata[int(idx)]
                match = {
                    'path': file_paths[int(idx)],
                    'similarity': float(similarity * 100),  # Convert back to 0-100
                    'face_index': metadata['face_index'],
                    'total_faces': metadata['total_faces']
                }
                yield f"data: {json.dumps({'type': 'match', 'match': match})}\n\n"
            
            yield f"data: {json.dumps({'type': 'complete', 'total_processed': len(file_paths), 'matches': len(results)})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def index_single_file(filepath):
    """Index a single file - extract embeddings for ALL faces and add to master index
    Returns: Number of faces found (0 if no faces)"""
    try:
        embeddings = extract_face_embedding(filepath, is_query=False)
        if embeddings and len(embeddings) > 0:
            # Add each face to the index
            total_faces = len(embeddings)
            for face_idx, embedding in enumerate(embeddings):
                index_manager.add_to_index(filepath, embedding, face_index=face_idx, total_faces=total_faces)
            return total_faces
        else:
            index_manager.add_no_face_entry(filepath)
            return 0
    except Exception as e:
        print(f"Error indexing {filepath}: {e}")
        index_manager.add_no_face_entry(filepath)
        return 0

@app.route('/search', methods=['POST'])
def search_faces():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        search_dir = request.form.get('directory', '')
        if not search_dir or not os.path.exists(search_dir):
            return jsonify({'error': 'Invalid directory path'}), 400
        
        # Get threshold from user (default 70)
        threshold = float(request.form.get('threshold', 70))
        
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        
        matches = find_matching_faces_insightface(temp_path, search_dir, threshold=threshold)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Filter out error messages from matches
        valid_matches = [m for m in matches if 'error' not in m]
        
        return jsonify({
            'success': True,
            'matches': valid_matches,
            'total_found': len(valid_matches),
            'gpu_enabled': GPU_AVAILABLE
        })
    
    except Exception as e:
        if os.path.exists('temp_upload.jpg'):
            os.remove('temp_upload.jpg')
        return jsonify({'error': str(e)}), 500

@app.route('/open_location', methods=['POST'])
def open_location():
    """Open file location in Windows Explorer"""
    try:
        import subprocess
        data = request.json
        filepath = data.get('path', '')
        # Normalize path for Windows
        norm_path = os.path.normpath(filepath)
        norm_path = norm_path.replace('/', '\\')  # Ensure backslashes
        if not os.path.exists(norm_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        # Try to select the file using shell=True for compatibility
        explorer_cmd = f'explorer /select,"{norm_path}"'
        subprocess.Popen(explorer_cmd, shell=True)
        return jsonify({'success': True})
    except Exception:
        # Fallback: open containing folder
        try:
            folder = os.path.dirname(norm_path)
            if os.path.exists(folder):
                subprocess.Popen(f'explorer "{folder}"', shell=True)
                return jsonify({'success': True, 'fallback': True})
            else:
                return jsonify({'success': False, 'error': 'Folder not found'}), 404
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/deduplicate', methods=['POST'])
def deduplicate_matches():
    """Deduplicate search results based on file similarity"""
    try:
        data = request.json
        matches = data.get('matches', [])
        
        if not matches:
            return jsonify({'success': False, 'error': 'No matches provided'}), 400
        
        # Create file_paths list from matches in order
        file_paths = [match['path'] for match in matches]
        
        # Convert matches to (index, similarity) tuples for deduplication
        results = [
            (idx, match['similarity'] / 100.0)  # Convert from 0-100 to 0-1
            for idx, match in enumerate(matches)
        ]
        
        # Apply deduplication
        deduplicated_results, duplicates_removed, duplicates_log = deduplicate_faiss_results(
            results,
            file_paths,
            duplicate_threshold=0.97
        )
        
        # Convert back to match format
        deduplicated_matches = [
            {
                'path': file_paths[int(idx)],
                'similarity': float(similarity * 100)
            }
            for idx, similarity in deduplicated_results
        ]
        
        return jsonify({
            'success': True,
            'deduplicated_matches': deduplicated_matches,
            'duplicates_removed': duplicates_removed,
            'duplicates_log': duplicates_log[:10]  # Return first 10 for reference
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export_files', methods=['POST'])
def export_files():
    """Copy selected files to destination folder"""
    try:
        import shutil
        data = request.json
        files = data.get('files', [])
        destination = data.get('destination', '')
        
        if not files:
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Create destination folder if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        copied = 0
        failed = 0
        
        for filepath in files:
            try:
                if os.path.exists(filepath):
                    filename = os.path.basename(filepath)
                    dest_path = os.path.join(destination, filename)
                    
                    # Handle duplicate filenames
                    counter = 1
                    base, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(destination, f"{base}_{counter}{ext}")
                        counter += 1
                    
                    shutil.copy2(filepath, dest_path)
                    copied += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Failed to copy {filepath}: {e}")
                failed += 1
        
        return jsonify({
            'success': True,
            'copied': copied,
            'failed': failed,
            'destination': destination
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/index_stats', methods=['GET'])
def index_stats():
    """Get statistics about the index"""
    try:
        stats = index_manager.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_indexing', methods=['POST'])
def stop_indexing():
    """Stop the current indexing operation"""
    global stop_indexing_flag
    stop_indexing_flag = True
    return jsonify({'success': True, 'message': 'Stopping indexing...'})

@app.route('/delete_indices', methods=['POST'])
def delete_indices():
    """Delete all index files and reset"""
    global index_manager
    try:
        import shutil
        indices_path = 'face_indices'
        
        if os.path.exists(indices_path):
            # Close database connection first
            if index_manager:
                index_manager.close()
            
            # Delete the folder
            shutil.rmtree(indices_path)
            
            # Reinitialize index manager
            index_manager = IndexManager()
            
            return jsonify({'success': True, 'message': 'Index deleted and reset successfully'})
        else:
            return jsonify({'success': True, 'message': 'No index exists'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def cleanup():
    """Clean up resources on shutdown"""
    try:
        print("\n🔄 Shutting down gracefully...")
        if index_manager:
            index_manager.save_master_index()
            index_manager.close()
        print("✅ Cleanup complete")
    except:
        pass

# Register cleanup handlers
atexit.register(cleanup)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⚠️  Interrupted by user")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 GPU-ACCELERATED FACE RECOGNITION SERVER")
    print("=" * 70)
    print(f"GPU: {'✅ CUDA Enabled' if GPU_AVAILABLE else '❌ CPU Only'}")
    print(f"Framework: InsightFace")
    print(f"Server: http://127.0.0.1:5001")
    print("=" * 70)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        cleanup()
