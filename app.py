"""
GPU-Accelerated Face Recognition Search
Built with InsightFace and CUDA
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
warnings.filterwarnings('ignore')

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

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}

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

def extract_face_embedding(image_path, is_query=False):
    """Extract face embedding using InsightFace with optimized preprocessing"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
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
            return None
        
        # Get the most confident face
        face = max(
            faces,
            key=lambda x: (
                getattr(x, 'det_score', 0.0),
                (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
            )
        )
        
        return face.embedding
        
    except Exception as e:
        return None

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
    """Stream search results as they are found"""
    
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
            yield f"data: {json.dumps({'type': 'info', 'message': '✅ Face detected! Scanning directory...'})}\n\n"
            
            # Scan directory and process files
            all_image_files = []
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if Path(file).suffix.lower() in SUPPORTED_FORMATS:
                        all_image_files.append(os.path.join(root, file))
            
            # Send info about how many files found
            yield f"data: {json.dumps({'type': 'info', 'message': f'📂 Found {len(all_image_files)} images to scan'})}\n\n"
            
            batch_size = 1500
            processed = 0
            
            for i in range(0, len(all_image_files), batch_size):
                batch = all_image_files[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=12) as executor:
                    future_to_path = {
                        executor.submit(process_image_for_match, filepath, query_embedding, threshold): filepath
                        for filepath in batch
                    }
                    
                    for future in as_completed(future_to_path):
                        result = future.result()
                        processed += 1
                        
                        if result:
                            # Stream this match immediately
                            yield f"data: {json.dumps({'type': 'match', 'match': result})}\n\n"
                        
                        # Send progress update every 50 files
                        if processed % 50 == 0:
                            yield f"data: {json.dumps({'type': 'progress', 'processed': processed, 'total': len(all_image_files)})}\n\n"
            
            yield f"data: {json.dumps({'type': 'complete', 'total_processed': processed})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            pass
    
    return Response(generate(), mimetype='text/event-stream')

def process_image_for_match(filepath, query_embedding, threshold):
    """Process a single image and return match if similarity >= threshold"""
    try:
        target_embedding = extract_face_embedding(filepath, is_query=False)
        if target_embedding is not None:
            similarity = compute_similarity(query_embedding, target_embedding)
            if similarity >= threshold:
                return {
                    'path': filepath,
                    'similarity': float(similarity)
                }
    except:
        pass
    return None

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
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Open Explorer and select the file
        subprocess.Popen(f'explorer /select,"{filepath}"')
        return jsonify({'success': True})
    
    except Exception as e:
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

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 GPU-ACCELERATED FACE RECOGNITION SERVER")
    print("=" * 70)
    print(f"GPU: {'✅ CUDA Enabled' if GPU_AVAILABLE else '❌ CPU Only'}")
    print(f"Framework: InsightFace")
    print(f"Server: http://127.0.0.1:5000")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
