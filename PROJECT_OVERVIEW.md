# GPU-Accelerated Face Recognition Search Engine

## Overview

A high-performance web application that enables users to search through large photo collections by simply uploading a face image. The system leverages GPU acceleration to rapidly scan thousands of images and identify matching faces with remarkable accuracy, making it ideal for organizing personal photo libraries or finding specific individuals across extensive image datasets.

## Functionality

### Core Features

**Intelligent Face Search**
- Upload or paste a screenshot of any face to use as a search query
- Automatically scans entire directory structures to find matching faces
- Real-time streaming results - matches appear instantly as they're found
- Live progress tracking showing files processed during the search
- Adjustable similarity thresholds (30-95%) for post-search filtering

**Advanced Face Detection**
- Optimized preprocessing for cropped or low-resolution query images
- Automatic upscaling of small images (< 400px) to improve detection accuracy
- Smart padding (20%) applied to query images for better face boundary detection
- Ultra-low detection threshold (0.01) for query images to handle challenging crops
- Dual analyzer system with different thresholds for query vs. directory images

**Rich User Interface**
- Drag-and-drop or paste image upload with instant preview
- Directory browser for easy folder selection
- Grid view with similarity percentages for all matches
- Individual image viewer popup for detailed inspection
- Checkboxes for selecting multiple results
- "Select All" and "Select Above Threshold" bulk selection options
- One-click "Open Location" to view files in Windows Explorer
- Export functionality to copy selected images to a destination folder

**Performance Optimizations**
- GPU-accelerated processing using CUDA (NVIDIA RTX 3080)
- Batch processing of 1500 images per batch
- Multi-threaded execution with 12 concurrent workers
- Efficient Server-Sent Events (SSE) streaming for real-time updates
- Progress bar with percentage and file count feedback

## Technology Stack

### Backend Framework
- **Flask** - Lightweight Python web framework for the server
- **Python 3.13** - Modern Python with optimized performance
- **Server-Sent Events (SSE)** - Real-time streaming of search results

### Face Recognition & AI
- **InsightFace** - State-of-the-art deep learning face recognition library
- **buffalo_l Model** - High-accuracy face detection and embedding model
- **ONNX Runtime** - Optimized inference engine with GPU support
- **Cosine Similarity** - Mathematical comparison of face embeddings (512-dimensional vectors)

### GPU Acceleration
- **CUDA 12.6** - NVIDIA's parallel computing platform
- **cuDNN 9.15** - Deep neural network library optimized for GPUs
- **CUDAExecutionProvider** - ONNX Runtime GPU backend
- **FP16 Precision** - Half-precision floating-point for faster computation
- **TensorRT Engine Caching** - Reuses optimized execution engines

### Image Processing
- **OpenCV (cv2)** - Computer vision library for image manipulation
- **NumPy** - Numerical computing for embedding operations
- **INTER_CUBIC Interpolation** - High-quality image upscaling
- **BORDER_REPLICATE** - Edge padding technique for face boundary preservation

### Frontend Technologies
- **HTML5** - Modern semantic markup
- **CSS3** - Gradient backgrounds, animations, and responsive grid layouts
- **Vanilla JavaScript** - Native Fetch API with ReadableStream for SSE parsing
- **Progressive Enhancement** - Real-time results with graceful fallbacks

### System Integration
- **Windows Explorer Integration** - Direct file location opening via `subprocess`
- **File Operations** - Copy with duplicate handling using `shutil`
- **Multi-threading** - `ThreadPoolExecutor` for concurrent image processing
- **Path Handling** - Cross-platform path management with `pathlib`

## Technical Architecture

### Detection Strategy
The system employs a sophisticated dual-analyzer approach:

1. **Query Image Analyzer** (det_size=640x640, threshold=0.01)
   - Designed for cropped or screenshot faces
   - Extremely permissive detection for challenging inputs
   - Upscales small images before processing
   - Adds 20% padding to improve boundary detection

2. **Directory Image Analyzer** (det_size=1280x1280, threshold=0.15)
   - Optimized for full photos with good quality
   - Higher threshold to reduce false positives
   - Larger detection size for better accuracy on standard photos

### Preprocessing Pipeline
1. Load image with OpenCV
2. Check dimensions - upscale if < 400px (query) or < 300px (directory)
3. Apply 20% padding using border replication
4. Run face detection with appropriate analyzer
5. Extract 512-dimensional face embedding
6. Normalize embedding for cosine similarity comparison

### Search Workflow
1. User uploads face image via web interface
2. Flask saves image and extracts embedding (with preprocessing)
3. Directory scan collects all supported image formats (.jpg, .jpeg, .png, .bmp)
4. Images divided into batches of 1500 for parallel processing
5. ThreadPoolExecutor processes 12 images concurrently
6. Each match streams immediately to frontend via SSE
7. Progress updates every 50 files
8. Results displayed in real-time with adjustable filtering

## Performance Metrics

- **Processing Speed**: ~12 images/second on RTX 3080 (GPU-accelerated)
- **Batch Size**: 1500 images per batch
- **Concurrent Workers**: 12 threads
- **Search Dataset**: Optimized for thousands of images
- **Memory Efficiency**: Streaming results prevent memory overflow
- **Upload Limit**: 16MB max file size

## Use Cases

- Personal photo library organization
- Finding all photos of a specific person across years of photos
- Event photography - locating all images of attendees
- Family archive management
- Professional photography workflows
- Security/surveillance image searches (with appropriate permissions)

---

*Built with modern AI and GPU acceleration technologies to deliver instant, accurate face recognition at scale.*
