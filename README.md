# Face Recognition Search Web Application

A web application that allows you to search for photos of a specific person in a directory by uploading or pasting a face image.

## Features

- 🖼️ **Upload or Paste Images**: Drag & drop, click to upload, or paste screenshots (Ctrl+V)
- � **GPU-Accelerated Deep Learning**: Uses FaceNet with PyTorch for state-of-the-art face recognition
- 📁 **Recursive Directory Search**: Searches through all subdirectories
- ⚡ **Multi-threaded Processing**: Processes images in parallel for faster results
- 📊 **Similarity Scores**: Shows match percentage for each found image
- 🎨 **Modern UI**: Clean, responsive interface with gradient design

## Installation

### Current Setup (Python 3.10 + Virtual Environment)

The application uses:
- Flask (web framework)
- PyTorch with CUDA support (GPU acceleration)
- facenet-pytorch (advanced face recognition with MTCNN + InceptionResnetV1)
- Pillow (image handling)
- Virtual environment at `venv/`

### System Requirements

- Python 3.10
- NVIDIA GPU with CUDA support (optional but recommended)
- Visual C++ Redistributable 2015-2022
- 8GB+ RAM recommended

## How to Use

1. **Start the application**:
   ```
   venv\Scripts\python.exe app_gpu.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Enter the directory path** where you want to search for photos
   - Example: `C:\Users\YourName\Pictures`
   - Example: `D:\Photos\Vacation`

4. **Upload or paste a face image**:
   - Click the upload area to select a file
   - Drag and drop an image
   - Press Ctrl+V to paste a screenshot

5. **Click "Search for Matching Faces"**

6. **View results** with similarity scores and file locations

## How It Works

1. **MTCNN Face Detection**: Detects and aligns faces in images
2. **FaceNet Embeddings**: Extracts 512-dimensional face embeddings using InceptionResnetV1 trained on VGGFace2
3. **Cosine Similarity**: Compares face embeddings using cosine distance
4. **Multi-threaded Processing**: Processes images in batches for optimal GPU utilization
5. **Results sorted by similarity** (highest match percentage first)

## Files in this Project

- `app_gpu.py` - GPU-accelerated version using FaceNet (recommended)
- `app.py` - CPU version with multiprocessing using OpenCV
- `templates/index.html` - Web interface
- `static/style.css` - Styling
- `venv/` - Python 3.10 virtual environment

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)
- TIFF (.tiff)
- WebP (.webp)

## Configuration

You can adjust the similarity threshold in `app_gpu.py`:
- Default: `threshold=70` (0-100 scale)
- Lower value = fewer but more accurate matches
- Higher value = more matches but may include false positives

## Notes

- First run will download FaceNet models (~100MB) automatically
- GPU version is significantly faster if CUDA is available
- CPU version will work but processes slower
- Processing time depends on the number of images and hardware
- The app shows progress updates as batches are processed

## Troubleshooting

**Issue**: No faces detected in reference image
- Make sure the uploaded image clearly shows a face
- Try a different image with better lighting

**Issue**: Search takes too long
- Reduce the directory size or specify a more specific subdirectory
- The app processes images sequentially for accuracy

**Issue**: Images not displaying in results
- This is normal - browsers restrict local file access
- Use the "Copy Path" button to get the file location
- Navigate to the path in Windows Explorer

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Recognition**: FaceNet (InceptionResnetV1 trained on VGGFace2)
- **Deep Learning**: PyTorch with CUDA support
- **Image Processing**: Pillow
- **Frontend**: HTML5, CSS3, JavaScript
- **Concurrency**: ThreadPoolExecutor for parallel processing
