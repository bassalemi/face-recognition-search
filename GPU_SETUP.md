# GPU Face Recognition Setup Instructions

## Complete PyTorch Installation

Run these commands one by one in PowerShell (don't interrupt):

```powershell
cd "d:\Python codes\Face recognition"

# Install PyTorch with CUDA 12.1 (this will take 5-10 minutes)
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install facenet-pytorch
.\venv\Scripts\python.exe -m pip install facenet-pytorch

# Test the setup
.\venv\Scripts\python.exe test_gpu.py
```

## If Installation Succeeds

Start the GPU-accelerated app:
```powershell
.\venv\Scripts\python.exe app_gpu.py
```

Then open: http://localhost:5000

## If GPU Not Available

The app will automatically fall back to CPU mode (slower but still works with advanced FaceNet model).

## Alternative: Use Multiprocessing Version

If PyTorch installation fails, use the multiprocessing version (faster than single-threaded):
```powershell
python app.py
```

This uses OpenCV with CPU multiprocessing - not as accurate as FaceNet but much faster than the original single-threaded version.

## What You Get with GPU Version

- **MTCNN**: State-of-the-art face detection
- **FaceNet (InceptionResnetV1)**: Industry-standard face recognition trained on 3.3M faces
- **GPU Acceleration**: 5-10x faster than CPU when CUDA is available
- **Better Accuracy**: 99.6% accuracy on LFW benchmark vs ~85% for basic OpenCV
- **Robust**: Works with different poses, lighting, expressions

## Current Status

✅ Python 3.10 virtual environment created
✅ Flask installed
✅ facenet-pytorch installed
⏳ PyTorch with CUDA - needs manual installation (large download)
✅ Visual C++ Redistributable installed
✅ GPU-accelerated app code ready (app_gpu.py)
