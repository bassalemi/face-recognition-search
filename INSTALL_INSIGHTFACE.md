# Installing InsightFace for GPU Acceleration

## Step 1: Install Visual C++ Build Tools

1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer
3. Select: **"Desktop development with C++"**
4. Install (takes ~10 minutes, needs ~6GB)
5. Restart your terminal

## Step 2: Fix NumPy Version

```powershell
.\venv\Scripts\python.exe -m pip install "numpy<2.0"
```

## Step 3: Install InsightFace

```powershell
.\venv\Scripts\python.exe -m pip install insightface
```

## Step 4: Test Installation

```powershell
.\venv\Scripts\python.exe -c "import insightface; print('InsightFace version:', insightface.__version__)"
```

## Benefits Once Installed:

- **5-10x faster** than DeepFace on GPU
- **True batch processing** - processes multiple images simultaneously
- **Better GPU memory usage** - can handle larger batches
- **Lower CPU overhead** - more efficient
- **Native ONNX Runtime integration** - optimized for your RTX 3080

## Alternative (if you don't want to install Build Tools):

The current DeepFace + ONNX Runtime setup is already GPU-accelerated and working well. 
InsightFace would be faster, but requires the C++ compiler to build from source.
