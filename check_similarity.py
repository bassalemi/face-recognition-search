import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
print("Initializing InsightFace...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# The two duplicate images
image1_path = r"D:\SD Cards\Old stuff\2010\images\digital camera\DCIM\105_2903\101_2303\IMGP1577.JPG"
image2_path = r"D:\SD Cards\Old stuff\2010\images\digital camera\DCIM\101_2303\IMGP1577.JPG"

print(f"\nAnalyzing images:")
print(f"Image 1: {image1_path}")
print(f"Image 2: {image2_path}")

# Read images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None:
    print(f"ERROR: Could not read image 1")
    exit(1)
    
if img2 is None:
    print(f"ERROR: Could not read image 2")
    exit(1)

# Detect faces and get embeddings
faces1 = app.get(img1)
faces2 = app.get(img2)

if len(faces1) == 0:
    print("ERROR: No face detected in image 1")
    exit(1)
    
if len(faces2) == 0:
    print("ERROR: No face detected in image 2")
    exit(1)

print(f"\nImage 1: Found {len(faces1)} face(s)")
print(f"Image 2: Found {len(faces2)} face(s)")

# Compare the first face from each image
embedding1 = faces1[0].normed_embedding
embedding2 = faces2[0].normed_embedding

# Calculate cosine similarity (dot product of normalized embeddings)
similarity = np.dot(embedding1, embedding2)

print(f"\n{'='*60}")
print(f"🎯 SIMILARITY SCORE: {similarity:.6f} ({similarity*100:.4f}%)")
print(f"{'='*60}")

# Provide threshold recommendation
print(f"\n📊 THRESHOLD RECOMMENDATIONS:")
print(f"   Current threshold: 0.97 (97%)")
print(f"   Similarity found:  {similarity:.6f} ({similarity*100:.4f}%)")

if similarity > 0.97:
    print(f"   ✅ Current threshold will catch this duplicate")
else:
    recommended_threshold = round(similarity - 0.01, 4)  # Slightly below the similarity
    print(f"   ❌ Current threshold is too high!")
    print(f"   💡 Recommended threshold: {recommended_threshold:.4f} ({recommended_threshold*100:.2f}%)")

print(f"\n🔍 Additional Info:")
print(f"   Embedding dimension: {len(embedding1)}")
print(f"   Embedding 1 norm: {np.linalg.norm(embedding1):.6f}")
print(f"   Embedding 2 norm: {np.linalg.norm(embedding2):.6f}")
