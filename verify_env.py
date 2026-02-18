import numpy, scipy, cv2
import torch
import sys

print(f"Python: {sys.version}")
print(f"Numpy: {numpy.__version__}")
print(f"Scipy: {scipy.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
try:
    print(f"CUDA Version: {torch.version.cuda}")
except:
    print("CUDA Version: None")

expected_numpy = "1.26.4"
expected_scipy = "1.11.4"
# OpenCV can vary slightly in build, but 4.8.1 is key
expected_opencv_prefix = "4.8.1"

errors = []
if numpy.__version__ != expected_numpy:
    errors.append(f"Numpy mismatch: expected {expected_numpy}, got {numpy.__version__}")
if scipy.__version__ != expected_scipy:
    errors.append(f"Scipy mismatch: expected {expected_scipy}, got {scipy.__version__}")
if not cv2.__version__.startswith(expected_opencv_prefix):
    errors.append(f"OpenCV mismatch: expected {expected_opencv_prefix}*, got {cv2.__version__}")

if errors:
    print("❌ ENVIRONMENT MISMATCH:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("✅ Environment matches locked plan.")
    sys.exit(0)
