# debug_speed.py - Run this once then delete
import sys
import os
sys.path.append(os.path.dirname(__file__))

import time
import numpy as np
from PIL import Image
from transforms_extra import build_transforms_with_extras

# Create test image
test_image = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))

# Test transform speed
transform = build_transforms_with_extras(320, train=True)

times = []
for i in range(10):
    start = time.time()
    result = transform(test_image)
    times.append(time.time() - start)

print(f"⏱️  Average transform time: {np.mean(times):.3f}s")
print(f"📐 Output shape: {result.shape}")