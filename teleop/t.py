import os
import pickle
import zlib

import cv2
import numpy as np

# Generate a random image (e.g. 256x256 with 3 color channels)
img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

# Save the image as a JPG file
cv2.imwrite("random_image.jpg", img)
jpg_size = os.path.getsize("random_image.jpg")

# Save the numpy array as a pickle file
with open("random_image.pkl", "wb") as f:
    pickle.dump(img, f)
pkl_size = os.path.getsize("random_image.pkl")

# Save the numpy array as a pickle with compression using zlib
compressed_pickle = zlib.compress(pickle.dumps(img))
with open("random_image_compressed.pkl", "wb") as f:
    f.write(compressed_pickle)
compressed_pickle_size = os.path.getsize("random_image_compressed.pkl")

# Print the file sizes
print(f"JPG file size: {jpg_size / 1024:.2f} KB")
print(f"Pickle file size: {pkl_size / 1024:.2f} KB")
print(f"Compressed Pickle file size: {compressed_pickle_size / 1024:.2f} KB")
