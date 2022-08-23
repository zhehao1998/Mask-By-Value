import cv2
import numpy as np
import os
import copy
import time

# Edit accordingly
classes = ["cat", "dog"]
class_RGB = [(40, 189, 113), (10, 22, 194)]  # important that order corresponds to label

# Find grayscale mapping of the mask
class_GRAY = []

for lab in class_RGB:
    gray = lab[0] * 0.299 + lab[1] * 0.587 + lab[2] * 0.114
    gray = int(gray)
    class_GRAY.append(gray)

# Read RGB masks
mask_path = "new masks"  # The path of original mask
save_folder_path = "class value mask"  # The save path

time_start = time.time()

# Generate new masks with pixel values corresponding to class labels
for mask in os.listdir(mask_path):
    img_path = os.path.join(mask_path, mask)
    mask_GRAY = cv2.imread(img_path, 0)
    class_values = np.unique(mask_GRAY)
    new_mask = copy.deepcopy(mask_GRAY)

    # Map each pixel to class
    for unique_value in class_values:
        # background pixels default to class 0
        if unique_value:
            label = class_GRAY.index(unique_value) + 1
            new_mask = np.where(new_mask == unique_value, label, new_mask)

    new_mask_path = os.path.join(save_folder_path, mask)
    cv2.imwrite(new_mask_path, new_mask)
    print("Successfully converted {mask_no} to labelled mask".format(mask_no=mask))

time_elapsed = time.time() - time_start

print("Conversion completed in {sec:.2f}s".format(sec=time_elapsed))