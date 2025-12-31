import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_stereobm_depth(left_path, right_path):
    """Complete StereoBM depth estimation pipeline"""
    
    # 1. Load images
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    
    # 2. Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    # 3. Initialize StereoBM with parameters
    stereo = cv2.StereoBM_create(
        numDisparities=64,    # Number of disparity levels
        blockSize=21        # Matching window size
    )
    
    # Optional: Set additional parameters
    stereo.setPreFilterType(1)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(63)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleRange(2)
    stereo.setSpeckleWindowSize(50)
    
    # 4. Compute disparity
    disparity = stereo.compute(gray_left, gray_right)
    
    # 5. Post-processing
    # Convert to float32 and scale
    disparity = disparity.astype(np.float32) / 16.0
    
    # Remove invalid values (negative or zero disparities)
    disparity[disparity <= 0] = 0.1
    
    return disparity, left, right

# Usage
disparity_map, left_img, right_img = compute_stereobm_depth("temp/undistorted_rectified_image_l.png", "temp/undistorted_rectified_image_r.png")

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
plt.title("Left Image"), plt.axis('off')
plt.subplot(132), plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
plt.title("Right Image"), plt.axis('off')
plt.subplot(133), plt.imshow(disparity_map, cmap='jet')
plt.title("Disparity Map"), plt.axis('off'), plt.colorbar()
plt.show()