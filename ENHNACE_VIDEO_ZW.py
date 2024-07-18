import numpy as np
import cv2
from skimage import io
import tifffile as tiff

# Load the multi-frame TIFF image
image_path = 'aligned_2_agar_sample2_cell2.tif'  # Replace with the actual path
video = io.imread(image_path)

# Initialize lists to hold processed frames
processed_frames = []

# Process each frame with bilateral filter and CLAHE
for frame in video:
    # Convert the frame to 8-bit
    frame_uint8 = (255 * (frame / np.max(frame))).astype(np.uint8)
    
    # Apply bilateral filtering
    bilateral_filtered_frame = cv2.bilateralFilter(frame_uint8, d=3, sigmaColor=75, sigmaSpace=75)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    clahe_frame = clahe.apply(bilateral_filtered_frame)
    
    # Append the processed frame to the list
    processed_frames.append(clahe_frame)

# Convert the list to a numpy array
processed_video = np.array(processed_frames)

# Bin every 10 frames by calculating their average
binned_frames = []
num_frames = processed_video.shape[0]
for i in range(0, num_frames, 10):
    # Calculate the average of the next 10 frames
    binned_frame = np.mean(processed_video[i:i+10], axis=0)
    binned_frames.append(binned_frame)

# Convert the list to a numpy array
binned_video = np.array(binned_frames)

# Save the binned video as a multi-frame TIFF file
output_tif_path = 'binned_processed_video.tif'  # Replace with the desired save path
tiff.imwrite(output_tif_path, binned_video.astype(np.uint8), photometric='minisblack')

# Print the save location
print(f"Video saved to {output_tif_path}")
