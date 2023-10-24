import os
import numpy as np
from PIL import Image

CORR_MATRICES_IMG_DIR_PATH = "/home/onyr/code/phdtrack/memory_graph_gcn/logs/results/feature_corr"

def concatenate_images(images):
    # Create a blank white image to fill in when there are fewer than 6 images
    blank_img = Image.new('RGB', images[0].size, (255, 255, 255))
    
    # Prepare a 3x2 grid of images, filling in blanks if necessary
    while len(images) < 6:
        images.append(blank_img)
    
    # Concatenate images in the specified grid
    row1 = np.hstack((np.array(images[0]), np.array(images[1])))
    row2 = np.hstack((np.array(images[2]), np.array(images[3])))
    row3 = np.hstack((np.array(images[4]), np.array(images[5])))
    
    concatenated_img = np.vstack((row1, row2, row3))
    
    return Image.fromarray(concatenated_img)

def main():
    image_files = [f for f in os.listdir(CORR_MATRICES_IMG_DIR_PATH) if f.endswith('.png')]
    image_files.sort()  # Sort the files to ensure consistency
    
    chunk_size = 6
    for i in range(0, len(image_files), chunk_size):
        chunk = image_files[i:i+chunk_size]
        images = [Image.open(os.path.join(CORR_MATRICES_IMG_DIR_PATH, f)) for f in chunk]
        
        concatenated = concatenate_images(images)
        concatenated.save(os.path.join(CORR_MATRICES_IMG_DIR_PATH, f"concatenated_{i//chunk_size + 1}.png"))

if __name__ == "__main__":
    main()
