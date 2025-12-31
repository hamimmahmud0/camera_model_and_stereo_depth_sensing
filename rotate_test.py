import os
from PIL import Image
from pathlib import Path

def rotate_images_180(source_dir):
    """
    Rotate all images matching pattern 'r_test_*.jpg' by 180 degrees.
    
    Args:
        source_dir: Path to directory containing images
    """
    # Convert to Path object for better path handling
    source_path = Path(source_dir)
    
    # Check if source directory exists
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' does not exist.")
        return
    
    # Find all matching files
    image_files = list(source_path.glob('l_test_*.jpg'))
    
    if not image_files:
        print(f"No files matching 'l_test_*.jpg' found in '{source_dir}'")
        return
    
    print(f"Found {len(image_files)} image(s) to rotate:")
    
    # Process each image
    for image_file in image_files:
        try:
            # Open the image
            with Image.open(image_file) as img:
                # Rotate by 180 degrees
                rotated_img = img.rotate(180)
                
                # Save the image (overwrites the original)
                rotated_img.save(image_file)
                
                print(f"✓ Rotated: {image_file.name}")
                
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {e}")

if __name__ == "__main__":
    # Define the directory path
    image_dir = "./images/test"
    
    # Rotate the images
    rotate_images_180(image_dir)
    
    print("\nRotation complete!")
