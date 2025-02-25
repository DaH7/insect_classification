import os
from PIL import Image

def resize_images(directory, size=(128, 128)):
    """
    Resizes all images in a directory to a specified size.

    Args:
        directory (str): The path to the directory containing the images.
        size (tuple): The desired size (width, height) for the images.
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                img = Image.open(filepath)
                img = img.resize(size)
                img.save(filepath)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    image_directory = "C:/Users/dahan/PycharmProjects/insect_classification/data/insects_dataset/all_insects"
    resize_images(image_directory)
    print("Images resized successfully.")