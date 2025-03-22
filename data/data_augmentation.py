import os
import cv2
import shutil
import random
import albumentations as A
from PIL import Image

root_path = "C:/Users/dahan/PycharmProjects/insect_classification/data/insect_classes_dataset_resized"
output_base_dir = "C:/Users/dahan/PycharmProjects/insect_classification/data/augmented_data"


def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # check for corruption
        return True
    except Exception as e:
        print(f"Corrupt image detected: {image_path} ({e})")
        return False


def clear_output_directory(output_base_dir):
    if os.path.exists(output_base_dir):
        # remove the existing output directory and all its contents
        shutil.rmtree(output_base_dir)
        print(f"Cleared existing data in {output_base_dir}")
    # Recreate the empty output directory
    os.makedirs(output_base_dir)
    print(f"Created a new empty directory: {output_base_dir}")


def augment_images_in_folders(input_folder, num_augmentations=5, output_base_dir="saved_augmented_images"):
    # clear the output directory (if needed)
    clear_output_directory(output_base_dir)

    for label in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label)

        if not os.path.isdir(label_path):  # skip non-folder files
            continue

        # create label-specific output directory inside output_base_dir
        output_label_dir = os.path.join(output_base_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        # get the list of images for the label folder
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No images found in {label_path}. Skipping...")
            continue

        for image_file in image_files:
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            base_name, ext = os.path.splitext(image_file)
            original_filepath = os.path.join(output_label_dir, f"{base_name}_original{ext}")
            shutil.copy(image_path, original_filepath)  # keep the original
            print(f"Saved original: {original_filepath}")

            transformations = [
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=30, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.6),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.ColorJitter(brightness=0.5),
                A.CoarseDropout(max_holes=3, max_height=16, max_width=16, min_holes=1, p=0.5)
            ]

            for i in range(num_augmentations):
                selected_transforms = random.sample(transformations, random.randint(2, 4))
                transform = A.Compose(selected_transforms)
                augmented = transform(image=image)

                if 'image' not in augmented or augmented['image'] is None:
                    print(f"Augmentation failed for {image_file} with {selected_transforms}")
                    continue

                transform_names = "_".join([t.__class__.__name__ for t in selected_transforms])
                augmented_filename = f"{base_name}_aug_{i + 1}_{transform_names}{ext}"
                augmented_filepath = os.path.join(output_label_dir, augmented_filename)

                # save the augmented image with a higher quality setting
                if not cv2.imwrite(augmented_filepath, augmented['image'], [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    print(f"Failed to save augmented image: {augmented_filepath}")
                    continue

                # validate the augmented image after saving
                if not is_valid_image(augmented_filepath):
                    print(f"Augmented image is corrupted: {augmented_filepath}")
                    continue

                print(f"Saved augmented image: {augmented_filepath}")

    print("Augmentation completed.")


augment_images_in_folders(root_path, num_augmentations=3,output_base_dir=output_base_dir)
