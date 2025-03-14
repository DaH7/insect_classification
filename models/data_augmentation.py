import random
import cv2
import albumentations as A
import os
import shutil
import csv

root_path = "C:/Users/dahan/PycharmProjects/insect_classification/models/all_insects"

# mapping of labels
label_mapping = {
    "Butterfly": 0,
    "Dragonfly": 1,
    "Grasshopper": 2,
    "Ladybug": 3,
    "Mosquito": 4
}


def get_label_from_filename(filename):
    """Extracts the label number from the filename based on keywords."""
    for insect, label in label_mapping.items():
        if insect.lower() in filename.lower():  # case sensative check
            return label
    return None


def augment_images_in_folder(input_folder, num_augmentations=5):

    output_dir = os.path.join("C:/Users/dahan/PycharmProjects/insect_classification/models" , "augmented_images")
    # checks if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # csv file path
    csv_filename = "augmented_images.csv"
    csv_path = os.path.join("C:/Users/dahan/PycharmProjects/insect_classification/models", csv_filename)

    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Filename", "Label"])  # header

        # gets list of images
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.bmp','.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("No images found in the folder.")
            return

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)

            # loads image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # extracts base name of the image without extention
            base_name, ext = os.path.splitext(image_file)

            # gets the label for the image
            label = get_label_from_filename(base_name)
            if label is None:
                print(f"Label not found for {image_file}. Skipping...")
                continue

            # save the original image
            original_filename = f"{base_name}_original{ext}"
            original_filepath = os.path.join(output_dir, original_filename)
            shutil.copy(image_path, original_filepath)
            csv_writer.writerow([original_filename, label])
            print(f"Saved original image: {original_filepath} with label {label}")

            # list of possible augmentations
            transformations = [
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=30, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.6),
                A.GaussianBlur(blur_limit=(3, 7),p=0.4),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,p=0.3),
                A.ColorJitter(brightness=0.5),
                A.CoarseDropout(max_holes=3, max_height=16, max_width=16, min_holes=1, p=0.5)


            ]

            # apply augmentations
            for i in range(num_augmentations):
                # two to four random transformations
                selected_transforms = random.sample(transformations, random.randint(2, 4))

                #transform images
                transform = A.Compose(selected_transforms)
                augmented = transform(image=image)

                # ensures transformation is applied correctly
                if 'image' not in augmented or augmented['image'] is None:
                    print(f"Augmentation failed for {image_file} with {selected_transforms}")
                    continue

                # create filename for the augmented image
                transform_names = "_".join([t.__class__.__name__ for t in selected_transforms])
                augmented_filename = f"{base_name}_augmented_{i + 1}_{transform_names}{ext}"
                augmented_filepath = os.path.join(output_dir, augmented_filename)

                # save augmented image
                cv2.imwrite(augmented_filepath, augmented['image'])
                csv_writer.writerow([augmented_filename, label])
                print(f"Saved augmented image: {augmented_filepath} with label {label}")

    print(f"CSV file saved at: {csv_path}")

# Example usage: provide the path to your folder of images
augment_images_in_folder(root_path, num_augmentations=5)

