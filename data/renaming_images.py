import os

# Define the root dataset directory
root_path = r"C:/Users/dahan/PycharmProjects/insect_classification/data/insects_dataset/"


def rename_files_in_folder(folder_path, category):
    """Renames all image files in a folder while skipping existing names."""
    existing_files = set(os.listdir(folder_path))  # Store existing filenames for quick lookup

    for i, entry in enumerate(sorted(os.scandir(folder_path), key=lambda e: e.name)):
        if entry.is_file():  # Ensure it's a file
            file_ext = os.path.splitext(entry.name)[1]  # Extract extension
            new_name = f"{category}_{i}{file_ext}"
            new_path = os.path.join(folder_path, new_name)

            if new_name in existing_files:
                print(f"Skipping: {entry.name}, {new_name} already exists.")
                continue  # Skip if the new name already exists

            os.rename(entry.path, new_path)
            print(f"Renamed: {entry.name} -> {new_name}")


def process_dataset(root_path):
    """Loops through dataset folders and renames images efficiently."""
    for category_entry in os.scandir(root_path):
        if category_entry.is_dir():  # Only process directories
            print(f"Processing folder: {category_entry.name}")
            rename_files_in_folder(category_entry.path, category_entry.name)
            print(f"Completed: {category_entry.name}\n")

    print("All folders processed.")


# Run the script
process_dataset(root_path)

