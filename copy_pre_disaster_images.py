import os
import shutil

# Original datasets folder path (example)
datasets_dir = './xbd'  # Modify to your actual path
# Destination folder path (parent directory where 'datasets' will be copied)
dest_dir = '.'  # Specify the current directory or desired parent path
include_tier3 = False # Set to True to include tier3 data in the 'train' split

def create_dataset_structure(base_path='.', dataset_folder_name='datasets'):
    """
    Creates the 'datasets' folder and its subdirectories ('hold', 'test', 'train'
    with 'images' and 'targets' inside each) under the given base_path.

    Parameters:
    -----------
    base_path: str
        The base path where the folder structure will be created (e.g., current directory '.').
    dataset_folder_name: str
        The name of the main dataset folder to be created (default: 'datasets').
    """
    # Define the folder structure to be created
    subfolders = [
        "hold/images",
        "hold/targets",
        "test/images",
        "test/targets",
        "train/images",
        "train/targets",
    ]

    for folder in subfolders:
        target_path = os.path.join(base_path, dataset_folder_name, folder)
        os.makedirs(target_path, exist_ok=True)
        print(f"Created: {target_path}")

# Create the "datasets" folder within the destination directory if it doesn't exist
if not os.path.exists(os.path.join(dest_dir, "datasets")):
    create_dataset_structure(dest_dir)

def copy_pre_disaster_images(datasets_dir, dest_dir, include_tier3=False):
    """
    Copies 'pre_disaster' images from the source datasets_dir to the
    'datasets' folder within the dest_dir.
    The 'include_tier3' parameter determines whether to include files from
    the 'tier3' folder in the 'train' data.

    Parameters:
    -----------
    datasets_dir: str
        Path to the original dataset folder (e.g., './xbd').
    dest_dir: str
        The parent directory for the destination 'datasets' folder
        (e.g., '.' means the created folder will be './datasets').
    include_tier3: bool
        If True, files from the 'tier3' folder will be copied to the 'train' data.
        If False, the 'tier3' folder will be ignored.
    """
    # Iterate through 'hold', 'test', and 'train' splits
    for split in ['hold', 'test', 'train']:
        source_images_dir = os.path.join(datasets_dir, split, 'images')
        source_targets_dir = os.path.join(datasets_dir, split, 'targets')

        if os.path.exists(source_images_dir):
            for filename in os.listdir(source_images_dir):
                if filename.endswith("pre_disaster.png"):
                    src_image_path = os.path.join(source_images_dir, filename)

                    # Construct destination paths
                    dest_image_path = os.path.join(dest_dir, 'datasets', split, 'images', filename)
                    # Replace '_pre_disaster.png' with '_target.png' for the label file
                    target_filename = filename.replace('_pre_disaster.png', '_target.png')
                    src_target_path = os.path.join(source_targets_dir, target_filename)
                    dest_target_path = os.path.join(dest_dir, 'datasets', split, 'targets', target_filename)

                    # Copy image and its corresponding label
                    shutil.copy2(src_image_path, dest_image_path)
                    shutil.copy2(src_target_path, dest_target_path)

                    print(f"[{split}] Copied {filename}.")
        else:
            print(f"Source folder does not exist: {source_images_dir}")

    # Process tier3 folder: copy to 'train' data if include_tier3 is True
    if include_tier3:
        tier3_images_dir = os.path.join(datasets_dir, 'tier3', 'images')
        tier3_targets_dir = os.path.join(datasets_dir, 'tier3', 'targets')

        if os.path.exists(tier3_images_dir):
            for filename in os.listdir(tier3_images_dir):
                if filename.endswith("pre_disaster.png"):
                    src_image_path = os.path.join(tier3_images_dir, filename)

                    # Destination for tier3 images and labels is always 'train'
                    dest_image_path = os.path.join(dest_dir, 'datasets', 'train', 'images', filename)
                    # Replace '_pre_disaster.png' with '_target.png' for the label file
                    target_filename = filename.replace('_pre_disaster.png', '_target.png')
                    src_target_path = os.path.join(tier3_targets_dir, target_filename)
                    dest_target_path = os.path.join(dest_dir, 'datasets', 'train', 'targets', target_filename)

                    # Copy image and its corresponding label
                    shutil.copy2(src_image_path, dest_image_path)
                    shutil.copy2(src_target_path, dest_target_path)

                    print(f"[tier3 -> train] Copied {filename}.")
        else:
            print(f"Source folder does not exist: {tier3_images_dir}")

# Example usage
if __name__ == "__main__":
    # Set include_tier3 to True to incorporate tier3 data into the 'train' split.
    copy_pre_disaster_images(
        datasets_dir=datasets_dir,
        dest_dir=dest_dir,
        include_tier3=include_tier3
    )