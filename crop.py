import argparse
from PIL import Image
from copy_pre_disaster_images import create_dataset_structure
import os

parser = argparse.ArgumentParser(description='create a 512x512 xBD building datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--datasets_dir', default='./datasets', type=str, dest='datasets_dir')
parser.add_argument('--save_dir', default='./datasets_512', type=str, dest='save_dir')

args = parser.parse_args()

def save_quarter_crops(path_image, path_label, dir_save, fname_image):
    """
    Performs a 4-way crop on the given original and label images and saves them
    to the specified directory.
    
    Parameters:
    -----------
    path_image : str
        Path to the original image file.
    path_label : str
        Path to the label image file.
    dir_save : str
        Directory where the cropped images and labels will be saved
        (expects 'images' and 'targets' subfolders).
    fname_image : str
        Prefix for the saved files (e.g., "sample" from "sample_pre_disaster.png").
    """
    # 1. Open original and label images
    img_input = Image.open(path_image)
    img_label = Image.open(path_label)

    # 2. Get (width, height)
    width_in, height_in = img_input.size
    width_lb, height_lb = img_label.size

    # 3. Define crop regions for each quadrant
    # Top-left (1)
    img_input_1 = img_input.crop((0, 0, width_in // 2, height_in // 2))
    img_label_1 = img_label.crop((0, 0, width_lb // 2, height_lb // 2))

    # Top-right (2)
    img_input_2 = img_input.crop((width_in // 2, 0, width_in, height_in // 2))
    img_label_2 = img_label.crop((width_lb // 2, 0, width_lb, height_lb // 2))

    # Bottom-left (3)
    img_input_3 = img_input.crop((0, height_in // 2, width_in // 2, height_in))
    img_label_3 = img_label.crop((0, height_lb // 2, width_lb // 2, height_lb))

    # Bottom-right (4)
    img_input_4 = img_input.crop((width_in // 2, height_in // 2, width_in, height_in))
    img_label_4 = img_label.crop((width_lb // 2, height_lb // 2, width_lb, height_lb))

    # 4. Save as PNG (to dir_save/images and dir_save/targets)
    # Top-left (1)
    img_input_1.save(os.path.join(dir_save, "images", f"{fname_image}_1_pre_disaster.png"))
    img_label_1.save(os.path.join(dir_save, "targets", f"{fname_image}_1_pre_disaster_target.png"))
    
    # Top-right (2)
    img_input_2.save(os.path.join(dir_save, "images", f"{fname_image}_2_pre_disaster.png"))
    img_label_2.save(os.path.join(dir_save, "targets", f"{fname_image}_2_pre_disaster_target.png"))
    
    # Bottom-left (3)
    img_input_3.save(os.path.join(dir_save, "images", f"{fname_image}_3_pre_disaster.png"))
    img_label_3.save(os.path.join(dir_save, "targets", f"{fname_image}_3_pre_disaster_target.png"))
    
    # Bottom-right (4)
    img_input_4.save(os.path.join(dir_save, "images", f"{fname_image}_4_pre_disaster.png"))
    img_label_4.save(os.path.join(dir_save, "targets", f"{fname_image}_4_pre_disaster_target.png"))
    
    print(f"Cropping of {fname_image} files completed")

def process_dataset_crops(datasets_dir, save_dir):
    """
    Performs 4-way cropping on 'pre_disaster' images found within the 'hold', 'test',
    and 'train' folders of the given datasets directory and saves them to save_dir.

    Parameters:
    -----------
    datasets_dir : str
        Path to the original dataset directory (e.g., './datasets').
    save_dir : str
        The final directory where the cropped images will be saved (e.g., './datasets_512').
    """
    # First, create the folder structure in the save directory
    # (using create_dataset_structure from an external module)
    # The create_dataset_structure function accepts 'fname_folder' to create the necessary folders.
    create_dataset_structure(fname_folder=save_dir)
    
    # Iterate through 'hold', 'test', 'train' folders
    for split in ['hold', 'test', 'train']:
        source_dir = os.path.join(datasets_dir, split, 'images')
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                if filename.endswith("pre_disaster.png"):
                    src_path = os.path.join(source_dir, filename)
                    
                    # Generate label image path
                    # e.g., "sample_pre_disaster.png" -> "sample_pre_disaster_target.png"
                    target_src_path = os.path.join(
                        datasets_dir, split, 'targets',
                        filename.split('.png')[0] + '_target.png'
                    )
                    
                    # Destination path: save_dir/split (expects 'images' and 'targets' subfolders)
                    dir_path = os.path.join(save_dir, split)
                    
                    # Extract file prefix (e.g., "sample")
                    fname_image = filename.split('_pre_disaster.png')[0]
                    
                    # Save quarter crops
                    save_quarter_crops(src_path, target_src_path, dir_path, fname_image)
        else:
            print(f"Folder does not exist: {source_dir}")

# Example usage
if __name__ == "__main__":
    datasets_dir = args.datasets_dir    # Path to original datasets
    save_dir = args.save_dir     # Path to save cropped datasets
    
    process_dataset_crops(datasets_dir, save_dir)