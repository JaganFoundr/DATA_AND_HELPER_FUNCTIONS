#function for re-arranging the whole dataset inside one folder to a structure of 2 folder(train, test) under the parent folder.
import os
import shutil

def folder_format(base_dir:str,train_file:str, test_file:str, train_dir:str, test_dir:str):
    # Paths
    '''base_dir = "data/food-101/images"
    train_file = "data/food-101/meta/train.txt"
    test_file = "data/food-101/meta/test.txt"

    # Create train and test directories
    train_dir = "data/food-101/train"
    test_dir = "data/food-101/test"'''

    for dir_path in [train_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Helper function to move files
    def move_files(file_list, dest_dir):
        with open(file_list, "r") as f:
            for line in f:
                relative_path = line.strip()  # e.g., class_1/image_1
                class_name = relative_path.split("/")[0]  # Extract class name
                # Create class directory if it doesn't exist
                class_dir = os.path.join(dest_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                # Move the file
                src = os.path.join(base_dir, relative_path + ".jpg")
                dest = os.path.join(class_dir, os.path.basename(src))
                shutil.move(src, dest)

    # Move training images
    move_files(train_file, train_dir)

    # Move testing images
    move_files(test_file, test_dir)
