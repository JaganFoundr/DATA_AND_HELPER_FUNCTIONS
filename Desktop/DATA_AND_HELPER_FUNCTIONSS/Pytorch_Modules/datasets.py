import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


##########################################################

def create_dataset(train_folder:str,
                   test_folder:str,
                   train_transform:T.Compose,
                   test_transform:T.Compose,
                   target_train_transform:None,
                   target_test_transform:None):
    
    train_data=ImageFolder(root=train_folder, transform=train_transform, target_transform=target_train_transform)
    test_data=ImageFolder(root=test_folder, transform=test_transform, target_transform=target_test_transform)

    return train_data, test_data


##############################################################

def Dataloader(train_dataset,
               test_dataset,
               batch_size:int,
               num_workers:int,
               train_shuffle:bool,
               test_shuffle:bool):
    train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=train_shuffle)
    test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=test_shuffle)

    return train_loader, test_loader


##################################################################

import os
import shutil
from sklearn.model_selection import train_test_split

def traintest_split(input_dir, output_dir, train_split=0.75):
    """
    Organize a flat dataset into a structured format with train/test folders and class subfolders.

    Args:
    - input_dir (str): Path to the folder containing all images.
    - output_dir (str): Path where the organized dataset will be stored.
    - train_split (float): Ratio of training data (default is 0.8).
    """
    # Ensure output directories exist
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read class names from folder structure or infer from filenames
    classes = list(set([fname.split('_')[0] for fname in os.listdir(input_dir)]))  # Assumes 'className_xxx.jpg'

    for class_name in classes:
        # Create class-specific directories for train and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Gather all images of this class
        class_images = [f for f in os.listdir(input_dir) if f.startswith(class_name)]
        train_images, test_images = train_test_split(class_images, train_size=train_split)

        # Move images to respective folders
        for image in train_images:
            shutil.move(os.path.join(input_dir, image), os.path.join(train_dir, class_name, image))
        for image in test_images:
            shutil.move(os.path.join(input_dir, image), os.path.join(test_dir, class_name, image))

    print(f"Dataset organized at {output_dir}")
