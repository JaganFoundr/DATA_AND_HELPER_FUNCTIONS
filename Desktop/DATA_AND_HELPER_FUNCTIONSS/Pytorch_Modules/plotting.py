#4. plotting random images from the whole dataset.
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transform

from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transform

# Global variable to store the random image path
random_image_path_global = None

def plot_raw_random(main_path: str):
    """
    # Plotting non-transformed (raw) random images from the whole dataset.
    """
    global random_image_path_global  # Access the global variable
    image_path = Path(main_path)
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path_global = random.choice(image_path_list)

    image_label = random_image_path_global.parent.name
    image = Image.open(random_image_path_global)
    image_as_an_array = np.asarray(image)

    # Plot the raw image
    plt.imshow(image)
    plt.title(f"Raw Image: {image_label}")
    plt.axis(False)
    plt.show()

    print(f"\nImage class: {image_label}")
    print(f"Image height: {image.height}")
    print(f"Image width: {image.width}")
    print(f"Image shape: {image_as_an_array.shape}")
    print(f"Image data-type: {image_as_an_array.dtype}\n")


def plot_transformed_random(main_path:str,transform):
    """
    # Plotting transformed images using the stored random image path.
    """
    global random_image_path_global  # Access the global variable

    image_label = random_image_path_global.parent.name
    image = Image.open(random_image_path_global)

    # Apply the transformation
    transformed_image = transform(image)
    color_image = transformed_image.permute(1, 2, 0)

    # Plot the transformed image
    plt.imshow(color_image)
    plt.title(f"Transformed Image: {image_label}")
    plt.axis(False)
    plt.show()

    print(f"\nImage class: {image_label}")
    print(f"Image shape: {transformed_image.shape}")
    print(f"Image data type: {transformed_image.dtype}")
