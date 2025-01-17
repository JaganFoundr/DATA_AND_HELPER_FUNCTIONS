import torch
import torchvision
from torchvision.datasets import ImageFolder 
import torchvision.transforms as T
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

import numpy as np

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import random
from PIL import Image
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter #tensorboard from tensorflow for experiment tracking

###########################################################################################

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############################################################################################

def accuracy(output, labels):
    '''# Accuracy Function'''
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100


#############################################################################################

# Loss Batch Function
def loss_batch(model, loss_function, images, labels, opt, metrics=accuracy):
    prediction = model(images)
    loss = loss_function(prediction, labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = metrics(prediction, labels) if metrics else None
    return loss.item(), len(images), metric_result


###############################################################################################

# Evaluation Function
def evaluate(model, loss_function, test_loader, metrics=accuracy):
    with torch.inference_mode():
        result = [loss_batch(model, loss_function, images.to(device), labels.to(device), opt=None, metrics=accuracy)
                  for images, labels in test_loader]

        losses, num, metric = zip(*result)
        total = np.sum(num)
        loss = np.sum(np.multiply(losses, num)) / total

        metric_result = np.sum(np.multiply(metric, num)) / total if metrics else None
        return loss, total, metric_result

##################################################################################################

def train_plot_tensorboard_multiple_experiments(experiment_configs, train_loaders, test_loaders, loss_function, metrics=accuracy):
    """
    Train multiple models with different configurations (model, optimizer, epochs) and log to TensorBoard for visualization.
    
    :param experiment_configs: List of dictionaries, each specifying a model, optimizer, epochs, and experiment name
    :param train_loaders: List of DataLoader for training data for each model
    :param test_loaders: List of DataLoader for test data for each model
    :param loss_function: Loss function (same for all models)
    :param metrics: Metric function (e.g., accuracy)
    """
    
    for exp_idx, config in enumerate(experiment_configs):
        model = config['model']
        optimizer = config['optimizer']
        exp_name = config['name']
        nepochs = config['epochs']  # Epochs are provided explicitly outside the function

        # Select the corresponding train_loader and test_loader for this experiment
        train_loader = train_loaders[exp_idx]  # Fetch correct loader for model
        test_loader = test_loaders[exp_idx]    # Fetch correct loader for model

        writer = SummaryWriter(f'runs/{exp_name}')  # Dynamic logging directory for each experiment

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        print(f"Training {exp_name}...")

        for epoch in tqdm(range(nepochs)):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                train_loss, _, train_acc = loss_batch(model, loss_function, images, labels, optimizer, metrics=metrics)

            model.eval()
            test_loss, _, test_acc = evaluate(model, loss_function, test_loader, metrics=metrics)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch + 1)
            writer.add_scalar('Loss/Test', test_loss, epoch + 1)
            writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
            writer.add_scalar('Accuracy/Test', test_acc, epoch + 1)

            print(f"Epoch {epoch + 1}/{nepochs}")
            print(f"Training loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
            print(f"Training accuracy: {train_acc:.2f}%, Test accuracy: {test_acc:.2f}%")
            print("---------------------------------------------------------\n")

        print(f"Average Training loss for {exp_name}: {sum(train_losses)/len(train_losses):.4f}")
        print(f"Average Test loss for {exp_name}: {sum(test_losses)/len(test_losses):.4f}")
        print(f"Average Training accuracy for {exp_name}: {sum(train_accuracies)/len(train_accuracies):.2f}%")
        print(f"Average Test accuracy for {exp_name}: {sum(test_accuracies)/len(test_accuracies):.2f}%")
        
        # Close the writer for this experiment
        writer.close()


'''Launch TensorBoard in the terminal:
bash
tensorboard --logdir=runs
Open a browser and go to http://localhost:6006 to view the visualizations.'''


###################################################################################################

def conf_matrix_for_train(model,image_path:str,train_loader):
    # Step 1: Fetch all class names
    main_path = Path(image_path)
    class_names = sorted([folder.name for folder in main_path.iterdir() if folder.is_dir()])
    num_classes = len(class_names)

    # Step 2: Initialize Confusion Matrix
    confusion_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    # Step 3: Run Inference and Collect Predictions
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    # Convert to tensors
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    conf_matrix_tensor = confusion_matrix(all_labels, all_preds)
    conf_matrix_np = conf_matrix_tensor.cpu().numpy()

    # Step 4: Plot Confusion Matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_matrix_np,
        class_names=class_names,
        figsize=(12, 12),
        cmap="Reds"
    )
    plt.title("Train data Confusion Matrix")
    plt.show()


########################################################################################

def conf_matrix_for_test(model,image_path:str,test_loader):
    # Step 1: Fetch all class names
    main_path = Path(image_path)
    class_names = sorted([folder.name for folder in main_path.iterdir() if folder.is_dir()])
    num_classes = len(class_names)

    # Step 2: Initialize Confusion Matrix
    confusion_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    # Step 3: Run Inference and Collect Predictions
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    # Convert to tensors
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute confusion matrix
    conf_matrix_tensor = confusion_matrix(all_labels, all_preds)
    conf_matrix_np = conf_matrix_tensor.cpu().numpy()

    # Step 4: Plot Confusion Matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_matrix_np,
        class_names=class_names,
        figsize=(12, 12),
        cmap="Reds"
    )
    plt.title("Test data Confusion Matrix")
    plt.show()


#########################################################################################

def prediction(images, model):
    input = images.to(device).unsqueeze(0)
    with torch.inference_mode():
      output = model(input)
    _, pred = torch.max(output, dim=1)
    return pred[0].item()

# Testing the Model

def test_prediction(class_names_parent_path:str,model,image_path:str):

    main_path = Path(class_names_parent_path)
    class_names = sorted([folder.name for folder in main_path.iterdir() if folder.is_dir()])

    main_path=Path(image_path)
    image_path_list=list(main_path.glob("*/*.jpg"))

    random_image_list=random.choice(image_path_list)
    image_label=random_image_list.parent.name

    image=Image.open(random_image_list) # This is the exact image code snippet.

    transforms = T.Compose([

        T.Resize(size=(224,224)),
        T.ToTensor()
    ])

    transformed_image=transforms(image)
    color_image=transformed_image.permute(1,2,0)

    plt.imshow(color_image)
    plt.title(f"{image_label}")
    plt.axis(False);
    plt.show()

    print("\nModel Prediction on Testset: ", class_names[prediction(transformed_image.to(device), model)])


    ###########################################################################################################

def train_prediction(class_names_parent_path:str,model,image_path:str):

    main_path = Path(class_names_parent_path)
    class_names = sorted([folder.name for folder in main_path.iterdir() if folder.is_dir()])

    main_path=Path(image_path)
    image_path_list=list(main_path.glob("*/*.jpg"))

    random_image_list=random.choice(image_path_list)
    image_label=random_image_list.parent.name

    image=Image.open(random_image_list) # This is the exact image code snippet.

    transforms = T.Compose([

        T.Resize(size=(224,224)),
        T.ToTensor()
    ])

    transformed_image=transforms(image)
    color_image=transformed_image.permute(1,2,0)

    plt.imshow(color_image)
    plt.title(f"{image_label}")
    plt.axis(False);
    plt.show()

    print("\nModel Prediction on Train set: ",class_names[prediction(transformed_image.to(device), model)])


    ########################################################################################################

    #23. testing the custom image

def custom_image_plot(class_names_parent_path:str,image_path:str,device,model):

    main_path = Path(class_names_parent_path)
    class_names = sorted([folder.name for folder in main_path.iterdir() if folder.is_dir()])

    custom_image = torchvision.io.read_image(image_path)

    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(custom_image.permute(1, 2, 0))
    plt.title("Original Image")
    plt.axis(False)
    plt.show()

    print("\n----------------------------------\n")

    # Define the transformation
    custom_transform = T.Compose([
        T.Resize(size=(224,224))# Resize to 64x64
    ]) # here we are not converting the image to tensor because torchvision.io converts jpg to tensor.

    # Apply the transformation
    transformed_image = custom_transform(custom_image)

    # Convert the transformed image to a format suitable for display
    transformed_image_after_permute = transformed_image.permute(1, 2, 0)  # Change dimensions for plt.imshow

    # Plot the transformed image
    plt.figure(figsize=(6, 6))
    plt.imshow(transformed_image_after_permute)
    plt.title("Transformed Image")
    plt.axis(False)
    plt.show()

    #24. plotting the custom image and model's prediction on that image.
    transformed_image=transformed_image.type(torch.float32)

    plt.imshow(transformed_image_after_permute)
    plt.title("Testing custom image from other source(Google)")
    plt.axis(False);
    plt.show()

    print("\nModel Prediction on custom image: ",class_names[prediction(transformed_image.to(device), model)])