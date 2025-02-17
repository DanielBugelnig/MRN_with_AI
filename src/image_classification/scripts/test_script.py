import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
# Script to test the neural network visually

# Define transformations for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Same normalization as training
])

# Function to denormalize the images (optional)
def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std[:, None, None] + mean[:, None, None]  # Undo normalization
    return image

# Load the validation set
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=7,shuffle=False)


print(f'Len: {len(val_dataset)}\n')
labels_map = {
    0: "Cheezit",
    1: "Dominos",
    2: "Frenchis"
}

# Get a batch of images from the DataLoader
data_iter = iter(val_loader)
images, labels = next(data_iter)


# selectin device and loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('results/res_model_new2.pth', weights_only=False)
model = model.to(device)



# Prediction function for a single image
def predict_single(image, model):
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        pred = model(image)
        return pred.argmax(1).item()
    return predicted_class

def test_loop(dataloader, model, loss_fn):
    # set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    # Evaluation with torch_no_grad() ensures that no gradients are computed during test mode
    # also serves tp reduce unnecessary gradient computations and memory usage
    # for tensors with requires_grad = True

    with torch.no_grad():
        for X,y, in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')

        

def show_sample_predictions(model, grid):
    row, col = grid, grid
    fig = plt.figure(figsize=(10, 10))

    for i in range(grid**2):
        rand_idx = torch.randint(len(val_dataset), size=(1,)).item() 
        img, groundtruth = val_dataset[rand_idx]

        # Predict the label
        predicted_label = predict_single(img, model)

        # Denormalize and prepare for display
        img_display = denormalize(img).permute(1, 2, 0).numpy()  # Convert to HxWxC format

        # Plot each image in a grid layout
        ax = fig.add_subplot(row, col, i + 1)  # `i + 1` to start subplot index from 1
        ax.set_title(f"True: {labels_map[groundtruth]}, Pred: {labels_map[predicted_label]}")
        ax.axis("off")
        ax.imshow(img_display)

    plt.show()

show_sample_predictions(model, 4)
loss_fn = nn.CrossEntropyLoss()
test_loop(val_loader, model, loss_fn)

