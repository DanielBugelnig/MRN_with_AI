import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import copy

batch_size = 5 # with 6 --> 70.8%
# Define transforms
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match pretrained model input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize as per pretrained models
])

#transform recommendations from ChatGPT when handling with small datasets
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='ycbv_classification/train', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the validation set
val_dataset = datasets.ImageFolder(root='ycbv_classification/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = resnet50(weights=ResNet50_Weights.DEFAULT)  # resnet50, freeze all parameters
model = torch.load('res_model_92.pth', weights_only=False)  #load best model
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters(): # set the last conv block and the fc to true
    if 'fc' in name or 'layer4' in name or 'layer3' in name or 'layer2' in name:
        param.requires_grad = True 

num_classes = 3
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  #modify last layer to three output





# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader) 
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practise
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        # Compute prediction and loss
        #print(f"Shape of image: {X.shape}\n")  # Check shape to confirm it's 4D
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # performs backpropagation
        optimizer.step() # adjusts parameters
        optimizer.zero_grad() # reset gradient

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

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
    return correct, test_loss


#Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
epochs = 6 
learning_rate = 0.00005
weight_decay = 1e-4
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1/1.05)


# Initialize variables to track the best validation accuracy and corresponding model weights
best_val_accuracy = 0.0
best_model_weights = None
counter = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    val_accuracy, val_loss = test_loop(val_loader, model, loss_fn)
     # Check if validation accuracy has improved
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = copy.deepcopy(model.state_dict())  # Save current best model weights
        print(f"New best validation accuracy: {100*val_accuracy:.3f}% - saving model weights")
        counter=0
    if val_accuracy == 1:
        break
    elif (val_accuracy <= best_val_accuracy):
        counter += 1
        print(f"erhöhe counter: {counter}\n")
    if(counter >= 3):
        print(f"Validation accuracy dropped to {100*val_accuracy:.3f}%. Reverting to best model weights.")
        model.load_state_dict(best_model_weights)  # Revert to best weights if accuracy drops
        counter = 0
        scheduler.step(val_loss)
    print(counter)
    # Optional: Adjust learning rate based on validation loss
    
model.load_state_dict(best_model_weights)  # Revert to best weights if accuracy drops
print("Done!")

torch.save(model, 'res_model.pth')
print(f"Model trained with hp: epochs: {epochs}, learning_rate{learning_rate}, batch_size: {batch_size}\n, optimizer: Adam")
#model = torch.load('model.pth', weights_only=False),



#trained a 92% accurate model on layer 2, 3, 4, and ff again, with parameters: epochs 6, learning_rate 5e-5, batch_size 5 to atain 100% on val set -> stored under res_model_100.pth