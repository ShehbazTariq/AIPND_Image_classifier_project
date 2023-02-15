import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from collections import OrderedDict
import seaborn as sns
import os

def load_data(path,batch_size):
    print("Loading data from the directory {} ...".format(path))
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    data_transforms = transforms.Compose([transforms.RandomRotation(10),
                                    transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=data_transforms)


    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    print("Finished data loading")
    return train_data, train_dl, test_dl , val_dl

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image) 
    
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.RandomRotation(10),
                                    transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])


    return transform(img)
   

def build_model(arch, hidden):
    print(" Architecture: {}, hidden_units: {}".format(arch, hidden))
    try:
        mdl = getattr(models, arch)(pretrained=True)
    except:
        print('Model not found') 
    
    for param in mdl.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 4096)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(4096, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    mdl.classifier = classifier
    print("Model built.")
    
    return mdl

def train_nn(model,epochs,learning_rate,train_dl,val_dl):
    print("Training network ... epochs: {}, learning_rate: {}".format(epochs, learning_rate))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    model.to(device)
          
          
    # Training the network
    steps = 0
    print_every = 10
    train_loss = 0
          
    for epoch in range(epochs):
        for inputs, labels in train_dl:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in val_dl:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logits = model.forward(inputs)
                        batch_loss = criterion(logits, labels)
                        valid_loss += batch_loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(val_dl):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(val_dl):.3f}")

                train_loss = 0

                model.train()

    print("Finished training network.")            
    
    return model, criterion
           

          
def evaluate_model(model, testloader, criterion):
    print("Testing network ... gpu used for testing: {}".format(gpu))
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    
    # Validation on the test set
    test_loss = 0
    test_accuracy = 0
    model.eval() 
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model.forward(inputs)
            batch_loss = criterion(logits, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy of test set
            ps = torch.exp(logits)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(test_dl):.3f}, "
          f"Test accuracy: {test_accuracy/len(test_dl):.3f}")
    running_loss = 0
    
    print("Finished testing.")
    
def save_model(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    print("Saving model ... epochs: {}, learning_rate: {}, save_dir: {}".format(epochs, learning_rate, save_dir))
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    checkpoint_path = save_dir + "checkpoint.pth"

    torch.save(checkpoint, checkpoint_path)
    
    print("Model saved to {}".format(checkpoint_path))
    
def load_model(filepath):
    print("Loading and building model from {}".format(filepath))

    checkpoint = torch.load(filepath)
    model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model
          
          
def predict(processed_image, model, topk): 
    model.eval()
    with torch.no_grad():
        logits = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logits)
        probs, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probs.numpy()[0], classes
