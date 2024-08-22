#load librarires
import numpy as np


import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import matplotlib.pyplot as plt
import pathlib

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

#transforms
transformer=transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5],  #0-1 to [-1,1], formula x-mean/std
                         [0.5,0.5,0.5])    
])

#Dataloader

#Path to directories
train_path="C:/Users/HP/Desktop/Extras/PythonML/CatClassifier/dataset/train"
test_path="C:/Users/HP/Desktop/Extras/PythonML/CatClassifier/dataset/test"
train_loader=DataLoader(torchvision.datasets.ImageFolder(train_path,transform=transformer),batch_size=256, shuffle = True)
test_loader=DataLoader(torchvision.datasets.ImageFolder(test_path,transform=transformer),batch_size=256, shuffle = True)

 #categoires
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

#CNN network
class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet,self).__init__()

        #OUTPUT SIZE AFTER CONVOLUTION FILETER
        # ((w-f+2P)/s)+1

        #input shape =(256,3,150,150)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #shape=(256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #shape=(256,12,150,150)
        self.relu1=nn.ReLU()
        #shape=(256,12,150,150)

        self.pool=nn.MaxPool2d(kernel_size=2)
        #reduce the image size be factor 2
        #shape=(256,12,75,75)

        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #shape=(256,20,75,75)
        self.relu2=nn.ReLU()
        #shape=(256,22,75,75)

        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #shape=(256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #shape=(256,32,75,75)
        self.relu3=nn.ReLU()
        #shape=(256,32,75,75)

        self.fc=nn.Linear(in_features=32*75*75,out_features=num_classes)

    #feed forward function
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        
        output=self.pool(output)
        output=self.conv2(output)
        output=self.relu2(output)
        
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        #above output will be in matrix form with shape(256,32,75,75)
        output=output.view(-1,32*75*75)

        output=self.fc(output)

        return output
    
model=ConvNet(num_classes=2).to(device)

#optimizer and loss function
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

num_epochs=20

#calculation the size af training and testing images
train_count=len(glob.glob(train_path+'/**/*'))
test_count=len(glob.glob(test_path+'/**/*'))

# Initialize lists to store training and test metrics
train_losses = []
test_accuracies = []
train_accuracies = []

#model training and savin best model

best_accuracy=0
for epoch in range(num_epochs):
    #evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i,(images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=images.to(device)
            labels=labels.to(device)

        
        optimizer.zero_grad()
        outputs=model(images)

        

        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.cpu()*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))

    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    # Store train loss and accuracy
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    #evaluaiton on testinf datasert
    model.eval()
    test_accuracy=0.0
    print("testing started")
    for i,(images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=images.to(device)
            labels=labels.to(device)

        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy=test_accuracy/test_count
    test_accuracies.append(test_accuracy)

    print('Epoch: '+str(epoch)+' Train Loss: '+str(int(train_loss))+' Train Accuracy: '+ str(train_accuracy)+' Test accuracy: '+str(test_accuracy))

    #save the best model
    if test_accuracy>best_accuracy:
        print("Best model saved")
        torch.save(model.state_dict(),'best_checkpoint.model')
        best_accuracy=test_accuracy 
        
# Plotting the results
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))
train_losses_numpy = [loss.detach().numpy() for loss in train_losses]
# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_numpy, 'g', label='Training loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
plt.plot(epochs, test_accuracies, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()