import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import pathlib
from tkinter import filedialog, Tk


# Paths
train_path = "C:/Users/HP/Desktop/Extras/PythonML/CatClassifier/dataset/train"

# Categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CNN network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32*75*75)
        output = self.fc(output)

        return output

# Load model
checkpoint = torch.load('best_checkpoint.model')
model = ConvNet(num_classes=2)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Transforms
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Prediction function
def prediction(img, transformer):
    if isinstance(img, str):
        # If img is a path, load the image
        image = Image.open(img).convert('RGB')
    else:
        # If img is a PIL Image, use it directly
        image = img.covert('RGB')
    
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor = image_tensor.to(device)

    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()

    pred = classes[index]
    return pred


def select_image():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path

user_image_path = select_image()

if user_image_path:
    file_name = os.path.basename(user_image_path)
    prediction_result = prediction(user_image_path, transformer)
    print(f"\nFile: {file_name} \nPrediction: {prediction_result}")
else:
    print("No image selected.")
