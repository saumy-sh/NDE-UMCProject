import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchsummary import summary
from Trainer import ModelTrainer

BATCH_SIZE = 32
LR = 0.0001
N_EPOCHS = 10

dataset_name = "DB"

os.makedirs('results',exist_ok=True)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach experimentation/
ROOT_DIR = os.path.dirname(CURRENT_DIR)
print(f"Root Directory inside dataset.py: {ROOT_DIR}")
# Path to datasets/DB
DATASET_DIR = os.path.join(ROOT_DIR, "datasets", dataset_name)

print("Dataset root:", DATASET_DIR)


base_dir = DATASET_DIR
splits = os.listdir(base_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

file_path = os.path.join(base_dir,'training')
training_data = ImageFolder(file_path,transform=transform)

file_path = os.path.join(base_dir,'validation')
validation_data = ImageFolder(file_path,transform=transform)

file_path = os.path.join(base_dir,'testing')
testing_data = ImageFolder(file_path,transform=transform)
print(testing_data.class_to_idx)

training_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE,shuffle=True)
validation_dataloader = DataLoader(validation_data,batch_size=BATCH_SIZE,shuffle=True)
testing_dataloader = DataLoader(testing_data,batch_size=BATCH_SIZE,shuffle=True)



class CNN(nn.Module):
    def __init__(self,n_features):
        super(CNN,self).__init__()
        model = models.squeezenet1_0(pretrained=True)
        self.backend = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,n_features),
            nn.Softmax()
        )
    def forward(self,x):
        x = self.backend(x)
        x = self.classifier(x)
        return x

model = CNN(n_features=4).to(device)
summary(model,(3,224,224))


criterion = nn.CrossEntropyLoss().to(device)
optimiser = Adam(model.parameters(),lr=LR)


trainer = ModelTrainer(model, criterion, optimiser, device)
trainer.train(training_dataloader, validation_dataloader, N_EPOCHS)
trainer.plot_metrics()
trainer.validate(testing_dataloader,test=True)
trainer.save_model('squeezenet_fine-tuned.pth')
