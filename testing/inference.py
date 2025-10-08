import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_FEATURES = 1



class Crack(Dataset):
    def __init__(self,df,transform=None,device=device):
        self.df = df
        self.transform = transform
        self.labels = torch.tensor(df['label'],dtype=torch.float32).to(device)
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self,idx):
        label = round(self.labels[idx].item())
        img_path = self.df['img_path'][idx]
        # print(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(image).to(device)
        return img,label


class CNN(nn.Module):
    def __init__(self,n_features=N_FEATURES):
        super().__init__()
        self.n_features = n_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.autopool = nn.AdaptiveAvgPool2d((2,2))
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(128*2*2,256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.logit = nn.Linear(256,self.n_features)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.autopool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.logit(x)
        return x
    
model = CNN().to(device)
model.load_state_dict(torch.load('fine-tuned-weld.pth'),strict=False)

test_csv = './weld_test.csv'
test_df = pd.read_csv(test_csv)

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

test_dataset = Crack(test_df,test_transform)
model.eval()
acc = 0

with torch.no_grad():
    for i in range(len(test_df)):
        img_path = test_df['img_path'].iloc[i]
        label = test_df['label'].iloc[i]

        # PIL image -> transform -> batch -> device
        img = Image.open(img_path).convert('RGB')
        img = test_transform(img).unsqueeze(0).to(device)

        # predict
        pred = model(img)                 # [1,1]
        pred = torch.sigmoid(pred)
        pred_cls = (pred >= 0.5).float()  # [1,1]

        # label tensor
        label_tensor = torch.tensor(label, device=device).float().unsqueeze(0).unsqueeze(1)

        acc += (pred_cls == label_tensor).sum().item()

print("Test accuracy:", acc / len(test_df))
