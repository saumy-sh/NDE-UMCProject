import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
import torchvision.transforms as transforms
from torchsummary import summary
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


images_dir = './weld_defects/JPEGImages'
labels_dir = './weld_defects/Annotations'


def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # collect all categories (in case there are multiple <object> tags)
    category = [obj.find('name').text for obj in root.findall('object')]
    if category[0] == 'good':
        return 0
    return 1


labels = os.listdir(labels_dir)

if 'weld_test.csv' not in os.listdir('../cement_cracks'):

    y = [read_xml(os.path.join(labels_dir,f)) for f in labels]
    x = [os.path.join(images_dir,f) for f in os.listdir(images_dir) if f.endswith(('.jpg'))]
    X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.2,shuffle=True)
    X_val,X_test,y_val,y_test = train_test_split(X_val,y_val,test_size=0.5,shuffle=True)

    train_data = {
        'img_path':X_train,
        'label':y_train
    }

    val_data = {
        'img_path':X_val,
        'label':y_val
    }

    test_data = {
        'img_path':X_test,
        'label':y_test
    }

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv('weld_train.csv',index=False)
    val_df.to_csv('weld_val.csv',index=False)
    test_df.to_csv('weld_test.csv',index=False)
else:
    train_df = pd.read_csv('weld_train.csv')
    val_df = pd.read_csv('weld_val.csv')


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])





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
    

train_dataset = Crack(train_df,transform)
val_dataset = Crack(val_df,transform)

BATCH_SIZE = 8
EPOCHS = 20
LR = 0.0001
N_FEATURES = 1

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)



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
model.load_state_dict(torch.load('pre-trained-weld.pth'),strict=False)

summary(model,input_size=(3,224,224))
criterion = nn.BCEWithLogitsLoss().to(device)
optimiser = Adam(model.parameters(),lr=LR)


acc_train_plot, acc_val_plot, loss_train_plot, loss_val_plot = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    train_acc, train_loss = 0, 0
    total_train = 0
    
    for images, labels in train_loader:
        labels = labels.to(device).float().unsqueeze(1)
        predictions = model(images)
        optimiser.zero_grad()
        batch_loss = criterion(predictions, labels)
        batch_loss.backward()
        optimiser.step()
        
        train_loss += batch_loss.item()
        
        
        preds = torch.sigmoid(predictions)
        preds_cls = (preds >= 0.5).float()
        train_acc += (preds_cls == labels).sum().item()
        total_train += labels.size(0)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / total_train
    
    # --- validation ---
    model.eval()
    val_acc, val_loss, total_val = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.to(device).float().unsqueeze(1)
            predictions = model(images)
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            
            preds = torch.sigmoid(predictions)
            preds_cls = (preds >= 0.5).float()
            val_acc += (preds_cls == labels).sum().item()
            total_val += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / total_val
    
    
    loss_train_plot.append(avg_train_loss)
    loss_val_plot.append(avg_val_loss)
    acc_train_plot.append(avg_train_acc)
    acc_val_plot.append(avg_val_acc)
    
    print(f"Epoch {epoch+1}:")
    print(f"Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f} ") 
    print(f"Val Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")


torch.save(model.state_dict(),'fine-tuned-weld.pth')

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
ax[0].plot(loss_train_plot,label="Training Loss")
ax[0].plot(loss_val_plot,label="Validation Loss")
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_ylim([0,1])
ax[0].set_title("Loss vs Epoch")
ax[0].legend()

ax[1].plot(acc_train_plot,label="Training Accuracy")
ax[1].plot(acc_val_plot,label="Validation Accuracy")
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_ylim([0,1])
ax[1].set_title("Accuracy vs Epoch")
ax[1].legend()
plt.savefig('accuracy-loss-weld-fine-tuned.jpg')
