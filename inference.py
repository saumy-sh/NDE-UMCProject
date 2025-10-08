import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# classes
classes = {0: 'Difetto1', 1: 'Difetto2', 2: 'Difetto4', 3: 'NoDifetto'}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model.load_state_dict(torch.load('experimentation/squeezenet/squeezenet_fine-tuned.pth',map_location=device),strict=False)   


test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

def inference(img_path):
    model.eval()
    with torch.no_grad():
        # PIL image -> transform -> batch -> device
        img = Image.open(img_path).convert('RGB')
        img = test_transform(img).unsqueeze(0).to(device)

        # predict
        pred = model(img)
        print(pred)
        pred = torch.argmax(pred,dim=1).item()
        print(classes[pred])
        return classes[pred]