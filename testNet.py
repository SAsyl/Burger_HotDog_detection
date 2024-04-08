from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch.efficientnet.models.efficientnet import EfficientNet

from torch import nn
from collections import OrderedDict


checkpoint = torch.load('./efficientnet_pytorch/experiments/fastfood_v2/best.pth')
model = EfficientNet(1.0, 1.0, 0.2)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2), nn.Softmax(dim=1))

state_dict = checkpoint['model']
model_dict = model.state_dict()
new_state_dict = OrderedDict()
matched_layers, discarded_layers = [], []
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k[7:]

    if k in model_dict and model_dict[k].size() == v.size():
        new_state_dict[k] = v
        matched_layers.append(k)
    else:
        discarded_layers.append(k)

model_dict.update(new_state_dict)
model.load_state_dict(model_dict)

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open('Cat.jpg')).unsqueeze(0)
# img = tfms(Image.open('test_fastfood/Burger-Test (1).jpeg')).unsqueeze(0)
# img = tfms(Image.open('test_fastfood/Burger-Test (5).jpeg')).unsqueeze(0)
# img = tfms(Image.open('test_fastfood/Hot Dog-Test (86).jpeg')).unsqueeze(0)
# img = tfms(Image.open('test_fastfood/Hot Dog-Test (93).jpeg')).unsqueeze(0)
img = tfms(Image.open('mytest/burger2.jpg')).unsqueeze(0)
# img = tfms(Image.open('mytest/burger3.jpg')).unsqueeze(0)
# img = tfms(Image.open('mytest/burger4.jpg')).unsqueeze(0)
# img = tfms(Image.open('mytest/hotdog.jpg')).unsqueeze(0)

model.eval()
with torch.no_grad():
    res = model(img)


# print(res)

classes = ['Burger', 'Hot Dog']
predicted_class = torch.argmax(res[0])
print(f"It's a {classes[predicted_class]} with probability {res[0][predicted_class]:.3f}")
