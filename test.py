import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

model = torch.load("DCNet.pth")
image_path = "DataSets/test/cats/cat.7500.jpg"


img = Image.open(image_path)
data_transform = transforms.Compose([transforms.Scale([224, 224]),
                                     transforms.ToTensor()])
img = data_transform(img)
x = []
for i in range(16):
    x.append(img)

x = torch.stack(x, dim=0)
x = Variable(x.cpu())
y = model(x)
y = y[0]
if y[0] < y[1]:
    print("this is a dog")
else:
    print("this is a cat")

img = img.numpy().transpose([1, 2, 0])
plt.imshow(img)
plt.show()