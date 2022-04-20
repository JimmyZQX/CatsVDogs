import torch
from torchvision import datasets, transforms
from torch.utils import data
import torchvision
import os
import matplotlib.pyplot as plt

data_dir = "DataSets"
data_transform = {x: transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor()])
                  for x in ["train", "test"]}
image_dataSets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "test"]}
data_loader = {x: torch.utils.data.DataLoader(dataset=image_dataSets[x],
                                              batch_size=16,
                                              shuffle=True)
               for x in ["train", "test"]}

def dataLoader():
    # x_example, y_example = next(iter(data_loader["train"]))
    # print(y_example)
    # img = torchvision.utils.make_grid(x_example)
    # img = img.numpy().transpose([1, 2, 0])
    # plt.imshow(img)
    # plt.show()
    return data_loader

