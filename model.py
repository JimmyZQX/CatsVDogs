import torch
from torchvision import models
import data_load
from data_load import dataLoader
from torch.autograd import Variable

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(7*7*512, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 2),
)
model.avgpool = torch.nn.Sequential()
model = model.cpu()
print(model)
loss_fn = torch.nn.CrossEntropyLoss() # softmax
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)
epoch_n = 5
data_loader = dataLoader()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch + 1, epoch_n))
    print("=" * 25)

    running_loss = 0.0
    running_corrects = 0
    for batch, data in enumerate(data_loader["train"], 1):
        x, y = data
        x, y = Variable(x), Variable(y)

        y_pred = model(x)

        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(pred == y.data)

        if batch % 2 == 0:
            print("Batch {}, Train loss:{:.4f}, Train ACC:{:.2f}%"
                  .format(batch,
                          running_loss / batch,
                          100 * running_corrects / (16 * batch)))


    epoch_loss = running_loss * 16 / len(data_load.image_dataSets["train"])
    epoch_acc = 100 * float(running_corrects) / len(data_load.image_dataSets["test"])
    print("Train: Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))

torch.save(model, "DCNet.pth")

print("Testing...")
print("="*25)
running_loss = 0.0
running_corrects = 0
for batch, data in enumerate(data_loader["test"], 1):
    x, y = data
    x, y = Variable(x.cpu()), Variable(y.cpu())

    y_pred = model(x)

    _, pred = torch.max(y_pred.data, 1)
    loss = loss_fn(y_pred, y)

    running_loss += loss.item()
    running_corrects += torch.sum(pred == y.data)

epoch_loss = running_loss * 16 / len(data_load.image_dataSets["test"])
epoch_acc = 100 * float(running_corrects) / len(data_load.image_dataSets["test"])
print("Test: Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))