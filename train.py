import torch
import random
import numpy as np
import torch.optim as optim
from models import Ensemble_model
from sklearn.metrics import accuracy_score
from dataset import train_dataset, val_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(train_dataset('OCT'), batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset('OCT'), batch_size=32, shuffle=True)

model = Ensemble_model(num_classes=4)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model.cuda()

epochs = 50
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

k = 1
print('开始训练：')
for i in range(epochs):
    model.train()
    train_loss = 0
    train_accuracy = 0
    for image, label in train_loader:
        image = image.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        train_loss += loss.item()
        pred = torch.max(pred.data, dim=1)[1]
        train_accuracy += accuracy_score(pred.cpu(), label.cpu())
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), str(k) + '.pth')
    k += 1

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        model.eval()
        for image, label in val_loader:
            image = image.cuda()
            label = label.cuda()
            pred = model(image)
            test_loss += criterion(pred, label)
            pred = torch.max(pred.data, dim=1)[1]
            test_accuracy += accuracy_score(pred.cpu(), label.cpu())

    print('num: {}/{}.. '.format(i + 1, epochs),
          'current_lr: {:.5f}.. '.format(current_lr),
          'train_loss: {:.5f}.. '.format(train_loss / len(train_loader)),
          'train_accuracy: {:.5f}.. '.format(train_accuracy / len(train_loader)),
          'test_loss: {:.5f}.. '.format(test_loss / len(val_loader)),
          'test_accuracy: {:.5f}.. '.format(test_accuracy / len(val_loader)))
