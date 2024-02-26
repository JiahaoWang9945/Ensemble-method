import torch
import random
import numpy as np
from models import Ensemble_model
from dataset import test_dataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

test_loader = torch.utils.data.DataLoader(test_dataset('OCT'), batch_size=32, shuffle=True)

model = Ensemble_model(num_classes=4)
model.load_state_dict(torch.load(''))

model.cuda()

pred_list = []
label_list = []
with torch.no_grad():
    model.eval()
    for image, label in test_loader:
        image = image.cuda()

        pred = model(image)
        pred = torch.max(pred.data, dim=1)[1]
        pred = pred.cpu()
        pred = pred.tolist()
        pred_list.extend(pred)

        label = label.tolist()
        label_list.extend(label)

    accuracy = accuracy_score(label_list, pred_list)
    print(accuracy)

    cm = confusion_matrix(label_list, pred_list)

    # 获取类别数
    num_classes = 4

    precisions = []
    recalls = []
    f1_scores = []
    for class_idx in range(num_classes):
        precision = precision_score(label_list, pred_list, labels=[class_idx], average='micro')
        precisions.append(precision)

        recall = recall_score(label_list, pred_list, labels=[class_idx], average='micro')
        recalls.append(recall)

        f1 = f1_score(label_list, pred_list, labels=[class_idx], average='micro')
        f1_scores.append(f1)

    for class_idx in range(num_classes):
        print("Class {}: Precision={}, Recall={}, F1-score={}".format(class_idx, precisions[class_idx],
                                                                      recalls[class_idx], f1_scores[class_idx]))

