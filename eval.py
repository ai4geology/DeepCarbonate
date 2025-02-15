import torch
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

def calculate_topk_accuracy(outputs, labels, k):
    """
    计算Top-k准确率
    """
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return correct.float().sum(0) / labels.size(0) * 100

def calculate_metrics(outputs, labels, num_classes):
    """
    计算准确率、召回率、精确率、F1分数和Top-k准确率
    """
    _, preds = torch.max(outputs, 1)
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    # 准确率
    accuracy = (preds == labels).mean() * 100

    # 召回率、精确率、F1分数
    precision, recall, f1, _ = score(labels, preds, average='weighted')

    # Top-1和Top-5准确率
    top1 = calculate_topk_accuracy(outputs, labels, k=1)
    top5 = calculate_topk_accuracy(outputs, labels, k=5)

    return accuracy, precision, recall, f1, top1, top5

def calculate_confusion_matrix(outputs, labels, num_classes):
    """
    计算混淆矩阵
    """
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return cm