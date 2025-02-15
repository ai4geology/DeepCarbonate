# trainer.py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

class MetricTracker:
    """Indicator Tracker"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.all_outputs = []
        self.all_labels = []
        self.total_loss = 0.0
        self.top1_correct = 0
        self.top5_correct = 0
        self.total_samples = 0
        
    def update(self, outputs, labels, loss):
        with torch.no_grad():
            # Basic indicators
            self.all_outputs.append(outputs.cpu())
            self.all_labels.append(labels.cpu())
            self.total_loss += loss.item()
            
            # Top-k accuracy
            _, preds = torch.topk(outputs, k=5, dim=1)
            correct = preds.eq(labels.view(-1,1).expand_as(preds))
            self.top1_correct += correct[:,0].sum().item()
            self.top5_correct += correct.any(dim=1).sum().item()
            self.total_samples += labels.size(0)

def evaluate_metrics(tracker, phase='val'):
    """Calculate complete evaluation indicators"""
    # Merge the results of all batches
    outputs = torch.cat(tracker.all_outputs)
    labels = torch.cat(tracker.all_labels)
    probs = torch.softmax(outputs, dim=1).numpy()
    preds = torch.argmax(outputs, dim=1).numpy()
    labels_np = labels.numpy()
    
    # Basic indicator calculation
    metrics = {
        'loss': tracker.total_loss / len(tracker.all_outputs),
        'accuracy': accuracy_score(labels_np, preds),
        'precision': precision_score(labels_np, preds, average='weighted'),
        'recall': recall_score(labels_np, preds, average='weighted'),
        'f1': f1_score(labels_np, preds, average='weighted'),
        'top1_acc': 100 * tracker.top1_correct / tracker.total_samples,
        'top5_acc': 100 * tracker.top5_correct / tracker.total_samples,
    }
    
    # Multi class AUC (OvR strategy)
    if tracker.num_classes > 2:
        metrics['auc'] = roc_auc_score(
            labels_np, probs, 
            multi_class='ovr',
            average='weighted'
        )
    else:
        metrics['auc'] = roc_auc_score(labels_np, probs[:,1])
    
    # Classification report and confusion matrix
    print("\nClassification report:")
    print(classification_report(
        labels_np, preds,
        target_names=[str(i) for i in range(tracker.num_classes)]
    ))
    
    # Generate confusion matrix only on the test set
    if phase == 'test':
        cm = confusion_matrix(labels_np, preds)
        print("\nconfusion matrix:")
        print(cm)
        
        # The correct number of predictions for each category
        class_correct = cm.diagonal()
        print("\nCorrectly predicted numbers for each category:")
        for i, count in enumerate(class_correct):
            print(f"Category {i}: {count}")
    
    return metrics

# trainer.py
class EarlyStopping:
    """Early stop mechanism"""
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # Number of epochs allowed without improvement
        self.delta = delta        # Minimum improvement threshold
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_accuracy, model, save_path):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Early stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), save_path)
            self.counter = 0

def train_model(model, loaders, num_classes, epochs, lr=0.01, early_stopping_patience=5):
    """Complete training process"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = loaders
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    model.to(device)
    early_stopping = EarlyStopping(patience=early_stopping_patience, delta=0.005)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_tracker = MetricTracker(num_classes)
        
        for inputs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_tracker.update(outputs, labels, loss)
        
        # Training metrics
        train_metrics = evaluate_metrics(train_tracker, 'train')
        
        # Val
        model.eval()
        val_tracker = MetricTracker(num_classes)
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_tracker.update(outputs, labels, loss)
        
        val_metrics = evaluate_metrics(val_tracker, 'val')
        
        # 打印指标
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Training Set - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2%}")
        print(f"Validation Set - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2%}")
        print(f"Detailed test indicators:")
        print(f"• AUC: {val_metrics['auc']:.4f}")
        print(f"• Precision: {val_metrics['precision']:.4f}")
        print(f"• Recall: {val_metrics['recall']:.4f}") 
        print(f"• F1-score: {val_metrics['f1']:.4f}")
        print(f"• Top-1 Acc: {val_metrics['top1_acc']:.2f}%")
        print(f"• Top-5 Acc: {val_metrics['top5_acc']:.2f}%")
        
        # 早停检查
        if early_stopping(val_metrics['accuracy'], model, 'best_model.pth'):
            print("\nEarly stop trigger, training ends early")
            break
    
    # 最终测试
    model.load_state_dict(torch.load('best_model.pth'))
    test_tracker = MetricTracker(num_classes)
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_tracker.update(outputs, labels, loss)
    
    test_metrics = evaluate_metrics(test_tracker, 'test')
    
    print("\nFinal Test Set Metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k.upper():<10}: {v:.4f}")