import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

# Model setup with partial layer unfreezing
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated for deprecated 'pretrained'
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:  # Fine-tune last layer and fully connected layer
        param.requires_grad = True
    else:
        param.requires_grad = False

num_classes = 27
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to('cuda')  # Move model to GPU

# Optimizer and Scheduler with cosine annealing
learning_rate = 0.0001
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Enhanced data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='D:/ASL_Alphabet_Dataset/asl_alphabet_train', transform=transform)
val_dataset = datasets.ImageFolder(root='D:/ASL_Alphabet_Dataset/asl_alphabet_test', transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Training loop with accuracy tracking
num_epochs = 20
early_stopping_patience = 5
best_val_acc = 0.0
epochs_no_improve = 0

scaler = GradScaler("cuda")  # initialize the gradient scaler

# Training loop with mixed precision
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0.0, 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
        images, labels = images.to('cuda', non_blocking=True), labels.to('cuda', non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast("cuda"):  # enable mixed precision
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
        
        scaler.scale(loss).backward()  # scales loss and backpropagates
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            images, labels = images.to('cuda', non_blocking=True), labels.to('cuda', non_blocking=True)
            
            with autocast("cuda"):  # mixed precision for validation
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Update the scheduler and early stopping mechanism
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break
