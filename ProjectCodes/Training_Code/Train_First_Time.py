import torch
import torch.nn as nn
from Load_Data import load_data
from Model import TrashCNN
from collections import Counter




def main():    
    # Setups
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()

    # Calculate class weights for imbalanced dataset
    counts = Counter(train_loader.dataset.targets)
    print("Class counts:", counts)

    total = sum(counts.values())

    weights = [total / counts[i] for i in range(len(counts))]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("Class weights:", weights)




    num_classes = len(train_loader.dataset.classes)

    model = TrashCNN(num_classes).to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='max', 
                                                           factor=0.5, 
                                                           patience=3,
                                                           min_lr=1e-6)

    best_val_acc = 0

    num_epochs = 20

    #run epochs
    print("START EPOCH LOOP")
    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )
        
        scheduler.step(val_acc)   # if mode='max'
        # or scheduler.step(val_loss) if mode='min'

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
            }, "Training_Code/checkpoint.pth")

            print("Model saved!")


# Training
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    return running_loss / len(loader), accuracy


# Validation
def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return running_loss / len(loader), accuracy



if __name__ == "__main__":
    main()