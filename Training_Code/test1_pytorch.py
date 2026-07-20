import torch
import torch.nn as nn
from Load_Data import load_data
from Model import TrashCNN

def main():
    # Setups
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()

    num_classes = 4 #len(test_loader.dataset.classes)

    final_model = torch.load("Training_Code/checkpoint.pth",map_location=device)

    model = TrashCNN(num_classes).to(device)
    model.load_state_dict(final_model["model_state_dict"])  

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = testing(
            model,
            test_loader,
            criterion,
            device
        )

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )


# Validation
def testing(model, loader, criterion, device):

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