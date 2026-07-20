import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def load_data():
    # 1. Define  transformations using torchvision.transforms.v2
    train_transform = v2.Compose([
        v2.Resize((96, 96)),
        v2.RandomRotation(degrees=15),        
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # 2. Point to the root directory containing class folders
    dataset_train = datasets.ImageFolder(root='Training_Code/Data_Splitted/train', transform=train_transform)
    dataset_val = datasets.ImageFolder(root='Training_Code/Data_Splitted/val', transform=val_test_transform)
    dataset_test = datasets.ImageFolder(root='Training_Code/Data_Splitted/test', transform=val_test_transform)


    # 3. Wrap with a DataLoader to handle batching and shuffling   
    train_loader = DataLoader(dataset_train, 
                            batch_size=16, 
                            shuffle=True, 
                            num_workers=0)

    val_loader = DataLoader(dataset_val, 
                            batch_size=16, 
                            shuffle=False, 
                            num_workers=0)

    test_loader = DataLoader(dataset_test, 
                            batch_size=16, 
                            shuffle=False, 
                            num_workers=0)
    
    print('data loaded successfully')
    print("Total train images:", len(dataset_train))
    print("Num classes:", len(dataset_train.classes))

    # for images, labels in train_loader:
    #     print(images.shape)
    #     print(labels.shape)
    #     print(labels[:10])
    #     break

    # for images, labels in val_loader:
    #     print(images.shape)
    #     print(labels.shape)
    #     print(labels[:10])
    #     break

    # for images, labels in test_loader:
    #     print(images.shape)
    #     print(labels.shape)
    #     print(labels[:10])
    #     break

    return train_loader, val_loader, test_loader

