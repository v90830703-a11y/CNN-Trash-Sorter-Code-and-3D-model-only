from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.datasets as datasets

data_dir = "Training_Code/Data_Processed"

dataset = datasets.ImageFolder(data_dir)
targets = np.array(dataset.targets)

# Train: 70%, Temp: 30%
train_idx, temp_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.3,
    stratify=targets,
    random_state = 101
)

# Val: 15%, Test: 15%
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=targets[temp_idx],
    random_state = 101
)


# Verify the distribution of classes in each split
from collections import Counter

def show_distribution(name, labels):
    counts = Counter(labels)
    total = len(labels)

    print(f"\n{name} ({total} images)")
    for cls, count in sorted(counts.items()):
        print(
            f"Class {cls}: {count:4d} "
            f"({100*count/total:.2f}%)"
        )

show_distribution("Train", targets[train_idx])
show_distribution("Val", targets[val_idx])
show_distribution("Test", targets[test_idx])
show_distribution("Original", targets)




# Save files to respective folders
import os
import shutil

output_dir = 'Training_Code/Data_Splitted'

def copy_images_to_folders(indices, subset_name):
    for idx in indices:
        image_path, label_idx = dataset.samples[idx]
        class_name = dataset.classes[label_idx]
        
        # Create destination directory (e.g., split/train/dogs)
        dest_class_dir = os.path.join(output_dir, subset_name, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        # Copy the image
        shutil.copy(image_path, dest_class_dir)

# Execute copying for each split
copy_images_to_folders(train_idx, 'train')
copy_images_to_folders(val_idx, 'val')
copy_images_to_folders(test_idx, 'test')
print("Image stratification and storage complete!")
