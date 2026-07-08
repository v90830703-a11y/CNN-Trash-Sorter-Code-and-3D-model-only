import pandas as pd
import os
import shutil

SOURCE_DIR = "Data"          # contains subclass folders
DEST_DIR = "Data_Processed"   # output

# Read mapping file
df = pd.read_csv("Classify_Subclasses.txt", sep=r"\s+")

class_map = {
    1: "Bio-Waste",
    2: "E-Waste",
    3: "General-Waste",
    4: "Recyclable-Waste"
}

# Create the 4 class folders
for class_name in class_map.values():
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)

# Process each subclass
for _, row in df.iterrows():

    subclass_id = int(row["ID"])
    subclass_name = row["Subclass"]

    class_folder = class_map[subclass_id // 100]

    src_folder = os.path.join(SOURCE_DIR, subclass_name)
    dst_folder = os.path.join(DEST_DIR, class_folder)

    if not os.path.isdir(src_folder):
        print(f"Missing: {src_folder}")
        continue

    # Move all images directly into class folder
    for filename in os.listdir(src_folder):

        src_file = os.path.join(src_folder, filename)

        if os.path.isfile(src_file):
            dst_file = os.path.join(dst_folder, filename)

            # Avoid filename collisions
            if os.path.exists(dst_file):
                base, ext = os.path.splitext(filename)
                dst_file = os.path.join(
                    dst_folder,
                    f"{subclass_name}_{base}{ext}"
                )

            shutil.move(src_file, dst_file)

    # Remove now-empty subclass folder
    os.rmdir(src_folder)

    print(f"Processed {subclass_name}")