import os


from Uni_Size_Image import resize_pad_and_replace_image

Data = r"Training_Code/Data"

all_subtypes = [f for f in os.listdir(Data) if os.path.isdir(os.path.join(Data, f))]

for subtype in all_subtypes:
    folder_path = os.path.join(Data, subtype)
    print(f"Processing folder: {folder_path}")
    # Get a sorted list of all files in the directory
    files = sorted(os.listdir(folder_path))

    if files:
        for file in files:
            file_path = os.path.join(folder_path, file)# Combine the folder path and the first filename
            resize_pad_and_replace_image(file_path)
    else:
        print("The folder is empty.")
