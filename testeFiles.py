# import os
# import pandas as pd

# # Specify the parent folder path
# folder_path = 'mixedDataset/train'

# # Initialize an empty list to store file information
# data = []

# # Loop through the subfolders (e.g., class_43, class_44)
# for class_folder in os.listdir(folder_path):
#     class_path = os.path.join(folder_path, class_folder)
    
#     # Ensure it's a directory
#     if os.path.isdir(class_path):
#         # Get the list of file names in the subfolder
#         file_names = os.listdir(class_path)
        
#         # Add file info to the data list (file name and class label)
#         for file_name in file_names:
#             file_path = os.path.join(class_path, file_name)
#             if os.path.isfile(file_path):
#                 data.append({'image_name': file_name, 'number': class_folder})

# # Create a DataFrame with the file information
# df = pd.DataFrame(data)
# class_mapping = {
#     'minibus': 43,
#     'sport games motorcycle racing': 44
# }

# df['Class'] = df['Class'].replace(class_mapping)
# # Display the DataFrame
# print(df)
import os
import pandas as pd

def readImages(path):
    # Specify the parent folder path
    folder_path = 'mixedDataset/' + path + '/'

    # Initialize an empty list to store file information
    data = []
    # Loop through the subfolders (e.g., minibus, sport games motorcycle racing)
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        
        # Ensure it's a directory
        if os.path.isdir(class_path):
            # Get the list of file names in the subfolder
            file_names = os.listdir(class_path)
            
            # Add file info to the data list (full file path and class label), only for .jpg files
            for file_name in file_names:
                if file_name.lower().endswith('.jpg'):  # Filter only .jpg files
                    file_path = os.path.join(class_folder, file_name)  # Include the subfolder in the path
                    if os.path.isfile(os.path.join(class_path, file_name)):
                        data.append({'image_name': file_path, 'number': class_folder})

    # Create a DataFrame with the file information
    df = pd.DataFrame(data)

    # Replace class labels with numbers
    class_mapping = {
        'minibus': 43,
        'cab': 44
    }
    df['number'] = df['number'].replace(class_mapping)
    print(df.head(1))
    cabDf = df[df['number'] == 44]
    print(cabDf.shape)
    minibusDf = df[df['number'] == 43]
    print(minibusDf.shape)
    return df



readImages("train")