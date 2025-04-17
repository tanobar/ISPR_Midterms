'''RUN THIS SCRIPT TO DELETE model.pth AND encodings.pkl FILES IN THE CURRENT DIRECTORY'''
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate through all files in the directory
for filename in os.listdir(script_dir):
    # Check if the file ends with .pth or .pkl
    if filename.endswith('.pth') or filename.endswith('.pkl'):
        file_path = os.path.join(script_dir, filename)
        try:
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")