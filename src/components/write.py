import os

# Path to the artifacts folder
folder_path = "artifacts"
test_file_path = os.path.join(folder_path, "test_write.txt")

try:
    # Try creating and writing to the test file
    os.makedirs(folder_path, exist_ok=True)  # Make sure the directory exists
    with open(test_file_path, 'w') as f:
        f.write("This is a test to check write permission.")
    print(f"File written successfully to {test_file_path}")
except PermissionError:
    print(f"Permission denied: Cannot write to {folder_path}")
except Exception as e:
    print(f"Error occurred: {e}")
