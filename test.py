import os
from plagiarism_detection import detect_clone_type


def process_output_dataset(base_dir):
    """
    Traverse the output_dataset directory and find pairs of files to process.
    """
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            file1 = find_file_in_subdir(os.path.join(folder_path, "1"))
            file2 = find_file_in_subdir(os.path.join(folder_path, "2"))

            if file1 and file2:
                print(f"Processing pair: {file1} and {file2}")
                print(f"Type: {detect_clone_type(file1, file2)}")
            else:
                print(f"Missing files in folder: {folder_path}")


def find_file_in_subdir(base_subdir):
    """
    Find the first Python file in the given subdirectory.
    """
    for root, _, files in os.walk(base_subdir):
        for file in files:
            if file.endswith(".py"):
                return os.path.join(root, file)
    return None


if __name__ == "__main__":
    base_dir = "/home/oskar/Documents/ITC/software-avanzado/reto/output_dataset"
    # process_output_dataset(base_dir)
    # Example usage: Compare two specific files
    file_path1 = "/home/oskar/Documents/ITC/software-avanzado/reto/dummy_1.py"
    file_path2 = "/home/oskar/Documents/ITC/software-avanzado/reto/dummy_2.py"

    if os.path.exists(file_path1) and os.path.exists(file_path2):
        with open(file_path1, "r") as f1, open(file_path2, "r") as f2:
            code1 = f1.read()
            code2 = f2.read()
            similarity = detect_clone_type(code1, code2)
            print(f"Type: {file_path1} and {file_path2}: {similarity}")
    else:
        print("One or both file paths do not exist.")
