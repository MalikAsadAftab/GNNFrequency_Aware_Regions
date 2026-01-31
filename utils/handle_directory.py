# utils/handle_directory.py

import os

def get_dataset_dir(dataset_name):
    """
    Return the path to the dataset directory.
    Assumes datasets are stored in a folder called 'datasets' at the project root.
    """
    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    return dataset_dir

def get_result_dir(result_prefix):
    """
    Create and return a result directory path.
    """
    base_dir = os.getcwd()
    results_root = os.path.join(base_dir, 'results')
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    # Generate unique result directory with timestamp
    import datetime
    now = datetime.datetime.now()
    result_dir_name = result_prefix + now.strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(results_root, result_dir_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_dir_in_result(result_dir, sub_dir_name):
    """
    Create sub-directory inside a given result directory.
    """
    sub_dir = os.path.join(result_dir, sub_dir_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    return sub_dir

def generate_results_viewer_1D(result_dir):
    """
    Optional: Could generate a simple HTML or text file to view results.
    Here we just create a placeholder text file listing results.
    """
    file_path = os.path.join(result_dir, "results_viewer.txt")
    with open(file_path, "w") as f:
        f.write("Results Viewer (1D AoA Estimation)\n")
        f.write("===============================\n")
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                f.write(f"{os.path.join(root, file)}\n")
