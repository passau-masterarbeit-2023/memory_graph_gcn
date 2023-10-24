import glob
import os


def find_gv_files(directory_path):
    return find_files_with_extension(directory_path, 'gv')

def find_pickle_files(directory_path):
    return find_files_with_extension(directory_path, 'pickle')

def find_files_with_extension(directory_path, extension):
    # Create a pattern to match .gv files in all subdirectories
    pattern = os.path.join(directory_path, '**', f'*.{extension}')
    
    # Use glob.glob with the recursive pattern
    files = glob.glob(pattern, recursive=True)
    
    return files