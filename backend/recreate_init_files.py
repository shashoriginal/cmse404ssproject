"""
Recreate all __init__.py files
"""

import os
import shutil

def recreate_init_files():
    """Delete and recreate all __init__.py files."""
    # Directories where __init__.py files should exist
    target_dirs = [
        'src',
        'src/models',
        'src/utils',
        'tests',
        'tests/models',
        'tests/utils'
    ]
    
    # Delete and recreate each __init__.py file
    for dir_path in target_dirs:
        if os.path.exists(dir_path):
            init_file = os.path.join(dir_path, '__init__.py')
            
            # Delete the file if it exists
            if os.path.exists(init_file):
                try:
                    os.remove(init_file)
                    print(f"Deleted {init_file}")
                except Exception as e:
                    print(f"Error deleting {init_file}: {e}")
            
            # Create a fresh __init__.py file
            try:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""\n{dir_path} package initialization.\n"""\n')
                print(f"Created {init_file}")
            except Exception as e:
                print(f"Error creating {init_file}: {e}")
        else:
            print(f"Directory {dir_path} does not exist, creating it")
            try:
                os.makedirs(dir_path, exist_ok=True)
                init_file = os.path.join(dir_path, '__init__.py')
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""\n{dir_path} package initialization.\n"""\n')
                print(f"Created {init_file}")
            except Exception as e:
                print(f"Error creating directory or file: {e}")

if __name__ == "__main__":
    print("Recreating __init__.py files...")
    recreate_init_files()
    print("Done!") 