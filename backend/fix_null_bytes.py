"""
Fix null bytes in Python files
"""

import os
import re

def fix_files(directory):
    """Remove null bytes from all Python files recursively."""
    fixed_files = 0
    
    print(f"Checking directory: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                # Open file and read content as binary
                try:
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    # Check if there are null bytes or other problematic binary data
                    if b'\x00' in content:
                        print(f"Found null bytes in {filepath}")
                        
                        # Replace null bytes
                        new_content = content.replace(b'\x00', b'')
                        
                        # Write the new content
                        with open(filepath, 'wb') as f:
                            f.write(new_content)
                        
                        fixed_files += 1
                        print(f"Fixed {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    
    return fixed_files

if __name__ == "__main__":
    print("Fixing null bytes in Python files...")
    
    # Fix files in src and tests directories
    fixed_in_src = fix_files("src")
    fixed_in_tests = fix_files("tests")
    
    # Create fresh __init__.py files in all directories
    for root, dirs, _ in os.walk('.'):
        # Skip .git directory
        if '.git' in root:
            continue
            
        # Skip directories that start with .
        if os.path.basename(root).startswith('.'):
            continue
            
        for dir_name in dirs:
            # Skip directories that start with .
            if dir_name.startswith('.'):
                continue
                
            dir_path = os.path.join(root, dir_name)
            init_file = os.path.join(dir_path, '__init__.py')
            
            # Create a fresh __init__.py file in each directory if not exists
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Package initialization"""\n')
                print(f"Created {init_file}")
    
    print(f"Fixed files in src: {fixed_in_src}")
    print(f"Fixed files in tests: {fixed_in_tests}")
    print("Done!") 