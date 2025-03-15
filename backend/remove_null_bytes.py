"""
Remove null bytes from Python files
"""

import os

def remove_null_bytes(directory):
    """Remove null bytes from all Python files in the given directory recursively."""
    files_fixed = 0
    files_checked = 0
    
    print(f"Checking directory: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                files_checked += 1
                try:
                    # Read file content as binary
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    # Check if there are null bytes
                    if b'\x00' in content:
                        print(f"Found null bytes in {filepath}")
                        
                        # Remove null bytes
                        content_fixed = content.replace(b'\x00', b'')
                        
                        # Write back the content
                        with open(filepath, 'wb') as f:
                            f.write(content_fixed)
                        
                        files_fixed += 1
                        print(f"Fixed {filepath}")
                    else:
                        print(f"Checked {filepath} - No null bytes found")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    print(f"Checked {files_checked} files, fixed {files_fixed} files in {directory}")
    return files_fixed

if __name__ == "__main__":
    print("Removing null bytes from Python files...")
    src_dir = "src"
    tests_dir = "tests"
    
    if os.path.exists(src_dir):
        src_files_fixed = remove_null_bytes(src_dir)
        print(f"Fixed {src_files_fixed} files in {src_dir}")
    
    if os.path.exists(tests_dir):
        tests_files_fixed = remove_null_bytes(tests_dir)
        print(f"Fixed {tests_files_fixed} files in {tests_dir}")
    
    print("Done!") 