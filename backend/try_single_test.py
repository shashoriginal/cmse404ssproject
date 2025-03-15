"""
Try running a single test file directly
"""

import os
import sys
import unittest

def run_test(test_file):
    """Run a single test file directly."""
    # Add the backend directory to the path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Load the test module
    test_module = __import__(test_file.replace('.py', '').replace('/', '.'))
    
    # Run the tests
    unittest.main(module=test_module)

if __name__ == "__main__":
    # Run a specific test file
    test_file = "tests/utils/test_helpers.py"
    run_test(test_file) 