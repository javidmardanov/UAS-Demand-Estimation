#!/usr/bin/env python3
"""
Run tests for the UAS Last-Mile Delivery Simulator
"""

import pytest
import sys

def main():
    # Run the tests
    print("Running tests...")
    result = pytest.main(["-xvs", "test_app.py"])
    
    # Check the result
    if result == 0:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 