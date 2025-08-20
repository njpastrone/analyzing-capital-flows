#!/usr/bin/env python3
"""
Test Runner for Capital Flows Research Project

Simple test runner script to execute the full test suite and provide summary results.
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the full test suite and provide summary"""
    print("🧪 Capital Flows Research - Test Suite")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return False
    
    # Run pytest with verbose output
    try:
        print(f"Running tests from: {tests_dir}")
        print()
        
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(tests_dir),
            "-v",
            "--tb=short",
            "--durations=10"  # Show 10 slowest tests
        ], 
        cwd=project_root,
        capture_output=False  # Let pytest output stream directly
        )
        
        # Check results
        if result.returncode == 0:
            print("\n🎉 All tests passed!")
            print("\n✅ Capital Flows data pipeline is healthy and reliable")
            return True
        else:
            print("\n❌ Some tests failed!")
            print(f"Exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function"""
    success = run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()