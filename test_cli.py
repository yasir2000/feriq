#!/usr/bin/env python3
"""
Test script for Feriq CLI

This script demonstrates the CLI functionality.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show the result."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main test function."""
    print("Feriq CLI Test Suite")
    print("=" * 60)
    
    # Change to the feriq directory
    feriq_dir = Path(__file__).parent
    os.chdir(feriq_dir)
    
    # Install in development mode
    print("Installing Feriq in development mode...")
    install_result = run_command("pip install -e .", "Install Feriq package")
    
    if not install_result:
        print("❌ Installation failed. Cannot proceed with tests.")
        return
    
    # Test basic CLI functionality
    tests = [
        ("python -m feriq.cli.main --help", "CLI Help"),
        ("python -m feriq.cli.main version", "Version Command"),
        ("python -m feriq.cli.main doctor", "Doctor Command"),
        ("python -m feriq.cli.main config", "Config Command"),
        ("python -m feriq.cli.main model list", "Model List Command"),
        ("python -m feriq.cli.main init --help", "Init Help"),
        ("python -m feriq.cli.main agent --help", "Agent Help"),
        ("python -m feriq.cli.main goal --help", "Goal Help"),
        ("python -m feriq.cli.main workflow --help", "Workflow Help"),
        ("python -m feriq.cli.main status --help", "Status Help"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<40} {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)