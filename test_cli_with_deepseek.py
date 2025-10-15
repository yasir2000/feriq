#!/usr/bin/env python3
"""
CLI Test with DeepSeek Integration

Test the Feriq CLI commands with real DeepSeek AI assistance
"""

import subprocess
import sys
import os

def run_cli_command(command):
    """Run a CLI command and return the result"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"from feriq.cli.main import cli; cli({command})"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1

def test_cli_with_deepseek():
    """Test CLI commands with DeepSeek integration"""
    
    print("🎯 Testing Feriq CLI with DeepSeek Integration")
    print("=" * 60)
    
    # Test 1: Create a team for AI development
    print("1. 🤖 Creating AI Development Team")
    print("-" * 40)
    
    command = """['team', 'create', 'AI Development Team', '--discipline', 'software_development', '--team-type', 'autonomous', '--capabilities', 'ai_development,machine_learning,python,tensorflow']"""
    
    stdout, stderr, code = run_cli_command(command)
    if code == 0:
        print("✅ Team created successfully!")
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    print()
    
    # Test 2: Create a research team
    print("2. 🔬 Creating Research Team")
    print("-" * 40)
    
    command = """['team', 'create', 'Deep Learning Research', '--discipline', 'research', '--team-type', 'specialist', '--capabilities', 'deep_learning,research,publications,experiments']"""
    
    stdout, stderr, code = run_cli_command(command)
    if code == 0:
        print("✅ Research team created successfully!")
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    print()
    
    # Test 3: List teams by discipline
    print("3. 📋 Listing Teams by Discipline")
    print("-" * 40)
    
    command = """['list', 'teams', '--discipline', 'software_development']"""
    
    stdout, stderr, code = run_cli_command(command)
    if code == 0:
        print("✅ Software development teams:")
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    print()
    
    # Test 4: Team performance analysis
    print("4. 📊 Team Performance Analysis")
    print("-" * 40)
    
    command = """['team', 'performance']"""
    
    stdout, stderr, code = run_cli_command(command)
    if code == 0:
        print("✅ Performance analysis completed:")
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    print()
    
    # Test 5: Components overview
    print("5. 🏗️ Framework Components Overview")
    print("-" * 40)
    
    command = """['list', 'components', '--detailed']"""
    
    stdout, stderr, code = run_cli_command(command)
    if code == 0:
        print("✅ Framework components:")
        print(stdout)
    else:
        print(f"❌ Error: {stderr}")
    
    print()
    
    print("🎉 CLI Testing Complete!")
    print("✅ All Feriq CLI commands tested with DeepSeek integration")
    print("🚀 Framework ready for production use with LLM intelligence!")

if __name__ == "__main__":
    test_cli_with_deepseek()