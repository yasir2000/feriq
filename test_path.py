#!/usr/bin/env python3

from pathlib import Path

def test_path_operations():
    current_dir = Path.cwd()
    agents_dir = current_dir / 'agents'
    
    print(f"Current directory: {current_dir}")
    print(f"Agents directory: {agents_dir}")
    print(f"Agents directory exists: {agents_dir.exists()}")
    
    if agents_dir.exists():
        agent_files = list(agents_dir.glob('*.yaml')) + list(agents_dir.glob('*.yml'))
        print(f"Agent files: {agent_files}")
        print(f"Number of agent files: {len(agent_files)}")
    else:
        print("No agents directory found")

if __name__ == "__main__":
    test_path_operations()