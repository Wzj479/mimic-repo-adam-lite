"""
Code snapshot utility for saving changed files to training log directories.
This helps track which code version was used for each training run.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def get_git_root() -> Optional[str]:
    """Get the git repository root directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_changed_files(git_root: str) -> List[str]:
    """
    Get list of changed files using git.
    Returns files that are:
    - Modified but not staged
    - Staged for commit
    - Untracked Python files
    """
    changed_files = []
    
    try:
        # Get modified and staged files
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        changed_files.extend(result.stdout.strip().split('\n'))
        
        # Get untracked Python files
        result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        untracked = result.stdout.strip().split('\n')
        # Only include Python files and config files
        untracked_filtered = [
            f for f in untracked 
            if f and (f.endswith('.py') or f.endswith('.yaml') or f.endswith('.yml'))
        ]
        changed_files.extend(untracked_filtered)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Remove empty strings and duplicates
    changed_files = list(set([f for f in changed_files if f]))
    
    return changed_files


def get_git_diff(git_root: str) -> Optional[str]:
    """Get the git diff for all changes."""
    try:
        result = subprocess.run(
            ['git', 'diff', 'HEAD'],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_info(git_root: str) -> dict:
    """Get git commit info."""
    info = {}
    
    try:
        # Get current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        info['commit_hash'] = result.stdout.strip()
        
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        info['branch'] = result.stdout.strip()
        
        # Get commit message
        result = subprocess.run(
            ['git', 'log', '-1', '--pretty=%B'],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        info['commit_message'] = result.stdout.strip()
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return info


def save_code_snapshot(log_dir: str, verbose: bool = True) -> bool:
    """
    Save a snapshot of changed code files to the log directory.
    
    Args:
        log_dir: Directory where training logs are saved
        verbose: Whether to print status messages
    
    Returns:
        True if snapshot was saved successfully, False otherwise
    """
    if log_dir is None:
        return False
    
    # Get git root
    git_root = get_git_root()
    if git_root is None:
        if verbose:
            print("[Code Snapshot] Git repository not found. Skipping code snapshot.")
        return False
    
    # Create snapshot directory
    snapshot_dir = os.path.join(log_dir, 'code_snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Get changed files
    changed_files = get_changed_files(git_root)
    
    if not changed_files:
        if verbose:
            print("[Code Snapshot] No changed files detected.")
        # Still save git info even if no changes
    else:
        if verbose:
            print(f"[Code Snapshot] Found {len(changed_files)} changed file(s):")
        
        # Copy changed files maintaining directory structure
        for rel_path in changed_files:
            src_path = os.path.join(git_root, rel_path)
            if not os.path.exists(src_path):
                continue
            
            dst_path = os.path.join(snapshot_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            try:
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"  - {rel_path}")
            except Exception as e:
                if verbose:
                    print(f"  - Failed to copy {rel_path}: {e}")
    
    # Save git diff
    git_diff = get_git_diff(git_root)
    if git_diff:
        diff_path = os.path.join(snapshot_dir, 'git_diff.patch')
        with open(diff_path, 'w', encoding='utf-8') as f:
            f.write(git_diff)
        if verbose:
            print(f"[Code Snapshot] Saved git diff to: git_diff.patch")
    
    # Save git info
    git_info = get_git_info(git_root)
    if git_info:
        info_path = os.path.join(snapshot_dir, 'git_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Commit Hash: {git_info.get('commit_hash', 'N/A')}\n")
            f.write(f"Branch: {git_info.get('branch', 'N/A')}\n")
            f.write(f"Commit Message:\n{git_info.get('commit_message', 'N/A')}\n")
        if verbose:
            print(f"[Code Snapshot] Saved git info to: git_info.txt")
    
    if verbose:
        print(f"[Code Snapshot] Code snapshot saved to: {snapshot_dir}")
    
    return True


def copy_all_code_files(log_dir: str, source_dirs: List[str], verbose: bool = True) -> bool:
    """
    Alternative approach: Copy all Python files from specified directories.
    Use this if you want to snapshot all code regardless of git status.
    
    Args:
        log_dir: Directory where training logs are saved
        source_dirs: List of directories to copy (e.g., ['legged_gym', 'scripts'])
        verbose: Whether to print status messages
    
    Returns:
        True if files were copied successfully, False otherwise
    """
    if log_dir is None:
        return False
    
    snapshot_dir = os.path.join(log_dir, 'code_snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)
    
    git_root = get_git_root()
    if git_root is None:
        if verbose:
            print("[Code Snapshot] Git repository not found.")
        return False
    
    copied_count = 0
    for src_dir in source_dirs:
        src_path = os.path.join(git_root, src_dir)
        if not os.path.exists(src_path):
            if verbose:
                print(f"[Code Snapshot] Directory not found: {src_dir}")
            continue
        
        dst_path = os.path.join(snapshot_dir, src_dir)
        
        try:
            # Use shutil to copy entire directory tree
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            
            def ignore_patterns(directory, files):
                """Ignore pycache, pyc files, and other non-code files."""
                ignore = []
                for f in files:
                    if f == '__pycache__' or f.endswith('.pyc') or f.endswith('.pyo'):
                        ignore.append(f)
                return ignore
            
            shutil.copytree(src_path, dst_path, ignore=ignore_patterns)
            copied_count += 1
            if verbose:
                print(f"[Code Snapshot] Copied directory: {src_dir}")
        except Exception as e:
            if verbose:
                print(f"[Code Snapshot] Failed to copy {src_dir}: {e}")
    
    if verbose:
        print(f"[Code Snapshot] Code snapshot saved to: {snapshot_dir}")
    
    return copied_count > 0

