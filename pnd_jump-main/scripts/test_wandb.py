#!/usr/bin/env python3
"""
WandB Integration Test Script

This script verifies that WandB is properly installed and configured.

Usage:
    python scripts/test_wandb.py
"""

import sys
import os

def test_wandb_installation():
    """Test if WandB is installed"""
    print("=" * 60)
    print("Test 1: Check WandB Installation")
    print("=" * 60)
    
    try:
        import wandb
        print("✅ WandB is installed")
        print(f"   Version: {wandb.__version__}")
        return True
    except ImportError:
        print("❌ WandB is not installed")
        print("   Please run: pip install wandb")
        return False


def test_wandb_login():
    """Test WandB login status"""
    print("\n" + "=" * 60)
    print("Test 2: Check WandB Login")
    print("=" * 60)
    
    try:
        import wandb
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Logged in to WandB")
        print(f"   Username: {user.get('username', 'unknown')}")
        print(f"   Teams: {user.get('teams', [])}")
        return True
    except Exception as e:
        print("❌ Not logged in to WandB")
        print(f"   Error: {e}")
        print("   Please run: wandb login")
        return False


def test_wandb_integration():
    """Test if WandB integration works"""
    print("\n" + "=" * 60)
    print("Test 3: Test WandB Integration")
    print("=" * 60)
    
    try:
        from legged_gym.utils import WandbLogger, create_wandb_logger_from_config
        print("✅ WandB utilities imported successfully")
        
        # Test creating a logger (in dry-run mode)
        print("\n   Testing WandB logger creation...")
        
        # Create a test logger with disabled mode to avoid actual logging
        os.environ['WANDB_MODE'] = 'disabled'
        
        test_config = {
            'runner': {
                'use_wandb': True,
                'wandb_project': 'test_project',
                'wandb_entity': None,
                'experiment_name': 'test',
                'run_name': 'test_run',
                'wandb_tags': ['test'],
                'resume': False,
            }
        }
        
        logger = create_wandb_logger_from_config(test_config)
        
        if logger:
            print("✅ WandB logger created successfully (dry-run mode)")
        else:
            print("ℹ️  WandB logger not enabled (expected behavior)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import WandB utilities")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  WandB integration test failed")
        print(f"   Error: {e}")
        return False
    finally:
        # Reset environment
        if 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']


def test_config_files():
    """Test if config files have WandB options"""
    print("\n" + "=" * 60)
    print("Test 4: Check Config Files")
    print("=" * 60)
    
    try:
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
        
        # Check if wandb options exist
        runner_cfg = LeggedRobotCfgPPO.runner
        
        has_use_wandb = hasattr(runner_cfg, 'use_wandb')
        has_wandb_project = hasattr(runner_cfg, 'wandb_project')
        has_wandb_entity = hasattr(runner_cfg, 'wandb_entity')
        has_wandb_tags = hasattr(runner_cfg, 'wandb_tags')
        
        if has_use_wandb and has_wandb_project:
            print("✅ Config files contain WandB options")
            print(f"   use_wandb: {runner_cfg.use_wandb}")
            print(f"   wandb_project: {runner_cfg.wandb_project}")
            print(f"   wandb_entity: {runner_cfg.wandb_entity}")
            print(f"   wandb_tags: {runner_cfg.wandb_tags}")
            return True
        else:
            print("❌ Config files missing WandB options")
            return False
            
    except Exception as e:
        print(f"❌ Cannot check config files")
        print(f"   Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 13 + "WandB Integration Test" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Installation", test_wandb_installation()))
    results.append(("Login", test_wandb_login()))
    results.append(("Integration", test_wandb_integration()))
    results.append(("Config Files", test_config_files()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    # Overall result
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! WandB integration is properly configured.")
        print("\nNext Steps:")
        print("  1. Start training:")
        print("     python legged_gym/scripts/train.py --task=adam_lite --wandb")
        print("\n  2. View documentation:")
        print("     Check README or documentation files for more info")
    else:
        print("⚠️  Some tests failed, please check the error messages above.")
        print("\nNeed help?")
        print("  - View documentation for setup instructions")
        print("  - Run: wandb login")
        print("  - Install: pip install -e .")
    
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

