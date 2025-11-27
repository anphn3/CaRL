#!/usr/bin/env python3
"""
Script tự động patch file dd_ppo.py để support partial checkpoint loading
Chạy script này để tự động sửa file dd_ppo.py

Usage:
    python apply_patch_dd_ppo.py /home/anphn3/CaRL/CARLA/team_code/dd_ppo.py
"""

import sys
import os
import shutil
from datetime import datetime


LOAD_PARTIAL_CHECKPOINT_FUNCTION = '''

def load_partial_checkpoint(model, checkpoint_path, device, rank=0):
    """
    Load checkpoint với selective loading - chỉ load những weights khớp shape
    
    Args:
        model: PyTorch model
        checkpoint_path: Path đến checkpoint file
        device: Device để load (cuda/cpu)
        rank: Process rank (để logging)
    """
    if checkpoint_path is None:
        return model
    
    print(f'[Rank {rank}] Loading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get current model state
    model_state = model.state_dict()
    
    # Filter checkpoint: chỉ giữ lại keys có shape khớp
    filtered_checkpoint = {}
    skipped_keys = []
    loaded_keys = []
    
    for key, value in checkpoint.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_checkpoint[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append({
                    'key': key,
                    'checkpoint_shape': value.shape,
                    'model_shape': model_state[key].shape
                })
        else:
            skipped_keys.append({
                'key': key,
                'checkpoint_shape': value.shape,
                'model_shape': 'Key not found in model'
            })
    
    # Load filtered checkpoint
    model.load_state_dict(filtered_checkpoint, strict=False)
    
    # Logging
    print(f'\\n{"="*80}')
    print(f'[Rank {rank}] CHECKPOINT LOADING SUMMARY')
    print(f'{"="*80}')
    print(f'✅ Successfully loaded: {len(loaded_keys)}/{len(checkpoint)} parameters')
    print(f'⚠️  Skipped (shape mismatch): {len(skipped_keys)} parameters')
    
    if skipped_keys:
        print(f'\\n{"="*80}')
        print(f'SKIPPED PARAMETERS (Shape Mismatch):')
        print(f'{"="*80}')
        for item in skipped_keys[:10]:  # Chỉ show 10 đầu tiên
            print(f"  • {item['key']}")
            print(f"    Checkpoint shape: {item['checkpoint_shape']}")
            print(f"    Model shape: {item['model_shape']}")
            print()
        
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more parameters")
    
    # Kiểm tra xem có load được image encoder không
    image_encoder_loaded = any('features_extractor.cnn' in key for key in loaded_keys)
    
    if not image_encoder_loaded:
        print(f'\\n{"="*80}')
        print(f'⚠️  WARNING: Image encoder weights NOT loaded!')
        print(f'   Model will use RANDOM initialization for image encoder')
        print(f'   This is EXPECTED when changing encoder (mobilenet -> resnet50)')
        print(f'{"="*80}\\n')
    
    return model

'''


def patch_dd_ppo(file_path):
    """Patch file dd_ppo.py"""
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    # Backup original file
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'load_partial_checkpoint' in content:
        print("⚠️  File already patched! Skipping...")
        return True
    
    # 1. Add function after imports (after "jsonpickle.set_encoder_options")
    insert_position = content.find("torch.set_num_threads(1)")
    if insert_position == -1:
        print("❌ Error: Could not find insertion point")
        return False
    
    # Insert function before torch.set_num_threads(1)
    content = content[:insert_position] + LOAD_PARTIAL_CHECKPOINT_FUNCTION + '\n' + content[insert_position:]
    
    # 2. Replace load_state_dict line
    old_line = "agent.load_state_dict(torch.load(args.load_file, map_location=device), strict=True)"
    new_line = "agent = load_partial_checkpoint(agent, args.load_file, device, rank)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print("✅ Replaced checkpoint loading line")
    else:
        print("⚠️  Warning: Could not find exact load_state_dict line")
        print("   You may need to manually replace it")
    
    # Write patched file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully patched: {file_path}")
    print(f"{'='*80}")
    print(f"Backup saved to: {backup_path}")
    print(f"\nChanges made:")
    print(f"  1. Added load_partial_checkpoint() function")
    print(f"  2. Changed checkpoint loading to use partial loading")
    print(f"\nYou can now run your training script!")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python apply_patch_dd_ppo.py <path_to_dd_ppo.py>")
        print("\nExample:")
        print("  python apply_patch_dd_ppo.py /home/anphn3/CaRL/CARLA/team_code/dd_ppo.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = patch_dd_ppo(file_path)
    
    sys.exit(0 if success else 1)
