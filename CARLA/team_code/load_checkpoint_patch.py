"""
Patch để load partial checkpoint khi đổi image encoder
Thêm code này vào dd_ppo.py để replace phần load checkpoint

Cách sử dụng:
1. Copy function load_partial_checkpoint() vào dd_ppo.py
2. Thay thế dòng 551 từ:
   agent.load_state_dict(torch.load(args.load_file, map_location=device), strict=True)
   
   Thành:
   load_partial_checkpoint(agent, args.load_file, device, rank)
"""

import torch
import re


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
    print(f'\n{"="*80}')
    print(f'[Rank {rank}] CHECKPOINT LOADING SUMMARY')
    print(f'{"="*80}')
    print(f'✅ Successfully loaded: {len(loaded_keys)}/{len(checkpoint)} parameters')
    print(f'⚠️  Skipped (shape mismatch): {len(skipped_keys)} parameters')
    
    if skipped_keys:
        print(f'\n{"="*80}')
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
        print(f'\n{"="*80}')
        print(f'⚠️  WARNING: Image encoder weights NOT loaded!')
        print(f'   Model will use RANDOM initialization for image encoder')
        print(f'   This is EXPECTED when changing encoder (mobilenet -> resnet50)')
        print(f'{"="*80}\n')
    
    return model


# ============================================================
# HƯỚNG DẪN SỬA FILE dd_ppo.py
# ============================================================
"""
Tìm đoạn code này trong dd_ppo.py (khoảng dòng 545-551):

  start_step = 0
  if args.load_file is not None:
    load_file_name = os.path.basename(args.load_file)
    algo_step = re.findall(r'\d+', load_file_name)
    if len(algo_step) > 0:
      start_step = int(algo_step[0]) + 1  # That step was already finished.
      print('Start training from step:', start_step)
    agent.load_state_dict(torch.load(args.load_file, map_location=device), strict=True)  # <-- DÒNG NÀY

THAY THÀNH:

  start_step = 0
  if args.load_file is not None:
    load_file_name = os.path.basename(args.load_file)
    algo_step = re.findall(r'\d+', load_file_name)
    if len(algo_step) > 0:
      start_step = int(algo_step[0]) + 1  # That step was already finished.
      print('Start training from step:', start_step)
    agent = load_partial_checkpoint(agent, args.load_file, device, rank)  # <-- DÒNG MỚI
"""
