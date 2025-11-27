#!/usr/bin/env python3
import sys

FUNCTION_CODE = '''

def load_partial_checkpoint(model, checkpoint_path, device, rank=0):
    """Load checkpoint với selective loading - chỉ load những weights khớp shape"""
    if checkpoint_path is None:
        return model
    
    print(f'[Rank {rank}] Loading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    
    filtered_checkpoint = {}
    skipped_keys = []
    loaded_keys = []
    
    for key, value in checkpoint.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_checkpoint[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append({'key': key, 'checkpoint_shape': value.shape, 'model_shape': model_state[key].shape})
    
    model.load_state_dict(filtered_checkpoint, strict=False)
    
    print(f'\\n{"="*80}')
    print(f'[Rank {rank}] CHECKPOINT LOADING SUMMARY')
    print(f'{"="*80}')
    print(f'✅ Successfully loaded: {len(loaded_keys)}/{len(checkpoint)} parameters')
    print(f'⚠️  Skipped (shape mismatch): {len(skipped_keys)} parameters')
    
    if skipped_keys:
        print(f'\\nSKIPPED PARAMETERS (Shape Mismatch):')
        for item in skipped_keys[:10]:
            print(f"  • {item['key']}: {item['checkpoint_shape']} -> {item['model_shape']}")
    
    image_encoder_loaded = any('features_extractor.cnn' in key for key in loaded_keys)
    if not image_encoder_loaded:
        print(f'\\n⚠️  WARNING: Image encoder weights NOT loaded (expected when changing encoder)')
    
    return model

'''

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    content = f.read()

if 'def load_partial_checkpoint' not in content:
    marker = "torch.set_num_threads(1)"
    parts = content.split(marker, 1)
    new_content = parts[0] + marker + FUNCTION_CODE + parts[1]
    
    with open(sys.argv[1] + '.backup_fix', 'w', encoding='utf-8') as f:
        f.write(content)
    
    with open(sys.argv[1], 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Đã thêm function thành công!")
else:
    print("✅ Function đã tồn tại!")
