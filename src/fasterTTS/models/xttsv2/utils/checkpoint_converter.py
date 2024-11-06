import torch
from safetensors.torch import save_file
import os


def convert_checkpoint(pytorch_checkpoint_path, output_dir):
    """
    Convert PyTorch checkpoint to SafeTensors format, mapping weights to GPT2 or XTTS models
    based on specific substrings.

    Args:
        pytorch_checkpoint_path: Path to input PyTorch checkpoint
        output_dir: Directory to save the output SafeTensors files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load PyTorch checkpoint
    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu', weights_only=False) # to avoid warning

    # Initialize dictionaries for different models
    gpt2_weights = {}
    xtts_weights = {}

    # List of substrings to identify GPT2 weights
    gpt2_substrings = [
       'ln_1.weight', 'ln_1.bias', 'attn.c_attn.weight', 'attn.c_attn.bias', 'attn.c_proj.weight',
        'attn.c_proj.bias', 'ln_2.weight', 'ln_2.bias', 'mlp.c_fc.weight',
        'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias', 'ln_f.weight',
        'ln_f.bias', 'mel_head.weight', 'mel_head.bias'

    ]
    ignore_in_check_components = ['mel_embedding.weight', 'mel_pos_embedding.emb.weight']
    # mel_emb -> wte.emb.weight, mel_pos_emb -> wpe.emb.weight

    all_sub_str = gpt2_substrings + ignore_in_check_components
    # Separate weights based on substrings
    for key, tensor in checkpoint['model'].items():
        # Check if any GPT2 substring is in the key
        is_gpt2_weight = any(substring in key for substring in all_sub_str)

        if is_gpt2_weight:
            if 'mel_embedding.weight' in key:
                key = 'gpt.wte.weight'
            elif 'mel_pos_embedding.emb.weight' in key:
                key = 'gpt.wpe.emb.weight'
            elif 'mel_head' in key:
                key = key.replace('gpt.', '')
            else:
                key = key.replace('gpt.gpt.', 'gpt.')
            # Use a modded name for GPT-2 weights
            gpt2_weights[key] = tensor
        elif 'final_norm' in key:
            gpt2_weights[key.replace('gpt.', '')] = tensor
            xtts_weights[key.replace('gpt.', '')] = tensor
        else:
            # All other weights go to XTTS
            xtts_weights[key.replace('gpt.', '')] = tensor

    # Check if all the weights keys are matched
    assert all(any(substr in key for key in gpt2_weights.keys()) for substr in gpt2_substrings), \
        f"Missing substrings: {[substr for substr in gpt2_substrings if not any(substr in key for key in gpt2_weights.keys())]}"

    gpt2_path = os.path.join(output_dir, 'gpt2_model.safetensors')
    save_file(gpt2_weights, gpt2_path)
    print(f"Saved XTTSv2 GPT-2 weights to {gpt2_path}")
    print(f"XTTSv2 GPT-2 weights: {list(gpt2_weights.keys())}")

    # Save XTTS weights if any exist
    if xtts_weights:
        xtts_path = os.path.join(output_dir, 'xtts-v2.safetensors')
        save_file(xtts_weights, xtts_path)
        print(f"Saved XTTSv2 weights to {xtts_path}")
        print(f"XTTSv2 weights: {list(xtts_weights.keys())}")

