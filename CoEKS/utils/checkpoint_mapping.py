"""
Utility functions for mapping checkpoint parameters from MVMoEPolicy to CoEKSPolicy.
"""
import torch
from typing import Dict, Any, List, Tuple


def find_num_experts_from_target(target_state_dict: Dict[str, torch.Tensor]) -> int:
    """Find the number of experts from target model state dict."""
    max_expert_idx = -1
    for key in target_state_dict.keys():
        if "experts." in key:
            # Extract expert index, e.g., "experts.4" -> 4
            parts = key.split("experts.")
            if len(parts) > 1:
                try:
                    expert_idx = int(parts[1].split(".")[0])
                    max_expert_idx = max(max_expert_idx, expert_idx)
                except ValueError:
                    continue
    if max_expert_idx >= 0:
        return max_expert_idx + 1  # 0-indexed, so add 1
    return 5  # Default to 5 if not found


def map_mvmoe_to_coeks_state_dict(
    checkpoint_state_dict: Dict[str, torch.Tensor],
    target_model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Map parameters from MVMoEPolicy checkpoint to CoEKSPolicy structure.
    
    Args:
        checkpoint_state_dict: State dict from MVMoEPolicy checkpoint
        target_model_state_dict: State dict from CoEKSPolicy model (for reference)
    
    Returns:
        Mapped state dict compatible with CoEKSPolicy
    """
    mapped_state_dict = {}
    
    # Filter checkpoint state dict to only include policy parameters
    checkpoint_policy_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("policy."):
            # Remove "policy." prefix
            new_key = key[7:]  # len("policy.") = 7
            checkpoint_policy_dict[new_key] = value
        elif not key.startswith("_") and not key.startswith("state_dict"):
            # Sometimes checkpoint might have direct parameter names
            checkpoint_policy_dict[key] = value
    
    # If checkpoint_state_dict is already the policy state dict (no "policy." prefix)
    if not checkpoint_policy_dict:
        checkpoint_policy_dict = checkpoint_state_dict
    
    # Find number of experts from target model
    num_experts = find_num_experts_from_target(target_model_state_dict)
    print(f"Detected {num_experts} experts in target model")
    print(f"Found {len(checkpoint_policy_dict)} parameters in checkpoint")
    print(f"Target model has {len(target_model_state_dict)} parameters")
    
    # Print some sample keys for debugging
    if len(checkpoint_policy_dict) > 0:
        print(f"\nSample checkpoint keys (first 5):")
        for i, key in enumerate(list(checkpoint_policy_dict.keys())[:5]):
            print(f"  {i+1}. {key}")
    if len(target_model_state_dict) > 0:
        print(f"\nSample target model keys (first 5):")
        for i, key in enumerate(list(target_model_state_dict.keys())[:5]):
            print(f"  {i+1}. {key}")
    
    # Track unmapped keys for debugging
    unmapped_keys = []
    
    # Mapping rules - try direct match first, then try alternative mappings
    for ckpt_key, ckpt_value in checkpoint_policy_dict.items():
        mapped_keys = []
        
        # Strategy: Try direct match first, then try alternative mappings
        
        # 1. Try direct match first (most common case)
        if ckpt_key in target_model_state_dict:
            mapped_keys.append((ckpt_key, ckpt_value))
        
        # 2. If direct match failed, try alternative mappings
        elif "encoder.init_embedding.project_global_feats" in ckpt_key:
            # Check if target uses Task (experts) or Linear
            # Try Task.experts mapping first
            if "weight" in ckpt_key or "bias" in ckpt_key:
                suffix = ckpt_key.split("project_global_feats")[1]  # e.g., ".weight"
                for expert_idx in range(num_experts):
                    new_key = f"encoder.init_embedding.project_global_feats.experts.{expert_idx}{suffix}"
                    if new_key in target_model_state_dict:
                        mapped_keys.append((new_key, ckpt_value))
        
        elif "encoder.init_embedding.project_customers_feats" in ckpt_key:
            # Check if target uses Task (experts) or Linear
            if "weight" in ckpt_key or "bias" in ckpt_key:
                suffix = ckpt_key.split("project_customers_feats")[1]  # e.g., ".weight"
                for expert_idx in range(num_experts):
                    new_key = f"encoder.init_embedding.project_customers_feats.experts.{expert_idx}{suffix}"
                    if new_key in target_model_state_dict:
                        mapped_keys.append((new_key, ckpt_value))
        
        # 3. Try alternative layer naming patterns
        elif "encoder.net.layers" in ckpt_key or "encoder.layers" in ckpt_key:
            # Try replacing "net.layers" with "layers" or vice versa
            alt_key = ckpt_key.replace("net.layers", "layers")
            if alt_key in target_model_state_dict:
                mapped_keys.append((alt_key, ckpt_value))
            else:
                alt_key = ckpt_key.replace("layers", "net.layers")
                if alt_key in target_model_state_dict:
                    mapped_keys.append((alt_key, ckpt_value))
        
        # 4. Try removing/adding common prefixes
        elif not ckpt_key.startswith("encoder") and not ckpt_key.startswith("decoder"):
            # Try adding encoder/decoder prefix if missing
            if f"encoder.{ckpt_key}" in target_model_state_dict:
                mapped_keys.append((f"encoder.{ckpt_key}", ckpt_value))
            elif f"decoder.{ckpt_key}" in target_model_state_dict:
                mapped_keys.append((f"decoder.{ckpt_key}", ckpt_value))
        
        # Add all mapped keys to state dict
        found_match = False
        for mapped_key, mapped_value in mapped_keys:
            if mapped_key not in mapped_state_dict:
                # Check shape compatibility
                if mapped_key in target_model_state_dict:
                    target_shape = target_model_state_dict[mapped_key].shape
                    ckpt_shape = mapped_value.shape
                    if target_shape == ckpt_shape:
                        mapped_state_dict[mapped_key] = mapped_value.clone()
                        found_match = True
                    else:
                        print(f"Warning: Shape mismatch for {ckpt_key} -> {mapped_key}: "
                              f"target {target_shape} vs checkpoint {ckpt_shape}")
                else:
                    # Mapped key doesn't exist in target - this shouldn't happen
                    pass
        
        if not found_match:
            unmapped_keys.append(ckpt_key)
    
    # Check for missing target parameters and initialize them BEFORE printing summary
    missing_target_keys = set(target_model_state_dict.keys()) - set(mapped_state_dict.keys())
    
    # Separate bias and non-bias parameters
    missing_bias_keys = [key for key in missing_target_keys if "bias" in key]
    missing_non_bias_keys = [key for key in missing_target_keys if "bias" not in key]
    
    # Initialize missing bias parameters to zeros
    if missing_bias_keys:
        print(f"\nInitializing {len(missing_bias_keys)} missing bias parameters to zeros:")
        for key in sorted(missing_bias_keys):
            if key in target_model_state_dict:
                target_param = target_model_state_dict[key]
                mapped_state_dict[key] = torch.zeros_like(target_param)
                print(f"  - {key} (shape: {target_param.shape})")
    
    # Report non-bias parameters that will use random initialization
    if missing_non_bias_keys:
        print(f"\nWarning: {len(missing_non_bias_keys)} non-bias parameters will use random initialization:")
        for key in sorted(missing_non_bias_keys):
            if key in target_model_state_dict:
                target_param = target_model_state_dict[key]
                print(f"  - {key} (shape: {target_param.shape})")
    
    # Now print summary with updated counts
    final_mapped_count = len(mapped_state_dict)
    total_target_count = len(target_model_state_dict)
    
    print(f"\nMapping Summary:")
    print(f"  - Successfully mapped: {final_mapped_count} parameters")
    print(f"  - Target model total: {total_target_count} parameters")
    print(f"  - Coverage: {final_mapped_count / total_target_count * 100:.1f}%")
    
    if unmapped_keys:
        print(f"\nWarning: {len(unmapped_keys)} checkpoint parameters could not be mapped")
        if len(unmapped_keys) <= 20:
            print(f"Unmapped keys:")
            for key in unmapped_keys:
                print(f"  - {key}")
        else:
            print(f"First 20 unmapped keys:")
            for key in unmapped_keys[:20]:
                print(f"  - {key}")
    
    return mapped_state_dict


def save_converted_checkpoint(
    policy: torch.nn.Module,
    save_path: str,
    hyper_parameters: dict = None,
):
    """
    Save a converted checkpoint that can be directly loaded with CoEKS.load_from_checkpoint.
    
    Args:
        policy: The CoEKSPolicy model with loaded parameters
        save_path: Path to save the checkpoint
        hyper_parameters: Optional hyperparameters to save in the checkpoint
    """
    import os
    try:
        import lightning as L
        lightning_version = L.__version__
    except ImportError:
        try:
            import pytorch_lightning as pl
            lightning_version = pl.__version__
        except ImportError:
            lightning_version = "2.0.0"  # fallback version
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Get the final state dict from the model
    final_state_dict = policy.state_dict()
    
    # Default hyperparameters
    if hyper_parameters is None:
        hyper_parameters = {
            "embed_dim": 128,
            "num_encoder_layers": 6,
            "num_heads": 8,
            "normalization": "instance",
            "feedforward_hidden": 512,
            "env_name": "mtvrp",
            "num_experts": 5,
            "routing_method": "input_choice",
            "routing_level": "node",
            "topk": 2,
            "CoE_loc": ["enc0"],
            "hierarchical_gating": False,
        }
    
    # Create a checkpoint compatible with PyTorch Lightning format
    converted_checkpoint = {
        "pytorch-lightning_version": lightning_version,
        "state_dict": {"policy." + k: v for k, v in final_state_dict.items()},
        "hyper_parameters": hyper_parameters,
        "epoch": 0,
        "global_step": 0,
        "callbacks": {},
    }
    
    torch.save(converted_checkpoint, save_path)
    print(f"Successfully saved converted checkpoint to {save_path}")
    print(f"  - Total parameters: {len(final_state_dict)}")
    return save_path


def load_checkpoint_with_mapping(
    checkpoint_path: str,
    target_model: torch.nn.Module,
    strict: bool = False,
) -> torch.nn.Module:
    """
    Load checkpoint and map parameters to target model structure.
    
    Args:
        checkpoint_path: Path to checkpoint file
        target_model: Target model to load parameters into
        strict: Whether to strictly enforce that all parameters are matched
    
    Returns:
        Model with loaded parameters
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint_state_dict = checkpoint["state_dict"]
        else:
            checkpoint_state_dict = checkpoint
    else:
        checkpoint_state_dict = checkpoint
    
    # Get target model state dict
    target_state_dict = target_model.state_dict()
    
    # Map parameters
    mapped_state_dict = map_mvmoe_to_coeks_state_dict(
        checkpoint_state_dict, target_state_dict
    )
    
    # Load mapped parameters
    missing_keys, unexpected_keys = target_model.load_state_dict(
        mapped_state_dict, strict=strict
    )
    
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")
    
    return target_model

