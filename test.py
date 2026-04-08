import argparse
import os
import pickle
import time
import warnings
import pandas as pd
import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from CoEKS.data.utils import get_dataloader
from CoEKS.envs import MTVRPEnv
from CoEKS.models import Base
from CoEKS.models.baselines.CoEKS import CoEKS
from CoEKS.models.baselines.CoEKS.policy import CoEKSPolicy

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


# def run_test(
def test(
    policy,
    td,
    env,
    num_augment=8,
    augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
    num_starts=None,
    device="cuda",
):

    costs_bks = td.get("costs_bks", None)
    # print(costs_bks)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy
            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    # print(torch.abs(costs_bks).mean())
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})  # Dummy value

            return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        # default="all",
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint",
    )

    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:7")
    else:
        device = torch.device("cpu")

    if opts.datasets is not None:
        data_paths = opts.datasets.split(",")
    else:
        # list recursively all npz files in data/
        data_paths = []
        for root, _, files in os.walk("data"):
            for file in files:
                # print(file)
                if "test" not in root:
                    continue
                if file.endswith(".npz"):
                    if opts.remove_mixed_backhaul and "mb" in root:
                        continue
                    # if name in 50 or 100, append
                    if str(opts.size) in file:
                        if file == "50.npz" or file == "100.npz":
                            data_paths.append(os.path.join(root, file))
        assert len(data_paths) > 0, "No datasets found. Check the data directory."
        data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
        print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

    # Load model
    BaseLitModule = CoEKS
    env = MTVRPEnv()

    # print(opts.checkpoint)
    checkpoint = torch.load(opts.checkpoint, map_location="cpu")
    
    # Check if checkpoint has required Lightning metadata
    if "pytorch-lightning_version" not in checkpoint:
        # If missing metadata, try to load state_dict directly
        print("Warning: Checkpoint missing Lightning metadata, loading state_dict directly...")
        
        # Create policy first
        policy = CoEKSPolicy(
            embed_dim=128,
            num_encoder_layers=6,
            num_heads=8,
            normalization="instance",
            feedforward_hidden=512,
            env_name="mtvrp",
            num_experts=5,
            routing_method="input_choice",
            routing_level="node",
            topk=2,
            CoE_loc=["enc0"],
            hierarchical_gating=False,
        )
        
        # Extract state dict (handle both formats)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove "policy." prefix if present
        if any(k.startswith("policy.") for k in state_dict.keys()):
            state_dict = {k[7:] if k.startswith("policy.") else k: v for k, v in state_dict.items()}
        
        # Load state dict into policy
        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys")
        
        # Create model with policy
        model = BaseLitModule(env=env, policy=policy)
    else:
        # Standard Lightning checkpoint loading
        model = BaseLitModule.load_from_checkpoint(
            opts.checkpoint, map_location="cpu", strict=False
        )

    policy = model.policy.to(device).eval()  # Use mixed precision if supported

    results = {}

    gap_columns = []
    for dataset in tqdm(data_paths):

        print(f"Loading {dataset}")
        td_test = env.load_data(dataset)  # this also adds the bks cost
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        for batch in dataloader:
            td_test = env.reset(batch).to(device)
            o = test(policy, td_test, env, device=device)
            res.append(o)
        out = {}
        out["max_aug_reward"] = torch.cat([o["max_aug_reward"] for o in res])
        out["gap_to_bks"] = torch.cat([o["gap_to_bks"] for o in res])

        gap_values = out["gap_to_bks"].cpu().numpy().tolist()
        gap_columns.append(gap_values)

        inference_time = time.time() - start

        dataset_name = dataset.split("/")[-3].split(".")[0].upper()
        print(
                f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.4f} | Gap: {out['gap_to_bks'].mean().item():.4f}% | Inference time: {inference_time:.4f} s"
        )

        if results.get(dataset_name, None) is None:
            results[dataset_name] = {}
        results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
        results[dataset_name]["gap"] = out["gap_to_bks"].mean().item()
        results[dataset_name]["inference_time"] = inference_time

    max_length = max(len(col) for col in gap_columns)
    df = pd.DataFrame(
        {f"Dataset_{i + 1}": col + [None] * (max_length - len(col)) for i, col in enumerate(gap_columns)}
    )
    output_file = f"results/main/{opts.size}/gap_to_bks.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)


    if opts.save_results:
        checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
        savedir = f"results/main/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))
