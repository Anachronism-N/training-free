import argparse
import time
import torch
import os
import queue
import threading
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
import imageio.v2 as imageio
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.sampler import DistributedEvalSampler

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(
    description=(
        "Run Pyramid Forcing inference (Self-Forcing + per-head adaptive KV cache) on a "
        "list of prompts, writing one mp4 per (prompt, sample) into --output_folder."
    ),
)
parser.add_argument(
    "--config_path", type=str, required=True,
    help="YAML config (e.g. configs/pyramid-forcing.yaml). Merged "
         "over configs/default_config.yaml.",
)
parser.add_argument(
    "--checkpoint_path", type=str, required=True,
    help="Self-Forcing generator checkpoint (.pt) — e.g. "
         "checkpoints/self_forcing_dmd.pt downloaded from HF.",
)
parser.add_argument(
    "--data_path", type=str, required=True,
    help="Prompt list. .txt = one prompt per line; .csv = columns "
         "(index, prompt); image directory when --i2v is set.",
)
parser.add_argument(
    "--extended_prompt_path", type=str, default=None,
    help="Optional CSV mapping short prompts to extended captions; if "
         "provided, the extended caption is sent to the text encoder.",
)
parser.add_argument(
    "--output_folder", type=str, required=True,
    help="Directory to write generated videos into (created if missing).",
)
parser.add_argument(
    "--num_output_frames", type=int, default=21,
    help="Number of latent frames to generate. The VAE upsamples by 4x in "
         "time, so the default 21 yields ~84 video frames (~5.6s @ 16fps).",
)
parser.add_argument(
    "--i2v", action="store_true",
    help="Image-to-video mode (uses --data_path as an image directory). "
         "T2V is the default. I2V requires a single-process run — multi-GPU "
         "I2V is not supported.",
)
parser.add_argument(
    "--use_ema", action="store_true",
    help="Load EMA generator weights (the `generator_ema` entry in the "
         "checkpoint) instead of the live `generator` weights.",
)
parser.add_argument(
    "--seed", type=int, default=0,
    help="RNG seed; rank N uses seed + N when running under torchrun.",
)
parser.add_argument(
    "--num_samples", type=int, default=1,
    help="Number of independent samples per prompt (different seeds).",
)
parser.add_argument(
    "--save_with_index", action="store_true",
    help="Name output files by prompt index (0000.mp4, 0001.mp4 ...) "
         "instead of by truncated prompt text.",
)
parser.add_argument(
    "--fixed_prefix", type=str, default=None,
    help="If set, save videos as {fixed_prefix}{global_idx:03d}.mp4, "
         "ignoring per-prompt naming.",
)
parser.add_argument(
    "--pyramidkv_history_value_renorm_strength", type=float, default=None,
    help="Blend stale history V channel statistics toward the recent live window (0 disables).",
)
parser.add_argument(
    "--pyramidkv_history_value_recent_frames", type=int, default=None,
    help="Number of recent frames used as live V statistics for history renormalization.",
)
parser.add_argument(
    "--pyramidkv_history_value_gate_lambda", type=float, default=None,
    help="Echo-style discrepancy gate for stale-history V refresh (0 disables).",
)
parser.add_argument(
    "--reseed_per_prompt", action="store_true",
    help="Reset RNG to seed + prompt index before sampling each prompt for fair A/B runs.",
)
parser.add_argument(
    "--pyramidkv_history_value_labels", type=str, default=None,
    help="Comma-separated PF labels to refresh (for example: -1,1). Default: all labels.",
)
parser.add_argument(
    "--pyramidkv_history_value_layer_start", type=int, default=None,
    help="First transformer layer receiving history V refresh (inclusive).",
)
parser.add_argument(
    "--pyramidkv_history_value_layer_end", type=int, default=None,
    help="Last transformer layer receiving history V refresh (exclusive; -1 means all).",
)
parser.add_argument(
    "--pyramidkv_history_value_label_layer_routes", type=str, default=None,
    help="Per-label layer routes, e.g. '-1:10-20,1:10-20,2:0-30'. End is exclusive.",
)
parser.add_argument(
    "--pyramidkv_history_value_moment_mode",
    choices=("full", "variance_only", "mean_only"),
    default=None,
    help="Which live V moments are transported into stale history values.",
)
parser.add_argument(
    "--pyramidkv_history_value_target_frames", type=int, default=None,
    help="Frames used for moment targets; may exceed the untouched recent window.",
)
parser.add_argument(
    "--pyramidkv_history_value_transition_lambda", type=float, default=None,
    help="Suppress history transport when consecutive live moments disagree.",
)
parser.add_argument(
    "--pyramidkv_history_value_max_std_ratio", type=float, default=None,
    help="Maximum variance scale ratio; 0 disables bounding.",
)
parser.add_argument(
    "--pyramidkv_structured_memory", action="store_true",
    help="Enable compressed visual memory with an independent query-conditioned readout.",
)
parser.add_argument("--pyramidkv_structured_memory_budget_frames", type=int, default=None)
parser.add_argument("--pyramidkv_structured_memory_spatial_stride", type=int, default=None)
parser.add_argument("--pyramidkv_structured_memory_local_fusion_distance", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_core_fusion_weight", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_readout_gate", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_retrieval_temperature", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_confidence_threshold", type=float, default=None)
parser.add_argument(
    "--pyramidkv_structured_memory_value_mode",
    choices=("full", "spatial_detail"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_structured_memory_readout_mode",
    choices=("all", "clean_only", "noisy_only"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_structured_memory_storage_mode",
    choices=("compressed", "archive"),
    default=None,
)
parser.add_argument("--pyramidkv_structured_memory_archive_max_frames", type=int, default=None)
parser.add_argument(
    "--pyramidkv_structured_memory_archive_policy",
    choices=("uniform", "coverage"),
    default=None,
)
parser.add_argument("--pyramidkv_structured_memory_top_k_frames", type=int, default=None)
parser.add_argument("--pyramidkv_structured_memory_recent_exclude_frames", type=int, default=None)
parser.add_argument(
    "--pyramidkv_structured_memory_selection_policy",
    choices=("query", "least_similar", "oldest", "newest"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_structured_memory_selection_scope",
    choices=("shared", "per_head"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_structured_memory_fusion_mode",
    choices=("residual", "convex"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_structured_memory_head_labels",
    type=str,
    default=None,
    help="Optional comma-separated PF labels allowed to read archival memory.",
)
parser.add_argument("--pyramidkv_structured_memory_layer_start", type=int, default=None)
parser.add_argument("--pyramidkv_structured_memory_layer_end", type=int, default=None)
parser.add_argument("--pyramidkv_structured_memory_warmup_blocks", type=int, default=None)
parser.add_argument("--pyramidkv_structured_memory_head_routing", type=str, default=None)
parser.add_argument("--pyramidkv_structured_memory_routing_sharpness", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_margin_threshold", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_query_ema_decay", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_min_retrieval_margin", type=float, default=None)
parser.add_argument("--pyramidkv_structured_memory_max_retrieval_entropy", type=float, default=None)
parser.add_argument(
    "--pyramidkv_structured_memory_control_mode",
    choices=("normal", "shuffled_v", "abstain"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_structured_memory_position_mode",
    choices=("none", "local_grid"),
    default=None,
)
parser.add_argument("--pyramidkv_structured_memory_prompt_prior_weight", type=float, default=None)
parser.add_argument(
    "--pyramidkv_cache_transition",
    action="store_true",
    help="Gate clean middle-cache promotion using online K/V reliability.",
)
parser.add_argument(
    "--pyramidkv_cache_transition_mode",
    choices=("audit", "gate", "stagger", "full"),
    default=None,
)
parser.add_argument("--pyramidkv_cache_transition_min_reliability", type=float, default=None)
parser.add_argument("--pyramidkv_cache_transition_min_novelty", type=float, default=None)
parser.add_argument("--pyramidkv_cache_transition_shock_weight", type=float, default=None)
parser.add_argument("--pyramidkv_cache_transition_denoise_weight", type=float, default=None)
parser.add_argument("--pyramidkv_cache_transition_min_interval_blocks", type=int, default=None)
parser.add_argument("--pyramidkv_cache_transition_max_age_blocks", type=int, default=None)
parser.add_argument("--pyramidkv_cache_transition_warmup_blocks", type=int, default=None)
parser.add_argument("--pyramidkv_cache_transition_max_commit_fraction", type=float, default=None)
parser.add_argument("--pyramidkv_cache_transition_stagger_period", type=int, default=None)
parser.add_argument(
    "--pyramidkv_cache_transition_branches",
    choices=("both", "cond", "uncond"),
    default=None,
)
parser.add_argument("--pyramidkv_cache_transition_trace_path", type=str, default=None)
parser.add_argument("--pyramidkv_cache_transition_debug", action="store_true")
parser.add_argument(
    "--pyramidkv_probecache",
    action="store_true",
    help="Enable counterfactually profiled dual-lifecycle direct middle slots.",
)
parser.add_argument(
    "--pyramidkv_head_config_path",
    type=str,
    default=None,
    help="Override both PyramidKV label and policy CSV paths.",
)
parser.add_argument(
    "--pyramidkv_probecache_mode",
    choices=("audit", "persistent", "reactive", "full"),
    default=None,
)
parser.add_argument("--pyramidkv_probecache_archive_max_frames", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_persistent_top_k", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_reactive_top_k", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_recent_exclude_frames", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_reactive_horizon_frames", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_min_reliability", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_min_similarity", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_min_margin", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_max_entropy", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_retrieval_temperature", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_min_frame_spacing", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_prompt_weight", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_prompt_min_similarity", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_prompt_switch_threshold", type=float, default=None)
parser.add_argument("--pyramidkv_probecache_persistent_label", type=int, default=None)
parser.add_argument(
    "--pyramidkv_probecache_reactive_labels",
    type=str,
    default=None,
    help="Comma-separated labels assigned to the reactive lifecycle.",
)
parser.add_argument("--pyramidkv_probecache_layer_start", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_layer_end", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_trace_path", type=str, default=None)
parser.add_argument("--pyramidkv_probecache_trace_selection_stride", type=int, default=None)
parser.add_argument("--pyramidkv_probecache_debug", action="store_true")
parser.add_argument(
    "--pyramidkv_probecache_profile_recent_only",
    action="store_true",
    help="Profiler intervention: attend only to the native recent window.",
)
parser.add_argument("--probecache_profile_output", type=str, default=None)
parser.add_argument(
    "--probecache_profile_kind",
    choices=("prompt", "history"),
    default=None,
)
parser.add_argument("--probecache_profile_pair_id", type=str, default=None)
parser.add_argument("--probecache_profile_side", type=str, default=None)
parser.add_argument("--dynamic_cfg_enabled", action="store_true", default=False)
parser.add_argument("--dynamic_cfg_min_scale", type=float, default=1.0)
parser.add_argument("--dynamic_cfg_max_scale", type=float, default=5.0)
parser.add_argument("--per_head_cfg_enabled", action="store_true", default=False)
parser.add_argument("--per_head_cfg_min_scale", type=float, default=1.0)
parser.add_argument("--per_head_cfg_max_scale", type=float, default=5.0)
parser.add_argument("--few_step_cfg_enabled", action="store_true", default=False)
parser.add_argument("--few_step_cfg_mode", choices=("fixed", "dynamic"), default="fixed")
parser.add_argument("--few_step_cfg_scale", type=float, default=3.0)
parser.add_argument("--few_step_cfg_min_scale", type=float, default=1.5)
parser.add_argument("--few_step_cfg_max_scale", type=float, default=3.5)
args = parser.parse_args()


def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    return any(isinstance(v, torch.Tensor) for v in obj.values())


def _normalize_generator_state_dict(state_dict: dict) -> dict:
    sd = state_dict

    # Common case: whole-model state_dict with generator namespace.
    if any(k.startswith("model.generator.") for k in sd.keys()):
        sd = {k[len("model.generator."):]: v for k, v in sd.items() if k.startswith("model.generator.")}
    elif any(k.startswith("generator.") for k in sd.keys()):
        sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}

    # Remove known wrappers added by distributed/checkpoint wrappers.
    for prefix in ("module.", "_orig_mod.", "_checkpoint_wrapped_module."):
        if sd and all(k.startswith(prefix) for k in sd.keys()):
            sd = {k[len(prefix):]: v for k, v in sd.items()}

    return sd


def _extract_generator_state_dict(checkpoint, use_ema: bool):
    if not isinstance(checkpoint, dict):
        return checkpoint

    preferred = ["generator_ema", "ema", "model_ema"] if use_ema else []
    preferred += ["generator", "model", "state_dict"]

    for key in preferred:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            nested = checkpoint[key]
            if key == "state_dict":
                if use_ema:
                    for k in ("generator_ema", "ema", "model_ema"):
                        if k in nested and isinstance(nested[k], dict):
                            return nested[k]
                for k in ("generator", "model"):
                    if k in nested and isinstance(nested[k], dict):
                        return nested[k]
            return nested

    if _looks_like_state_dict(checkpoint):
        return checkpoint

    for v in checkpoint.values():
        if _looks_like_state_dict(v):
            return v

    top_keys = list(checkpoint.keys())
    raise KeyError(
        f"Unable to locate generator weights in checkpoint. Top-level keys: {top_keys[:20]}"
    )


# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # Pass device_id explicitly so NCCL doesn't fall back to GPU 0 for the
    # initial barrier (which warns: "using GPU 0 to perform barrier as
    # devices used by this process are currently unknown").
    dist.init_process_group(backend='nccl', device_id=device)
    gpu = device  # override module-level gpu (evaluated at import as cuda:0)
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

if args.i2v and world_size > 1:
    raise SystemExit(
        "I2V mode does not support distributed inference yet. Re-run without "
        "torchrun (single process) or drop --i2v for T2V mode."
    )

if local_rank == 0:
    if world_size > 1:
        print(f"Running on {world_size} GPUs (rank {local_rank} = {device}).")
    else:
        print(f"Running on a single GPU ({device}); skipping NCCL initialization.")

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.backends.cudnn.benchmark = True

# Force-load the JIT C++/CUDA extension up front so it doesn't pad iter 0
# with ~60s of compile time on a cold ~/.cache/torch_extensions. On warm
# cache this is a ~3s no-op. Done before the inference loop so the
# steady-state progress bar reflects actual model latency.
if local_rank == 0:
    _jit_t0 = time.time()
    print("Loading CUDA extension (first run compiles ~60s; cached run ~3s)...", flush=True)
try:
    from pyramidkv import _ops as _pyramidkv_ops
    _pyramidkv_ops._ensure_loaded()
except Exception as _e:
    if local_rank == 0:
        print(f"[warn] CUDA extension preload failed: {_e!r}", flush=True)
if local_rank == 0:
    print(f"CUDA extension ready in {time.time() - _jit_t0:.1f}s.", flush=True)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.pyramidkv_history_value_renorm_strength is not None:
    config.pyramidkv_history_value_renorm_strength = args.pyramidkv_history_value_renorm_strength
if args.pyramidkv_history_value_recent_frames is not None:
    config.pyramidkv_history_value_recent_frames = args.pyramidkv_history_value_recent_frames
if args.pyramidkv_history_value_gate_lambda is not None:
    config.pyramidkv_history_value_gate_lambda = args.pyramidkv_history_value_gate_lambda
if args.pyramidkv_history_value_labels is not None:
    config.pyramidkv_history_value_labels = [
        int(value.strip()) for value in args.pyramidkv_history_value_labels.split(",") if value.strip()
    ]
if args.pyramidkv_history_value_layer_start is not None:
    config.pyramidkv_history_value_layer_start = args.pyramidkv_history_value_layer_start
if args.pyramidkv_history_value_layer_end is not None:
    config.pyramidkv_history_value_layer_end = args.pyramidkv_history_value_layer_end
if args.pyramidkv_history_value_label_layer_routes is not None:
    routes = {}
    for item in args.pyramidkv_history_value_label_layer_routes.split(","):
        label_text, bounds_text = item.split(":", maxsplit=1)
        start_text, end_text = bounds_text.split("-", maxsplit=1)
        routes[int(label_text)] = [int(start_text), int(end_text)]
    config.pyramidkv_history_value_label_layer_routes = routes
if args.pyramidkv_history_value_moment_mode is not None:
    config.pyramidkv_history_value_moment_mode = args.pyramidkv_history_value_moment_mode
if args.pyramidkv_history_value_target_frames is not None:
    config.pyramidkv_history_value_target_frames = args.pyramidkv_history_value_target_frames
if args.pyramidkv_history_value_transition_lambda is not None:
    config.pyramidkv_history_value_transition_lambda = args.pyramidkv_history_value_transition_lambda
if args.pyramidkv_history_value_max_std_ratio is not None:
    config.pyramidkv_history_value_max_std_ratio = args.pyramidkv_history_value_max_std_ratio
if args.pyramidkv_structured_memory:
    config.pyramidkv_structured_memory_enabled = True
if args.pyramidkv_structured_memory_head_labels is not None:
    config.pyramidkv_structured_memory_head_labels = [
        int(value.strip())
        for value in args.pyramidkv_structured_memory_head_labels.split(",")
        if value.strip()
    ]
# Copy valid few-step CFG CLI overrides into the pipeline config.
config.few_step_cfg_enabled = bool(args.few_step_cfg_enabled)
config.few_step_cfg_mode = str(args.few_step_cfg_mode)
config.few_step_cfg_scale = float(args.few_step_cfg_scale)
config.few_step_cfg_min_scale = float(args.few_step_cfg_min_scale)
config.few_step_cfg_max_scale = float(args.few_step_cfg_max_scale)

for name in (
    "budget_frames",
    "spatial_stride",
    "local_fusion_distance",
    "core_fusion_weight",
    "readout_gate",
    "retrieval_temperature",
    "confidence_threshold",
    "value_mode",
    "readout_mode",
    "storage_mode",
    "archive_max_frames",
    "archive_policy",
    "top_k_frames",
    "recent_exclude_frames",
    "selection_policy",
    "selection_scope",
    "fusion_mode",
    "layer_start",
    "layer_end",
    "warmup_blocks",
    "head_routing",
    "routing_sharpness",
    "margin_threshold",
    "query_ema_decay",
    "min_retrieval_margin",
    "max_retrieval_entropy",
    "control_mode",
    "position_mode",
    "prompt_prior_weight",
):
    value = getattr(args, f"pyramidkv_structured_memory_{name}")
    if value is not None:
        setattr(config, f"pyramidkv_structured_memory_{name}", value)

if args.pyramidkv_cache_transition:
    config.pyramidkv_cache_transition_enabled = True
if args.pyramidkv_cache_transition_debug:
    config.pyramidkv_cache_transition_debug = True
for name in (
    "mode",
    "min_reliability",
    "min_novelty",
    "shock_weight",
    "denoise_weight",
    "min_interval_blocks",
    "max_age_blocks",
    "warmup_blocks",
    "max_commit_fraction",
    "stagger_period",
    "branches",
    "trace_path",
):
    value = getattr(args, f"pyramidkv_cache_transition_{name}")
    if value is not None:
        setattr(config, f"pyramidkv_cache_transition_{name}", value)

if args.pyramidkv_probecache:
    config.pyramidkv_probecache_enabled = True
if args.pyramidkv_head_config_path is not None:
    config.pyramidkv_config_path = args.pyramidkv_head_config_path
    config.pyramidkv_policy_csv_path = args.pyramidkv_head_config_path
if args.pyramidkv_probecache_debug:
    config.pyramidkv_probecache_debug = True
if args.pyramidkv_probecache_profile_recent_only:
    config.pyramidkv_probecache_profile_recent_only = True
if args.pyramidkv_probecache_reactive_labels is not None:
    config.pyramidkv_probecache_reactive_labels = [
        int(value.strip())
        for value in args.pyramidkv_probecache_reactive_labels.split(",")
        if value.strip()
    ]
for name in (
    "mode",
    "archive_max_frames",
    "persistent_top_k",
    "reactive_top_k",
    "recent_exclude_frames",
    "reactive_horizon_frames",
    "min_reliability",
    "min_similarity",
    "min_margin",
    "max_entropy",
    "retrieval_temperature",
    "min_frame_spacing",
    "prompt_weight",
    "prompt_min_similarity",
    "prompt_switch_threshold",
    "persistent_label",
    "layer_start",
    "layer_end",
    "trace_path",
    "trace_selection_stride",
):
    value = getattr(args, f"pyramidkv_probecache_{name}")
    if value is not None:
        setattr(config, f"pyramidkv_probecache_{name}", value)

if args.probecache_profile_output:
    os.environ["PROBECACHE_PROFILE"] = "1"

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    generator_state_dict = _extract_generator_state_dict(checkpoint, use_ema=args.use_ema)
    generator_state_dict = _normalize_generator_state_dict(generator_state_dict)
    pipeline.generator.load_state_dict(generator_state_dict, strict=True)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedEvalSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def _write_video(path: str, frames: torch.Tensor, fps: int = 16) -> None:
    video = frames.detach().cpu()
    if video.dtype != torch.uint8:
        video = torch.clamp(video, 0, 255).to(torch.uint8)
    try:
        write_video(path, video, fps=fps)
    except Exception:
        imageio.mimwrite(path, video.numpy(), fps=fps)


class AsyncVideoWriter:
    def __init__(self, max_pending: int = 4):
        self._queue = queue.Queue(maxsize=max_pending)
        self._error = None
        self._thread = threading.Thread(target=self._worker, daemon=False)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, frames, fps = item
                _write_video(path, frames, fps=fps)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def write(self, path: str, frames: torch.Tensor, fps: int = 16) -> None:
        self._raise_if_failed()
        self._queue.put((path, frames, fps))
        self._raise_if_failed()

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Background video writer failed") from self._error


video_writer = AsyncVideoWriter()
# Per-prompt timing is reported via tqdm.write() AFTER each prompt's inner
# block bars finish, so we don't fight the (block N/7 - X/21) tqdm bars
# emitted from CausalInferencePipeline. The JIT compile cost is paid up
# front (see "Loading CUDA extension" above) so iter 0 vs steady-state is
# only the small (~5-10s) CUDA lazy-init + flash-attn autotune delta.
_loop_t0 = time.time() if local_rank == 0 else None
try:
    with torch.inference_mode():
        for i, batch_data in enumerate(dataloader):
            idx = batch_data['idx'].item()
            if args.reseed_per_prompt:
                set_seed(args.seed + int(idx))

            # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
            # Unpack the batch data for convenience
            if isinstance(batch_data, dict):
                batch = batch_data
            elif isinstance(batch_data, list):
                batch = batch_data[0]  # First (and only) item in the batch

            num_generated_frames = 0  # Number of generated (latent) frames

            if args.i2v:
                # For image-to-video, batch contains image and caption
                prompt = batch['prompts'][0]  # Get caption from batch
                prompts = [prompt] * args.num_samples

                # Process the image
                image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

                # Encode the input image as the first latent
                initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
                initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

                sampled_noise = torch.randn(
                    [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
                )
            else:
                # For text-to-video, batch is just the text prompt
                prompt = batch['prompts'][0]
                extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
                if extended_prompt is not None:
                    prompts = [extended_prompt] * args.num_samples
                else:
                    prompts = [prompt] * args.num_samples
                initial_latent = None

                sampled_noise = torch.randn(
                    [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
                )

            if os.environ.get("HEAD_DIAGNOSTIC", "0") == "1":
                try:
                    from wan.modules.attention.core import set_diagnostic_prompt_id
                    set_diagnostic_prompt_id(i)
                except Exception as e:
                    print(f"[DIAG] Failed to set prompt id: {e}")
            if args.probecache_profile_output:
                from wan.modules.attention.core import set_probecache_profile_prompt_id
                set_probecache_profile_prompt_id(i)

            # Generate video frames
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=initial_latent,
                low_memory=low_memory,
                profile=os.environ.get("ADAHEAD_PROFILE", "0") == "1",
            )
            num_generated_frames += latents.shape[1]

            # Final output video — clamp+uint8 on GPU, non-blocking D2H overlaps with cache clear
            video = torch.clamp(
                255.0 * rearrange(video, 'b t c h w -> b t h w c'), 0, 255,
            ).to(dtype=torch.uint8).to(device='cpu', non_blocking=True)

            # Clear VAE cache (overlaps with D2H transfer)
            pipeline.vae.model.clear_cache()

            # Ensure D2H completes before writing
            torch.cuda.current_stream().synchronize()

            # Save the video if the current prompt is not a dummy prompt
            if idx < num_prompts:
                model = "regular" if not args.use_ema else "ema"
                for seed_idx in range(args.num_samples):
                    # All processes save their videos
                    if args.fixed_prefix is not None:
                        # 固定前缀 + 数据集全局索引，多卡下天然唯一
                        filename = f"{args.fixed_prefix}{idx:03d}.mp4"
                        output_path = os.path.join(args.output_folder, filename)
                    elif args.save_with_index:
                        output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
                    else:
                        output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
                    video_writer.write(output_path, video[seed_idx], fps=16)

            # Per-prompt status line, printed AFTER the inner block bars
            # so they don't fight for the cursor. tqdm.write() also handles
            # the rare case where a downstream bar is still active. The
            # rate uses the steady-state average (excludes iter 0 once
            # i >= 1 since that's when JIT/lazy-init overhead is amortized
            # to its own one-time cost above).
            if local_rank == 0:
                done = i + 1
                elapsed = time.time() - _loop_t0
                avg = elapsed / done
                remaining = max(num_prompts - done, 0)
                eta = remaining * avg
                tqdm.write(
                    f"[{done}/{num_prompts}] elapsed={elapsed/60:.1f}m  "
                    f"avg={avg:.1f}s/prompt  eta={eta/60:.1f}m"
                )
finally:
    video_writer.close()

# Save head diagnostic report if enabled
if os.environ.get("HEAD_DIAGNOSTIC", "0") == "1":
    try:
        from wan.modules.attention.core import save_diagnostic_report
        diag_path = os.path.join(args.output_folder, "..", "diagnostic_report.json")
        save_diagnostic_report(diag_path)
    except Exception as e:
        print(f"[DIAG] Failed to save report: {e}")

if args.probecache_profile_output:
    try:
        from wan.modules.attention.core import save_probecache_profile
        profile_path = args.probecache_profile_output.format(
            rank=int(os.environ.get("RANK", "0")),
            pid=os.getpid(),
        )
        save_probecache_profile(
            profile_path,
            {
                "kind": args.probecache_profile_kind,
                "pair_id": args.probecache_profile_pair_id,
                "side": args.probecache_profile_side,
                "seed": int(args.seed),
                "data_path": os.path.abspath(args.data_path),
                "recent_only": bool(args.pyramidkv_probecache_profile_recent_only),
            },
        )
    except Exception as e:
        print(f"[ProbeCacheProfile] Failed to save profile: {e}")

# Persist memory-admission statistics for every run.
try:
    caches = list(getattr(pipeline, "kv_cache1", []) or [])
    total_calls = sum(int(getattr(cache, "_memory_readout_calls", 0)) for cache in caches)
    total_heads = sum(int(getattr(cache, "_memory_readout_heads", 0)) for cache in caches)
    accepted_heads = sum(int(getattr(cache, "_memory_accepted_heads", 0)) for cache in caches)
    if total_calls > 0:
        import json
        admission_report = {
            "readout_calls": total_calls,
            "evaluated_heads": total_heads,
            "accepted_heads": accepted_heads,
            "acceptance_rate": accepted_heads / max(total_heads, 1),
        }
        with open(os.path.join(args.output_folder, "memory_admission_report.json"), "w") as f:
            json.dump(admission_report, f, indent=2)
        print(f"[MEMORY] admission report: {admission_report}")
except Exception as e:
    print(f"[MEMORY] Failed to save admission report: {e}")

if getattr(pipeline, "few_step_cfg_enabled", False):
    import json
    scales = list(getattr(pipeline, "_few_step_cfg_scales", []))
    cfg_report = {
        "enabled": True,
        "mode": pipeline.few_step_cfg_mode,
        "num_scales": len(scales),
        "min": min(scales) if scales else None,
        "max": max(scales) if scales else None,
        "mean": sum(scales) / len(scales) if scales else None,
        "scales": scales,
    }
    with open(os.path.join(args.output_folder, "cfg_report.json"), "w") as f:
        json.dump(cfg_report, f, indent=2)
    print(f"[CFG] scale report: {cfg_report}")
