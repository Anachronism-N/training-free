import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# write_video fallback: torchvision >= 0.21 removed write_video from io
try:
    from torchvision.io import write_video
except ImportError:
    def write_video(filename, video_array, fps):
        """Fallback video writer using pyav."""
        import av
        import numpy as np
        video_array = video_array.cpu().numpy()
        if video_array.dtype == np.float32:
            video_array = (video_array * 255).astype(np.uint8)
        T, H, W, C = video_array.shape
        container = av.open(filename, mode='w')
        stream = container.add_stream('h264', rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'fast'}
        for i in range(T):
            frame = av.VideoFrame.from_ndarray(video_array[i], format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")

# --- Structured memory (EpisodicArchive) CLI ---------------------
# The archive is a training-free sidecar ported from Pyramid-Forcing.
# All hyper-parameters are read from env by the pipeline (see
# ``CausalInferencePipeline._init_structured_memory``); the CLI only flips
# the master switch and forwards overrides.  Parameter names mirror the
# PF CLI (without the ``pyramidkv_`` prefix) so experiment matrices stay
# comparable.
parser.add_argument("--structured_memory_enable", action="store_true", default=False,
                    help="Enable the EpisodicArchive sidecar. Mutually exclusive "
                         "with LIFECACHE_ENABLE. When off, the path is bitwise "
                         "equivalent to native Self-Forcing.")
parser.add_argument("--memory_gate", type=float, default=None,
                    help="Fusion gate (0 = native equivalent, >0 = memory readout active).")
parser.add_argument("--archive_max_frames", type=int, default=None,
                    help="Hard budget for the per-layer archive.")
parser.add_argument("--archive_policy", type=str, default=None,
                    choices=("uniform", "coverage"),
                    help="Eviction policy when the archive overflows.")
parser.add_argument("--archive_spatial_stride", type=int, default=None,
                    help="Spatial pooling stride for archived full-frame K/V (default 4).")
parser.add_argument("--top_k_frames", type=int, default=None,
                    help="Top-K frame selection budget (0 = use all eligible).")
parser.add_argument("--recent_exclude_frames", type=int, default=None,
                    help="Exclude the N most-recent archive frames from readout.")
parser.add_argument("--prompt_prior_weight", type=float, default=None,
                    help="Weight for the prompt-similarity prior in [0, 1].")
parser.add_argument("--episode_gate_mode", type=str, default=None,
                    choices=("off", "contrastive_strict", "contrastive_relative", "dual_evidence", "oracle"),
                    help="Historical episode admission mode.")
parser.add_argument("--episode_gate_activation_episode", type=int, default=None,
                    help="Episode id at which the contrastive gate activates.")
parser.add_argument("--oracle_episode_id", type=int, default=None,
                    help="Force-allow a single historical episode (-1 = off).")
parser.add_argument("--trace_enabled", action="store_true", default=False,
                    help="Enable structured-memory trace logging.")
parser.add_argument("--head_routing", type=str, default=None,
                    choices=("static", "confidence_adaptive", "functional_adaptive", "role_evidence", "off"),
                    help="Per-head routing mode (static = all-enabled in SF port).")
parser.add_argument("--retrieval_temperature", type=float, default=None)
parser.add_argument("--confidence_threshold", type=float, default=None)
parser.add_argument("--min_retrieval_margin", type=float, default=None)
parser.add_argument("--max_retrieval_entropy", type=float, default=None)
parser.add_argument("--position_mode", type=str, default=None,
                    choices=("none", "local_grid"))
parser.add_argument("--fusion_mode", type=str, default=None,
                    choices=("residual", "convex"))
parser.add_argument("--warmup_blocks", type=int, default=None)
parser.add_argument("--structured_memory_readout_mode", type=str, default=None,
                    choices=("all", "clean_only", "noisy_only"))
parser.add_argument("--structured_memory_value_mode", type=str, default=None,
                    choices=("full", "spatial_detail"))
parser.add_argument("--structured_memory_control_mode", type=str, default=None,
                    choices=("normal", "shuffled_v", "abstain"))
parser.add_argument("--structured_memory_selection_policy", type=str, default=None,
                    choices=("query", "least_similar", "oldest", "newest"))
parser.add_argument("--structured_memory_selection_scope", type=str, default=None,
                    choices=("shared", "per_head"))
parser.add_argument("--structured_memory_episode_frame_prior_mode", type=str, default=None,
                    choices=("auto", "on", "off"))
parser.add_argument("--structured_memory_routing_sharpness", type=float, default=None)
parser.add_argument("--structured_memory_margin_threshold", type=float, default=None)
parser.add_argument("--structured_memory_query_ema_decay", type=float, default=None)
parser.add_argument("--structured_memory_layer_start", type=int, default=None,
                    help="Inclusive lower layer index for archive+fusion (default 0).")
parser.add_argument("--structured_memory_layer_end", type=int, default=None,
                    help="Exclusive upper layer index for archive+fusion "
                         "(-1 or unset = all layers).")
parser.add_argument("--structured_memory_trace_path", type=str, default=None,
                    help="Path to write JSONL trace of archive commits and "
                         "episode transitions.  Only used when "
                          "--trace_enabled is also passed.")
parser.add_argument("--structured_memory_debug", action="store_true", default=False,
                    help="Print bounded HREM-v2 diagnostics to stdout.")
parser.add_argument("--structured_memory_debug_layers", type=str, default=None,
                    help="Comma-separated transformer layers to debug. Defaults to "
                         "the first and last active structured-memory layers.")
parser.add_argument("--structured_memory_debug_every_blocks", type=int, default=None,
                    help="Emit stdout diagnostics every N generated blocks.")
parser.add_argument("--structured_memory_memory_start_episode", type=int, default=None,
                    help="Disable the memory branch entirely for episodes "
                         "with id < this value (archive commits still "
                          "happen).  Default 0 = active on every episode.")
parser.add_argument("--dual_min_semantic_similarity", type=float, default=None)
parser.add_argument("--dual_min_visual_similarity", type=float, default=None)
parser.add_argument("--dual_min_combined_score", type=float, default=None)
parser.add_argument("--dual_min_episode_margin", type=float, default=None)
parser.add_argument("--dual_visual_head_fraction", type=float, default=None)
parser.add_argument("--dual_allow_disagreement", action="store_true", default=False,
                    help="Ablation: allow semantic/visual cue winners to disagree.")
parser.add_argument("--role_threshold", type=float, default=None)
parser.add_argument("--role_sharpness", type=float, default=None)
args = parser.parse_args()

# --- Forward structured-memory CLI overrides into env -------------------
# The pipeline reads all hyper-parameters from env (see
# ``CausalInferencePipeline._init_structured_memory``).  CLI values, when
# provided, take precedence over the raw environment so users can override
# a wrapper script without editing it.
if args.structured_memory_enable:
    os.environ["STRUCTURED_MEMORY_ENABLE"] = "1"

_CLI_ENV_MAP = {
    "memory_gate": "STRUCTURED_MEMORY_GATE",
    "archive_max_frames": "STRUCTURED_MEMORY_ARCHIVE_MAX_FRAMES",
    "archive_policy": "STRUCTURED_MEMORY_ARCHIVE_POLICY",
    "archive_spatial_stride": "STRUCTURED_MEMORY_SPATIAL_STRIDE",
    "top_k_frames": "STRUCTURED_MEMORY_TOP_K_FRAMES",
    "recent_exclude_frames": "STRUCTURED_MEMORY_RECENT_EXCLUDE_FRAMES",
    "prompt_prior_weight": "STRUCTURED_MEMORY_PROMPT_PRIOR_WEIGHT",
    "episode_gate_mode": "STRUCTURED_MEMORY_EPISODE_GATE_MODE",
    "episode_gate_activation_episode": "STRUCTURED_MEMORY_EPISODE_GATE_ACTIVATION_EPISODE",
    "oracle_episode_id": "STRUCTURED_MEMORY_ORACLE_EPISODE_ID",
    "head_routing": "STRUCTURED_MEMORY_HEAD_ROUTING",
    "retrieval_temperature": "STRUCTURED_MEMORY_RETRIEVAL_TEMPERATURE",
    "confidence_threshold": "STRUCTURED_MEMORY_CONFIDENCE_THRESHOLD",
    "min_retrieval_margin": "STRUCTURED_MEMORY_MIN_RETRIEVAL_MARGIN",
    "max_retrieval_entropy": "STRUCTURED_MEMORY_MAX_RETRIEVAL_ENTROPY",
    "position_mode": "STRUCTURED_MEMORY_POSITION_MODE",
    "fusion_mode": "STRUCTURED_MEMORY_FUSION_MODE",
    "warmup_blocks": "STRUCTURED_MEMORY_WARMUP_BLOCKS",
    "structured_memory_readout_mode": "STRUCTURED_MEMORY_READOUT_MODE",
    "structured_memory_value_mode": "STRUCTURED_MEMORY_VALUE_MODE",
    "structured_memory_control_mode": "STRUCTURED_MEMORY_CONTROL_MODE",
    "structured_memory_selection_policy": "STRUCTURED_MEMORY_SELECTION_POLICY",
    "structured_memory_selection_scope": "STRUCTURED_MEMORY_SELECTION_SCOPE",
    "structured_memory_episode_frame_prior_mode": "STRUCTURED_MEMORY_EPISODE_FRAME_PRIOR_MODE",
    "structured_memory_routing_sharpness": "STRUCTURED_MEMORY_ROUTING_SHARPNESS",
    "structured_memory_margin_threshold": "STRUCTURED_MEMORY_MARGIN_THRESHOLD",
    "structured_memory_query_ema_decay": "STRUCTURED_MEMORY_QUERY_EMA_DECAY",
    "structured_memory_layer_start": "STRUCTURED_MEMORY_LAYER_START",
    "structured_memory_layer_end": "STRUCTURED_MEMORY_LAYER_END",
    "structured_memory_trace_path": "STRUCTURED_MEMORY_TRACE_PATH",
    "structured_memory_debug_layers": "STRUCTURED_MEMORY_DEBUG_LAYERS",
    "structured_memory_debug_every_blocks": "STRUCTURED_MEMORY_DEBUG_EVERY_BLOCKS",
    "structured_memory_memory_start_episode": "STRUCTURED_MEMORY_MEMORY_START_EPISODE",
    "dual_min_semantic_similarity": "STRUCTURED_MEMORY_DUAL_MIN_SEMANTIC_SIMILARITY",
    "dual_min_visual_similarity": "STRUCTURED_MEMORY_DUAL_MIN_VISUAL_SIMILARITY",
    "dual_min_combined_score": "STRUCTURED_MEMORY_DUAL_MIN_COMBINED_SCORE",
    "dual_min_episode_margin": "STRUCTURED_MEMORY_DUAL_MIN_EPISODE_MARGIN",
    "dual_visual_head_fraction": "STRUCTURED_MEMORY_DUAL_VISUAL_HEAD_FRACTION",
    "role_threshold": "STRUCTURED_MEMORY_ROLE_THRESHOLD",
    "role_sharpness": "STRUCTURED_MEMORY_ROLE_SHARPNESS",
}
for cli_name, env_name in _CLI_ENV_MAP.items():
    value = getattr(args, cli_name, None)
    if value is not None:
        os.environ[env_name] = str(value)
if args.trace_enabled:
    os.environ["STRUCTURED_MEMORY_TRACE_ENABLED"] = "1"
if args.structured_memory_debug:
    os.environ["STRUCTURED_MEMORY_DEBUG"] = "1"
if args.dual_allow_disagreement:
    os.environ["STRUCTURED_MEMORY_DUAL_REQUIRE_AGREEMENT"] = "0"

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

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
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
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


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
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

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # --- K-Stability Calibration ---
    # When CALIBRATE_K_PATH is set, save per-layer K tensors from kv_cache1
    # for cross-seed head stability analysis.
    calibrate_k_path = os.environ.get("CALIBRATE_K_PATH")
    if calibrate_k_path and pipeline.kv_cache1 is not None:
        k_data = {}
        for lid, cache in enumerate(pipeline.kv_cache1):
            le = int(cache["local_end_index"].item())
            if le > 0:
                # Take last frame's K, mean across tokens → [12, 128]
                k = cache["k"][:, max(0, le - pipeline.frame_seq_length):le]
                k_data[str(lid)] = k.mean(dim=(0, 1)).cpu()
        torch.save(k_data, calibrate_k_path)
        print(f"[Calibrate] K tensors saved to {calibrate_k_path}")

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
