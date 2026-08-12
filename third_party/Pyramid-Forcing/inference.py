import argparse
import csv
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
    "--start_idx", type=int, default=0,
    help="First dataset prompt index to process (inclusive).",
)
parser.add_argument(
    "--end_idx", type=int, default=None,
    help="Last dataset prompt index to process (exclusive).",
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
    "--prompt_stride", type=int, default=1,
    help="Process every Nth prompt (1=all). Combined with --prompt_offset for rank-based sharding.",
)
parser.add_argument(
    "--prompt_offset", type=int, default=0,
    help="Starting offset for --prompt_stride. rank R uses offset=R.",
)
parser.add_argument(
    "--skip_existing", action="store_true",
    help="Skip prompts whose output video already exists.",
)
parser.add_argument(
    "--skip_video_decode",
    action="store_true",
    help=(
        "Return latent outputs without VAE decoding or video writing. This is "
        "restricted to cache-compatibility profiling runs."
    ),
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
    "--pyramidkv_prompt_warmup",
    action="store_true",
    help=(
        "Temporarily hide PF history from selected prompt-role heads while the "
        "underlying cache continues to warm."
    ),
)
parser.add_argument("--pyramidkv_prompt_warmup_blocks", type=int, default=None)
parser.add_argument("--pyramidkv_prompt_warmup_release_span", type=int, default=None)
parser.add_argument(
    "--pyramidkv_prompt_warmup_mode",
    choices=("middle", "history"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_prompt_warmup_shield_labels",
    type=str,
    default=None,
    help="Comma-separated prompt-role labels whose history is shielded.",
)
parser.add_argument("--pyramidkv_prompt_warmup_layer_start", type=int, default=None)
parser.add_argument("--pyramidkv_prompt_warmup_layer_end", type=int, default=None)
parser.add_argument("--pyramidkv_prompt_warmup_trace_path", type=str, default=None)
parser.add_argument("--pyramidkv_prompt_warmup_debug", action="store_true")
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
parser.add_argument(
    "--pyramidkv_cache_transition_role_conditioning",
    action="store_true",
    help=(
        "Use a separate persistent/reactive label matrix to schedule middle-cache "
        "writes without changing PyramidKV read policies."
    ),
)
parser.add_argument(
    "--pyramidkv_cache_transition_role_config_path",
    type=str,
    default=None,
    help=(
        "Orthogonal lifecycle-role CSV used by role-conditioned transition "
        "and/or prompt warmup."
    ),
)
parser.add_argument(
    "--pyramidkv_cache_transition_persistent_label",
    type=int,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_reactive_labels",
    type=str,
    default=None,
    help="Comma-separated transition-role labels treated as reactive.",
)
parser.add_argument(
    "--pyramidkv_cache_transition_persistent_min_novelty_scale",
    type=float,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_reactive_min_novelty_scale",
    type=float,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_persistent_max_age_blocks",
    type=int,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_reactive_max_age_blocks",
    type=int,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_reactive_utility_bias",
    type=float,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_role_layer_start",
    type=int,
    default=None,
)
parser.add_argument(
    "--pyramidkv_cache_transition_role_layer_end",
    type=int,
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
    "--pyramidkv_binary_responsive_policy",
    choices=(
        "cyclic",
        "cyclic_sink3",
        "merge",
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
        "recent",
        "recent8",
    ),
    default=None,
    help=(
        "Decouple binary role labels from PF labels: label 1 uses Anchor "
        "stride; label -1 uses the selected responsive middle policy."
    ),
)
parser.add_argument(
    "--pyramidkv_binary_stable_policy",
    choices=("stride", "hybrid"),
    default="stride",
    help=(
        "Middle policy for label 1 in a binary map. Hybrid uses two stride "
        "and two phase-aligned slots under the same four-frame read budget."
    ),
)
parser.add_argument(
    "--pyramidkv_history_polarity",
    action="store_true",
    help=(
        "Route neutral labels 10/11 as History-Supportive/Suppressive heads. "
        "This path is independent of PF's reserved -1/1/2 class semantics."
    ),
)
parser.add_argument(
    "--pyramidkv_history_support_policy",
    choices=(
        "stride",
        "hybrid",
        "cyclic",
        "recent8",
        "landmark",
        "motion_pair",
        "motion_pair1",
        "landmark_motion",
        "retrieval",
        "retrieval2",
        "retrieval1",
        "retrieval1_age24",
        "retrieval1_motion1_age24",
        "prototype",
        "prototype2",
        "reservoir",
        "reservoir2_motion1",
        "reservoir2_freshmotion1",
        "reservoir2_statemotion1",
        "reservoir2_freshmotion4",
        "reservoir2_statemotion1_strict",
        "reservoir2_directionmatch1",
        "reservoir2_directionfresh1",
        "reservoir2_dirstaletie003",
        "reservoir2_dirstaletie005",
        "reservoir2_multiscaledir1",
        "reservoir2_multiscalemotion1",
        "reservoir2_multiscalepareto1",
        "reservoir2_multiscaleconsensus1",
        "reservoir2_multiscalequeryweighted1",
        "reservoir2_multiscalebottleneck1",
        "reservoir2_staterankmotion1",
        "reservoir2_deficitstaterankmotion1",
        "reservoir2_deficitquery1",
        "reservoir2_deficitbaseline1",
        "reservoir4_multiscalemotion1",
        "profile_anchor",
        "recent8_exact",
        "snapshot",
        "snapshot2",
        "sparse75",
    ),
    default="hybrid",
)
parser.add_argument(
    "--pyramidkv_history_suppress_policy",
    choices=(
        "merge",
        "cyclic",
        "cyclic_sink3",
        "motion",
        "motion_cyclic",
        "cyclic_motion1",
        "recent",
        "recent5",
        "recent8",
        "recent8_sink1",
        "landmark",
        "motion_pair",
        "motion_pair1",
        "landmark_motion",
        "retrieval",
        "retrieval2",
        "retrieval1",
        "retrieval1_age24",
        "retrieval1_motion1_age24",
        "prototype",
        "prototype2",
        "reservoir",
        "reservoir2_motion1",
        "reservoir2_freshmotion1",
        "reservoir2_statemotion1",
        "reservoir2_freshmotion4",
        "reservoir2_statemotion1_strict",
        "reservoir2_directionmatch1",
        "reservoir2_directionfresh1",
        "reservoir2_dirstaletie003",
        "reservoir2_dirstaletie005",
        "reservoir2_multiscaledir1",
        "reservoir2_multiscalemotion1",
        "reservoir2_multiscalepareto1",
        "reservoir2_multiscaleconsensus1",
        "reservoir2_multiscalequeryweighted1",
        "reservoir2_multiscalebottleneck1",
        "reservoir2_staterankmotion1",
        "reservoir2_deficitstaterankmotion1",
        "reservoir2_deficitquery1",
        "reservoir2_deficitbaseline1",
        "reservoir4_multiscalemotion1",
        "profile_anchor",
        "recent8_exact",
        "snapshot",
        "snapshot2",
        "sparse75",
    ),
    default="merge",
)
parser.add_argument(
    "--pyramidkv_history_budget_profile",
    choices=("default", "sink3_extra", "sink3_budget9", "profile_exact8"),
    default="default",
    help=(
        "Explicit sink/middle/recent allocation for history-polarity routes. "
        "sink3_budget9 is valid only for landmark/motion_pair1; "
        "profile_exact8 is valid only for profile_anchor/recent8_exact."
    ),
)
parser.add_argument(
    "--pyramidkv_motion_event_top_k",
    type=int,
    default=None,
    help="Layer-shared motion-event frames selected per clean block.",
)
parser.add_argument(
    "--pyramidkv_motion_event_sample_tokens",
    type=int,
    default=None,
    help="Maximum sampled spatial tokens per frame for V-change scoring.",
)
parser.add_argument(
    "--pyramidkv_semantic_retrieval_min_similarity",
    type=float,
    default=None,
    help="Absolute cosine floor for confidence-gated role retrieval.",
)
parser.add_argument(
    "--pyramidkv_semantic_retrieval_min_margin",
    type=float,
    default=None,
    help="Required top-1 minus top-2 cosine margin for role retrieval.",
)
parser.add_argument(
    "--pyramidkv_semantic_retrieval_abstain",
    action="store_true",
    help=(
        "Return no retrieved middle frame when its similarity or margin "
        "gate fails; sink/recent and any paired motion memory remain active."
    ),
)
parser.add_argument(
    "--pyramidkv_scene_cache",
    action="store_true",
    help=(
        "For segmented prompts, archive/restore stride memory by scene and "
        "clear scene-local middle strategies at each boundary."
    ),
)
parser.add_argument(
    "--pyramidkv_scene_cache_match_mode",
    choices=("idf", "embedding"),
    default=None,
)
parser.add_argument(
    "--pyramidkv_scene_cache_similarity_threshold",
    type=float,
    default=None,
)
parser.add_argument(
    "--pyramidkv_scene_cache_manual_ids",
    type=str,
    default=None,
    help="Optional comma-separated canonical scene ids, e.g. 0,1,0.",
)
parser.add_argument(
    "--pyramidkv_scene_cache_max_scenes",
    type=int,
    default=None,
)
parser.add_argument(
    "--pyramidkv_scene_cache_bridge_recent_frames",
    type=int,
    default=None,
)
parser.add_argument(
    "--pyramidkv_scene_cache_trace_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--pyramidkv_scene_cache_debug",
    action="store_true",
)
parser.add_argument(
    "--pyramidkv_pf_extended_recent_ablation",
    choices=("anchor", "wave", "veil"),
    default=None,
    help=(
        "For a map where the selected PF class is encoded as label 3, "
        "replace its middle cache with approximately token-budget-matched "
        "additional recent frames."
    ),
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
parser.add_argument(
    "--head_qk_profile_output",
    type=str,
    default=None,
    help="Save bounded frame-level pre-softmax QK traces for head discovery.",
)
parser.add_argument(
    "--head_qk_profile_kind",
    choices=("prompt", "temporal", "middle_relative"),
    default="prompt",
)
parser.add_argument("--head_qk_profile_pair_id", type=str, default=None)
parser.add_argument("--head_qk_profile_side", type=str, default=None)
parser.add_argument(
    "--head_qk_profile_update_modes",
    default="noisy,clean",
    help="Comma-separated cache update modes to record.",
)
parser.add_argument(
    "--head_qk_profile_branches",
    default="cond,uncond",
    help="Comma-separated CFG branches to record.",
)
parser.add_argument(
    "--head_qk_profile_max_calls_per_location",
    type=int,
    default=4,
)
parser.add_argument(
    "--head_qk_profile_max_records_per_layer_branch",
    type=int,
    default=256,
)
parser.add_argument(
    "--cache_compat_profile_output",
    type=str,
    default=None,
    help=(
        "Save v173 equal-budget cache-operator compatibility records. "
        "Requires the reservoir4_multiscalemotion1 profiling composition."
    ),
)
parser.add_argument(
    "--pyramidkv_cache_compatibility_policy",
    action="store_true",
    help=(
        "Route neutral labels 20/21/22 as Recent/Coverage/Episode using "
        "the v173 equal-budget cache operators."
    ),
)
parser.add_argument(
    "--cache_compat_profile_kind",
    type=str,
    default="moviegen128_discovery",
)
parser.add_argument(
    "--cache_compat_profile_call_indices",
    type=str,
    default="0,2",
)
parser.add_argument(
    "--cache_compat_profile_ar_stride",
    type=int,
    default=3,
)
parser.add_argument(
    "--cache_compat_profile_query_stride",
    type=int,
    default=8,
)
parser.add_argument(
    "--cache_compat_profile_min_frame",
    type=int,
    default=12,
)
parser.add_argument(
    "--cache_compat_profile_chunk_offsets",
    type=str,
    default="0",
)
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

if args.pyramidkv_prompt_warmup:
    config.pyramidkv_prompt_warmup_enabled = True
if args.pyramidkv_prompt_warmup_debug:
    config.pyramidkv_prompt_warmup_debug = True
if args.pyramidkv_prompt_warmup_shield_labels is not None:
    config.pyramidkv_prompt_warmup_shield_labels = [
        int(value.strip())
        for value in args.pyramidkv_prompt_warmup_shield_labels.split(",")
        if value.strip()
    ]
for name in (
    "blocks",
    "release_span",
    "mode",
    "layer_start",
    "layer_end",
    "trace_path",
):
    value = getattr(args, f"pyramidkv_prompt_warmup_{name}")
    if value is not None:
        setattr(config, f"pyramidkv_prompt_warmup_{name}", value)
if (
    config.pyramidkv_prompt_warmup_enabled
    and not (
        args.pyramidkv_cache_transition_role_config_path
        or getattr(config, "pyramidkv_cache_transition_role_config_path", None)
    )
):
    parser.error(
        "--pyramidkv_prompt_warmup requires "
        "--pyramidkv_cache_transition_role_config_path"
    )

if args.pyramidkv_cache_transition:
    config.pyramidkv_cache_transition_enabled = True
if args.pyramidkv_cache_transition_debug:
    config.pyramidkv_cache_transition_debug = True
if args.pyramidkv_cache_transition_role_conditioning:
    config.pyramidkv_cache_transition_role_conditioning = True
if args.pyramidkv_cache_transition_reactive_labels is not None:
    config.pyramidkv_cache_transition_reactive_labels = [
        int(value.strip())
        for value in args.pyramidkv_cache_transition_reactive_labels.split(",")
        if value.strip()
    ]
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
    "role_config_path",
    "persistent_label",
    "persistent_min_novelty_scale",
    "reactive_min_novelty_scale",
    "persistent_max_age_blocks",
    "reactive_max_age_blocks",
    "reactive_utility_bias",
    "role_layer_start",
    "role_layer_end",
    "trace_path",
):
    value = getattr(args, f"pyramidkv_cache_transition_{name}")
    if value is not None:
        setattr(config, f"pyramidkv_cache_transition_{name}", value)
if (
    config.pyramidkv_cache_transition_role_conditioning
    and not config.pyramidkv_cache_transition_role_config_path
):
    parser.error(
        "--pyramidkv_cache_transition_role_conditioning requires "
        "--pyramidkv_cache_transition_role_config_path"
    )

if args.pyramidkv_probecache:
    config.pyramidkv_probecache_enabled = True
if args.pyramidkv_head_config_path is not None:
    config.pyramidkv_config_path = args.pyramidkv_head_config_path
    config.pyramidkv_policy_csv_path = args.pyramidkv_head_config_path
selected_policy_overrides = sum(
    (
        args.pyramidkv_binary_responsive_policy is not None,
        args.pyramidkv_history_polarity,
        args.pyramidkv_cache_compatibility_policy,
        args.pyramidkv_pf_extended_recent_ablation is not None,
    )
)
if selected_policy_overrides > 1:
    parser.error(
        "binary, history-polarity, cache-compatibility, and PF "
        "extended-recent policy overrides are mutually exclusive"
    )
if args.pyramidkv_binary_responsive_policy is not None:
    from pyramidkv.policy_overrides import binary_head_policy_overrides

    policy_overrides = binary_head_policy_overrides(
        args.pyramidkv_binary_stable_policy,
        args.pyramidkv_binary_responsive_policy,
    )
    for field_name, field_value in policy_overrides.items():
        setattr(config, field_name, field_value)
    print(
        "[BinaryPolicyOverride] "
        f"stable={args.pyramidkv_binary_stable_policy} "
        f"responsive={args.pyramidkv_binary_responsive_policy} "
        "sink_recent=role_specific",
        flush=True,
    )
if args.pyramidkv_history_polarity:
    if args.pyramidkv_head_config_path is None:
        parser.error(
            "--pyramidkv_history_polarity requires "
            "--pyramidkv_head_config_path with neutral labels 10/11"
        )
    try:
        with open(
            args.pyramidkv_head_config_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            history_rows = [
                [int(value.strip()) for value in row]
                for row in csv.reader(handle)
                if row
            ]
    except (OSError, ValueError) as error:
        parser.error(f"invalid history-polarity head map: {error}")
    if len(history_rows) != 30 or any(
        len(row) != 12 for row in history_rows
    ):
        parser.error(
            "history-polarity head map must be a complete 30x12 matrix"
        )
    history_labels = {
        value for row in history_rows for value in row
    }
    if not history_labels.issubset({10, 11}):
        parser.error(
            "history-polarity head map must contain only "
            "neutral labels 10/11 (both not required for ablation controls)"
        )
    from pyramidkv.policy_overrides import (
        HISTORY_SUPPORT_LABEL,
        HISTORY_SUPPRESS_LABEL,
        history_polarity_policy_overrides,
    )

    policy_overrides = history_polarity_policy_overrides(
        args.pyramidkv_history_support_policy,
        args.pyramidkv_history_suppress_policy,
        capacity=int(config.pyramidkv_default_capacity or 32760),
        budget_profile=args.pyramidkv_history_budget_profile,
    )
    for field_name, field_value in policy_overrides.items():
        setattr(config, field_name, field_value)
    if args.pyramidkv_semantic_retrieval_min_similarity is not None:
        value = float(args.pyramidkv_semantic_retrieval_min_similarity)
        if not -1.0 <= value <= 1.0:
            parser.error(
                "--pyramidkv_semantic_retrieval_min_similarity must be "
                "within [-1, 1]"
            )
        config.pyramidkv_semantic_retrieval_min_similarity = value
    if args.pyramidkv_semantic_retrieval_min_margin is not None:
        value = float(args.pyramidkv_semantic_retrieval_min_margin)
        if not 0.0 <= value <= 2.0:
            parser.error(
                "--pyramidkv_semantic_retrieval_min_margin must be "
                "within [0, 2]"
            )
        config.pyramidkv_semantic_retrieval_min_margin = value
    config.pyramidkv_semantic_retrieval_abstain = bool(
        args.pyramidkv_semantic_retrieval_abstain
    )
    print(
        "[HistoryPolarityPolicy] "
        f"support_label={HISTORY_SUPPORT_LABEL} "
        f"suppress_label={HISTORY_SUPPRESS_LABEL} "
        f"support={args.pyramidkv_history_support_policy} "
        f"suppress={args.pyramidkv_history_suppress_policy} "
        f"budget={args.pyramidkv_history_budget_profile} "
        f"counts=10:{sum(row.count(10) for row in history_rows)},"
        f"11:{sum(row.count(11) for row in history_rows)} "
        f"support_sink={policy_overrides['pyramidkv_label_sink_frames_map'][str(HISTORY_SUPPORT_LABEL)]} "
        f"suppress_sink={policy_overrides['pyramidkv_label_sink_frames_map'][str(HISTORY_SUPPRESS_LABEL)]} "
        f"support_recent={policy_overrides['pyramidkv_label_recent_frames_map'][str(HISTORY_SUPPORT_LABEL)]} "
        f"suppress_recent={policy_overrides['pyramidkv_label_recent_frames_map'][str(HISTORY_SUPPRESS_LABEL)]} "
        f"retrieval_abstain={bool(config.pyramidkv_semantic_retrieval_abstain)} "
        f"retrieval_min_similarity={float(getattr(config, 'pyramidkv_semantic_retrieval_min_similarity', -0.25)):.4f} "
        f"retrieval_min_margin={float(getattr(config, 'pyramidkv_semantic_retrieval_min_margin', 0.0)):.4f} "
        "legacy_pf_labels=false exclusive_owner=true",
        flush=True,
    )
if args.pyramidkv_cache_compatibility_policy:
    if args.pyramidkv_head_config_path is None:
        parser.error(
            "--pyramidkv_cache_compatibility_policy requires "
            "--pyramidkv_head_config_path"
        )
    try:
        with open(
            args.pyramidkv_head_config_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            compatibility_rows = [
                [int(value.strip()) for value in row]
                for row in csv.reader(handle)
                if row
            ]
    except (OSError, ValueError) as error:
        parser.error(f"invalid cache-compatibility head map: {error}")
    if len(compatibility_rows) != 30 or any(
        len(row) != 12 for row in compatibility_rows
    ):
        parser.error(
            "cache-compatibility head map must be a complete 30x12 matrix"
        )
    compatibility_labels = {
        value for row in compatibility_rows for value in row
    }
    if not compatibility_labels.issubset({20, 21, 22}):
        parser.error(
            "cache-compatibility head map may contain only labels 20/21/22"
        )
    from pyramidkv.policy_overrides import (
        CACHE_COMPAT_COVERAGE_LABEL,
        CACHE_COMPAT_EPISODE_LABEL,
        CACHE_COMPAT_RECENT_LABEL,
        cache_compatibility_policy_overrides,
    )

    policy_overrides = cache_compatibility_policy_overrides(
        capacity=int(config.pyramidkv_default_capacity or 32760),
    )
    for field_name, field_value in policy_overrides.items():
        setattr(config, field_name, field_value)
    print(
        "[CacheCompatibilityPolicy] "
        f"recent={CACHE_COMPAT_RECENT_LABEL}:"
        f"{sum(row.count(CACHE_COMPAT_RECENT_LABEL) for row in compatibility_rows)} "
        f"coverage={CACHE_COMPAT_COVERAGE_LABEL}:"
        f"{sum(row.count(CACHE_COMPAT_COVERAGE_LABEL) for row in compatibility_rows)} "
        f"episode={CACHE_COMPAT_EPISODE_LABEL}:"
        f"{sum(row.count(CACHE_COMPAT_EPISODE_LABEL) for row in compatibility_rows)} "
        "budget=9FFE owner=HeadComposition",
        flush=True,
    )
if args.pyramidkv_pf_extended_recent_ablation is not None:
    from pyramidkv.policy_overrides import pf_class_extended_recent_overrides

    policy_overrides = pf_class_extended_recent_overrides(
        args.pyramidkv_pf_extended_recent_ablation
    )
    for field_name, field_value in policy_overrides.items():
        setattr(config, field_name, field_value)
    print(
        "[PFClassExtendedRecentAblation] "
        f"target={args.pyramidkv_pf_extended_recent_ablation} "
        "replacement=label3 sink=native "
        f"recent={policy_overrides['pyramidkv_label_recent_frames_map']['3']} "
        "middle=none",
        flush=True,
    )
if args.pyramidkv_motion_event_top_k is not None:
    if args.pyramidkv_motion_event_top_k <= 0:
        parser.error("--pyramidkv_motion_event_top_k must be positive")
    config.motion_event_top_k = int(args.pyramidkv_motion_event_top_k)
if args.pyramidkv_motion_event_sample_tokens is not None:
    if args.pyramidkv_motion_event_sample_tokens <= 0:
        parser.error(
            "--pyramidkv_motion_event_sample_tokens must be positive"
        )
    config.motion_event_sample_tokens = int(
        args.pyramidkv_motion_event_sample_tokens
    )
if args.pyramidkv_scene_cache:
    config.pyramidkv_scene_cache_enabled = True
if args.pyramidkv_scene_cache_debug:
    config.pyramidkv_scene_cache_debug = True
for name in (
    "match_mode",
    "similarity_threshold",
    "max_scenes",
    "bridge_recent_frames",
    "trace_path",
):
    value = getattr(args, f"pyramidkv_scene_cache_{name}")
    if value is not None:
        setattr(config, f"pyramidkv_scene_cache_{name}", value)
if args.pyramidkv_scene_cache_manual_ids is not None:
    try:
        config.pyramidkv_scene_cache_manual_ids = [
            int(value.strip())
            for value in args.pyramidkv_scene_cache_manual_ids.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("--pyramidkv_scene_cache_manual_ids must be integers")
if getattr(config, "pyramidkv_scene_cache_enabled", False):
    if not getattr(config, "pyramidkv_composition_owns_dynamic", False):
        parser.error(
            "--pyramidkv_scene_cache requires an exclusive composition "
            "policy such as --pyramidkv_history_polarity"
        )
    if int(config.pyramidkv_scene_cache_max_scenes) <= 0:
        parser.error("--pyramidkv_scene_cache_max_scenes must be positive")
    if int(config.pyramidkv_scene_cache_bridge_recent_frames) < 0:
        parser.error(
            "--pyramidkv_scene_cache_bridge_recent_frames must be non-negative"
        )
    print(
        "[SceneCacheConfig] "
        f"match={config.pyramidkv_scene_cache_match_mode} "
        f"threshold={float(config.pyramidkv_scene_cache_similarity_threshold):.4f} "
        f"manual_ids={getattr(config, 'pyramidkv_scene_cache_manual_ids', None)} "
        f"max_scenes={int(config.pyramidkv_scene_cache_max_scenes)} "
        f"bridge_recent={int(config.pyramidkv_scene_cache_bridge_recent_frames)} "
        "owner=HeadComposition",
        flush=True,
    )
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
if args.head_qk_profile_output:
    from wan.modules.attention.head_profile import enable_head_qk_profile

    enable_head_qk_profile(
        num_layers=30,
        frame_seq_length=int(
            getattr(config, "pyramidkv_frame_seq_length", 1560)
        ),
        num_heads=12,
        max_calls_per_location=args.head_qk_profile_max_calls_per_location,
        max_records_per_layer_branch=(
            args.head_qk_profile_max_records_per_layer_branch
        ),
        update_modes=(
            value.strip()
            for value in args.head_qk_profile_update_modes.split(",")
        ),
        branches=(
            value.strip()
            for value in args.head_qk_profile_branches.split(",")
        ),
    )
if args.cache_compat_profile_output:
    if args.probecache_profile_output or args.head_qk_profile_output:
        parser.error(
            "cache compatibility profiling cannot share a run with other "
            "attention profilers"
        )
    if not args.pyramidkv_history_polarity or (
        args.pyramidkv_history_support_policy
        != "reservoir4_multiscalemotion1"
        or args.pyramidkv_history_suppress_policy
        != "reservoir4_multiscalemotion1"
    ):
        parser.error(
            "cache compatibility profiling requires history-polarity with "
            "both routes set to reservoir4_multiscalemotion1"
        )
    if args.cache_compat_profile_ar_stride <= 0:
        parser.error("--cache_compat_profile_ar_stride must be positive")
    if args.cache_compat_profile_query_stride <= 0:
        parser.error("--cache_compat_profile_query_stride must be positive")
    if args.cache_compat_profile_min_frame < 0:
        parser.error("--cache_compat_profile_min_frame must be non-negative")
    if not bool(getattr(config, "use_adaptive_pyramidkv", False)):
        parser.error("cache compatibility profiling requires AdaptiveKVCache")
    if not bool(getattr(config, "sink_grid_decoupling", False)):
        parser.error("cache compatibility profiling requires sink-grid decoupling")
    config.pyramidkv_cache_compat_profile_enabled = True
    config.pyramidkv_cache_compat_profile_recent_frames = 8
    os.environ["CACHE_COMPAT_PROFILE"] = "1"
    os.environ["CACHE_COMPAT_PROFILE_CALL_INDICES"] = (
        args.cache_compat_profile_call_indices
    )
    os.environ["CACHE_COMPAT_PROFILE_AR_STRIDE"] = str(
        args.cache_compat_profile_ar_stride
    )
    os.environ["CACHE_COMPAT_PROFILE_QUERY_STRIDE"] = str(
        args.cache_compat_profile_query_stride
    )
    os.environ["CACHE_COMPAT_PROFILE_MIN_FRAME"] = str(
        args.cache_compat_profile_min_frame
    )
    os.environ["CACHE_COMPAT_PROFILE_CHUNK_OFFSETS"] = (
        args.cache_compat_profile_chunk_offsets
    )
    os.environ["PYRAMIDKV_DISABLE_M6_FASTPATH"] = "1"
    print(
        "[CacheCompatProfileConfig] active=recent8 "
        "candidates=coverage4,episode2+motion_pair reference=union "
        f"calls={args.cache_compat_profile_call_indices} "
        f"ar_stride={args.cache_compat_profile_ar_stride} "
        f"query_stride={args.cache_compat_profile_query_stride} "
        f"min_frame={args.cache_compat_profile_min_frame} "
        f"chunk_offsets={args.cache_compat_profile_chunk_offsets} "
        f"skip_video_decode={bool(args.skip_video_decode)}",
        flush=True,
    )
if args.skip_video_decode and not args.cache_compat_profile_output:
    parser.error("--skip_video_decode requires --cache_compat_profile_output")
if args.skip_video_decode and not hasattr(config, "denoising_step_list"):
    parser.error("--skip_video_decode is only implemented for few-step inference")

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
if not args.skip_video_decode or args.i2v:
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
if not 0 <= args.start_idx < num_prompts:
    raise ValueError(
        f"--start_idx must be in [0, {num_prompts}), got {args.start_idx}"
    )
if args.end_idx is not None and not args.start_idx < args.end_idx <= num_prompts:
    raise ValueError(
        "--end_idx must be greater than --start_idx and no larger than "
        f"the dataset size ({num_prompts})"
    )
print(
    f"Prompt index range: [{args.start_idx}, "
    f"{args.end_idx if args.end_idx is not None else num_prompts})"
)

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
cache_profile_path = None
cache_profile_completed_prompt_ids = set()
if args.cache_compat_profile_output:
    from wan.modules.attention.cache_compat_profile import (
        resume_cache_compatibility_profile,
    )

    cache_profile_path = args.cache_compat_profile_output.format(
        rank=int(os.environ.get("RANK", "0")),
        pid=os.getpid(),
    )
    cache_profile_completed_prompt_ids = resume_cache_compatibility_profile(
        cache_profile_path
    )


def _cache_compatibility_metadata():
    return {
        "kind": args.cache_compat_profile_kind,
        "seed": int(args.seed),
        "data_path": os.path.abspath(args.data_path),
        "num_output_frames": int(args.num_output_frames),
        "call_indices": args.cache_compat_profile_call_indices,
        "ar_stride": int(args.cache_compat_profile_ar_stride),
        "query_stride": int(args.cache_compat_profile_query_stride),
        "min_frame": int(args.cache_compat_profile_min_frame),
        "chunk_offsets": args.cache_compat_profile_chunk_offsets,
        "head_config_path": (
            None
            if args.pyramidkv_head_config_path is None
            else os.path.abspath(args.pyramidkv_head_config_path)
        ),
        "skip_video_decode": bool(args.skip_video_decode),
    }
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
            if idx < args.start_idx:
                continue
            if args.end_idx is not None and idx >= args.end_idx:
                break
            if args.prompt_stride > 1 and (idx - args.prompt_offset) % args.prompt_stride != 0:
                continue
            if idx in cache_profile_completed_prompt_ids:
                print(
                    f"[CacheCompatProfileResume] skip completed prompt={idx}",
                    flush=True,
                )
                continue
            if args.skip_existing:
                _skip_dir = args.output_folder
                import glob as _glob
                if _glob.glob(os.path.join(_skip_dir, f"{idx}-*.mp4")) or _glob.glob(os.path.join(_skip_dir, f"video_{idx:05d}*.mp4")):
                    print(f"[skip] prompt {idx} already has output, skipping", flush=True)
                    continue
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
                    set_diagnostic_prompt_id(idx)
                except Exception as e:
                    print(f"[DIAG] Failed to set prompt id: {e}")
            if args.probecache_profile_output:
                from wan.modules.attention.core import set_probecache_profile_prompt_id
                set_probecache_profile_prompt_id(idx)
            if args.head_qk_profile_output:
                from wan.modules.attention.head_profile import (
                    set_head_qk_profile_prompt_id,
                )
                set_head_qk_profile_prompt_id(idx)
            if args.cache_compat_profile_output:
                from wan.modules.attention.cache_compat_profile import (
                    set_cache_compat_profile_prompt_id,
                )
                set_cache_compat_profile_prompt_id(idx)

            # Generate video frames
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=initial_latent,
                low_memory=low_memory,
                profile=os.environ.get("ADAHEAD_PROFILE", "0") == "1",
                **({"decode_video": False} if args.skip_video_decode else {}),
            )
            num_generated_frames += latents.shape[1]

            if cache_profile_path is not None:
                from wan.modules.attention.cache_compat_profile import (
                    save_cache_compatibility_profile,
                )

                save_cache_compatibility_profile(
                    cache_profile_path,
                    _cache_compatibility_metadata(),
                )
                cache_profile_completed_prompt_ids.add(int(idx))

            # Final output video — clamp+uint8 on GPU, non-blocking D2H overlaps with cache clear
            if video is not None:
                video = torch.clamp(
                    255.0 * rearrange(video, 'b t c h w -> b t h w c'), 0, 255,
                ).to(dtype=torch.uint8).to(device='cpu', non_blocking=True)

                # Clear VAE cache while the non-blocking D2H transfer runs.
                pipeline.vae.model.clear_cache()
                torch.cuda.current_stream().synchronize()

            # Save the video if the current prompt is not a dummy prompt
            if video is not None and idx < num_prompts:
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

if args.head_qk_profile_output:
    try:
        from wan.modules.attention.head_profile import save_head_qk_profile

        profile_path = args.head_qk_profile_output.format(
            rank=int(os.environ.get("RANK", "0")),
            pid=os.getpid(),
        )
        save_head_qk_profile(
            profile_path,
            {
                "kind": args.head_qk_profile_kind,
                "pair_id": args.head_qk_profile_pair_id,
                "side": args.head_qk_profile_side,
                "seed": int(args.seed),
                "data_path": os.path.abspath(args.data_path),
                "update_modes": args.head_qk_profile_update_modes,
                "branches": args.head_qk_profile_branches,
                "num_output_frames": int(args.num_output_frames),
                "few_step_cfg_enabled": bool(args.few_step_cfg_enabled),
                "config_path": os.path.abspath(args.config_path),
                "checkpoint_path": os.path.abspath(args.checkpoint_path),
                "head_config_path": (
                    None
                    if args.pyramidkv_head_config_path is None
                    else os.path.abspath(args.pyramidkv_head_config_path)
                ),
            },
        )
    except Exception as e:
        print(f"[HeadQKProfile] Failed to save profile: {e}")

if args.cache_compat_profile_output:
    from wan.modules.attention.cache_compat_profile import (
        save_cache_compatibility_profile,
    )

    profile_path = cache_profile_path
    save_cache_compatibility_profile(
        profile_path,
        _cache_compatibility_metadata(),
    )

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
