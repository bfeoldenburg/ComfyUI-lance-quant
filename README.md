# ComfyUI-Lance

ComfyUI node pack for [ByteDance Lance](https://huggingface.co/bytedance-research/Lance) and the quantized checkpoints published in the `Reza2kn` Hugging Face account.

This package wraps Lance's native PyTorch runtime for:

- text to image
- text to video
- image edit
- image understanding / VQA
- video edit
- video understanding / VQA

It supports both:

- `resident_worker`: keep a warm Lance process alive for repeated prompts
- `subprocess`: spawn a fresh process per request for debugging and fallback

This repository intentionally ships only the ComfyUI wrapper and helper scripts.
It does **not** vendor the full ByteDance `Lance/` source tree. Users should
clone Lance separately or provide `LANCE_SRC_PATH`.

## Current Status

The ComfyUI wrapper in this repo has been adapted for a root-level custom node layout:

- node files live at the repo root, not under `comfyui/`
- `run_quant_eval.py` and `run_baseline.py` are resolved from the installed node repo
- `inference_lance` bootstrapping is fixed for copied installs
- `Wan2.2_VAE.pth` can be overridden through `LANCE_VAE_PATH`
- JSON manifest normalization is fixed for on-demand single-sample runs
- `video_edit` and `x2t_video` nodes are now exposed

## Nodes

| Node | Purpose |
|---|---|
| `Lance: Model Loader` | Select checkpoint flavor, precision, backend, and fallback behavior |
| `Lance: Text → Image` | `t2i` |
| `Lance: Text → Video` | `t2v` |
| `Lance: Image Edit` | `image_edit` |
| `Lance: Image Understanding (VQA)` | `x2t_image` |
| `Lance: Video Edit` | `video_edit` |
| `Lance: Video Understanding (VQA)` | `x2t_video` |

## Requirements

- Linux
- NVIDIA GPU strongly recommended
- ComfyUI Python environment with PyTorch already working
- ByteDance Lance source code available at:
  - `ComfyUI/custom_nodes/ComfyUI-Lance/Lance/`, or
  - the path given by `LANCE_SRC_PATH`

## Install

### 1. Install the node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/GuardSkill/ComfyUI-lance-quant ComfyUI-Lance
cd ComfyUI-Lance
git clone https://github.com/bytedance/Lance.git
```

If you copied only the wrapper files instead of cloning the full repo, set:

```bash
export LANCE_QUANT_PATH=/absolute/path/to/ComfyUI-Lance
export LANCE_SRC_PATH=/absolute/path/to/Lance
```

### 2. Install Python dependencies into the same environment ComfyUI uses

```bash
pip install transformers==4.49.0 diffusers==0.29.1 flash-attn \
           accelerate safetensors einops decord opencv-python \
           imageio imageio-ffmpeg qwen-vl-utils kornia \
           omegaconf pydantic timm sentencepiece tiktoken
```

Optional but recommended for audio round-trip in video nodes:

```bash
pip install torchaudio
```

### 3. Put models in `ComfyUI/models/lance/`

Expected layout:

```text
ComfyUI/models/lance/
├── Lance_3B/
├── Lance_3B_Video/
├── Qwen2.5-VL-ViT/
├── Wan2.2_VAE.pth
├── Lance_3B-AWQ-INT4/
├── Lance_3B-NVFP4/
├── Lance_3B_Video-AWQ-INT4/
└── Lance_3B_Video-NVFP4/
```

The wrapper also accepts the video quant folder name used by some repos:

- `Lance_3B_Video-AWQ-INT4`
- `Lance-3B-Video-AWQ-INT4`

Example downloads:

```bash
huggingface-cli download bytedance-research/Lance --local-dir ComfyUI/models/lance
huggingface-cli download Reza2kn/Lance-3B-AWQ-INT4 --local-dir ComfyUI/models/lance/Lance_3B-AWQ-INT4
huggingface-cli download Reza2kn/Lance-3B-NVFP4 --local-dir ComfyUI/models/lance/Lance_3B-NVFP4
huggingface-cli download Reza2kn/Lance-3B-Video-AWQ-INT4 --local-dir ComfyUI/models/lance/Lance_3B_Video-AWQ-INT4
huggingface-cli download Reza2kn/Lance-3B-Video-NVFP4 --local-dir ComfyUI/models/lance/Lance_3B_Video-NVFP4
```

The helper scripts in this repo already patch the critical Lance loading path at runtime, so a separate manual patch step is not required for the ComfyUI wrapper.

### 4. Restart ComfyUI

All nodes will appear under the `Lance` category.

## Recommended Settings

### Image tasks

- `flavor = Lance_3B`
- `precision = awq_int4`
- `backend = resident_worker`

### Video tasks

- `flavor = Lance_3B_Video`
- `precision = awq_int4`
- `backend = resident_worker`

## Backend Modes

| Backend | Behavior |
|---|---|
| `resident_worker` | Starts one warm Lance process per loaded config. Best for repeated use. |
| `subprocess` | Starts a fresh helper process every run. Slower, but easier to debug. |

`fallback_to_subprocess=true` is useful during bring-up, but it can also hide the first real error and can increase VRAM pressure if the resident worker already occupied memory.

## VRAM Notes

Approximate LLM-side footprint:

| Precision | Approx VRAM |
|---|---|
| `bf16` | about 14 GB |
| `awq_int4` | about 7 GB |
| `nvfp4` | about 7 GB |

Generation tasks also need additional VAE memory, roughly another 2 GB.

## Troubleshooting

### `Cannot find run_quant_eval.py`

The node cannot find the helper scripts.

Fix:

- keep the node installed as the full repo at `ComfyUI/custom_nodes/ComfyUI-Lance`
- or set `LANCE_QUANT_PATH`

### `ModuleNotFoundError: No module named 'inference_lance'`

The helper script cannot find the Lance source tree.

Fix:

- keep ByteDance Lance at `ComfyUI/custom_nodes/ComfyUI-Lance/Lance`
- or set `LANCE_SRC_PATH`

### `FileNotFoundError: downloads/Wan2.2_VAE.pth`

Lance is falling back to its default internal path.

Fix:

- place `Wan2.2_VAE.pth` in `ComfyUI/models/lance/`
- the wrapper now forwards that path automatically through `LANCE_VAE_PATH`

### `KeyError: 'data'`

This used to happen during on-demand single-sample runs because Lance's validation dataset code expected normalized sample rows.

Fix:

- current wrapper versions normalize these manifests automatically
- restart ComfyUI after updating the node

### `CUDA out of memory` after resident fallback

This usually means:

1. the resident worker already loaded the model
2. fallback launched a second subprocess copy
3. both processes together exhausted VRAM

Fix:

- restart ComfyUI to clear old worker processes
- test with `backend=resident_worker` and `fallback_to_subprocess=false`

## Example Workflows

Example JSON files currently included:

- [Text to Image](example_workflows/lance_t2i.json)
- [Image Understanding](example_workflows/lance_x2t_image.json)

## Publishing Notes

This repo can be prepared for Comfy Registry publication, but actual publication requires:

- a Comfy Registry publisher id
- a Comfy Registry publishing API key
- GitHub push access to the target repository
- a decision on how to distribute the `Lance/` runtime source tree for end users

See the official docs:

- Publishing: https://docs.comfy.org/registry/publishing
- `pyproject.toml` spec: https://docs.comfy.org/registry/specifications

## License

Apache-2.0, consistent with Lance.
