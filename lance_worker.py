"""Persistent Lance worker for ComfyUI.

Spawned once per checkpoint+precision combo. Loads Lance into memory and waits
for one-shot inference requests over stdin (one JSON request per line). Each
request gets one JSON response per line on stdout. The worker stays alive
across many requests so the ~30 s model-load tax is paid only once per session.

Protocol (line-delimited JSON):

  request:
    {"task": "x2t_image", "manifest_path": "/tmp/foo.json", "save_dir": "/tmp/bar"}
  response:
    {"ok": true, "outputs": {"image-01.png": "text answer"}}
    {"ok": false, "error": "..."}

The companion `nodes.py` (v2) spawns this worker on the first LanceModelLoader
call, keeps it alive in a module-global, and pipes requests to it.

The companion `nodes.py` starts this process for the `resident_worker` backend
and falls back to the subprocess backend if the local Lance install cannot
support resident execution.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def _meta_init():
    orig_empty = torch.empty
    def _empty_meta(*sizes, **kw):
        kw.setdefault("device", "meta")
        return orig_empty(*sizes, **kw)
    torch.empty = _empty_meta
    try:
        yield
    finally:
        torch.empty = orig_empty


def _install_path_overrides() -> str | None:
    vae_path = os.environ.get("LANCE_VAE_PATH")
    if not vae_path:
        return None

    resolved = Path(vae_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"LANCE_VAE_PATH does not exist: {resolved}")

    import config.config_factory as config_factory

    original_get_model_path = config_factory.get_model_path

    def _get_model_path(path_key: str) -> str:
        if path_key == "vae.wan":
            return str(resolved)
        return original_get_model_path(path_key)

    config_factory.get_model_path = _get_model_path
    return str(resolved)


def _tuple_mul(values_a, values_b):
    return tuple(int(a) * int(b) for a, b in zip(values_a, values_b))


def _configure_dataset_config(dataset_config, model_args, inference_args, vae_config):
    if getattr(inference_args, "visual_und", False):
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

    if getattr(inference_args, "visual_gen", False) and vae_config is not None:
        dataset_config.latent_patch_size = model_args.latent_patch_size
        dataset_config.vae_downsample = _tuple_mul(
            tuple(model_args.latent_patch_size),
            (
                vae_config.downsample_temporal,
                vae_config.downsample_spatial,
                vae_config.downsample_spatial,
            ),
        )
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.max_num_frames = model_args.max_num_frames

    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    dataset_config.num_frames = inference_args.num_frames
    dataset_config.H = inference_args.video_height
    dataset_config.W = inference_args.video_width
    dataset_config.task = inference_args.task
    dataset_config.resolution = inference_args.resolution
    dataset_config.text_template = inference_args.text_template
    return dataset_config


def _normalise_manifest_for_worker(task: str, manifest_path: str) -> str:
    """Expand shorthand generation manifests into the sample shape the
    ValidationDataset expects when we bypass inference_lance.main()."""
    if task not in {"t2i", "t2v"}:
        return manifest_path

    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return manifest_path

    if not isinstance(payload, dict) or not payload:
        return manifest_path

    changed = False
    normalised: dict[str, object] = {}
    for sample_id, sample in payload.items():
        if isinstance(sample, str):
            stem, ext = osp.splitext(str(sample_id))
            normalised[str(sample_id)] = {
                "id": stem or str(sample_id),
                "file_name": str(sample_id),
                "image_path": str(sample_id) if ext.lower() in {".png", ".jpg", ".jpeg", ".webp"} else None,
                "video_path": str(sample_id) if ext.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"} else None,
                "data": sample,
                "prompt": sample,
                "text": sample,
            }
            changed = True
            continue

        if isinstance(sample, dict) and "data" not in sample:
            prompt = sample.get("prompt")
            if isinstance(prompt, str):
                patched = dict(sample)
                patched["data"] = prompt
                normalised[str(sample_id)] = patched
                changed = True
                continue

        normalised[str(sample_id)] = sample

    if not changed:
        return manifest_path

    fd, tmp_path = tempfile.mkstemp(prefix=f"lance_manifest_{task}_", suffix=".json")
    os.close(fd)
    Path(tmp_path).write_text(json.dumps(normalised), encoding="utf-8")
    return tmp_path


class _BootstrapReady(Exception):
    pass


def build_lance_resident(*, lance_src: Path, model_path: str, vit_path: str,
                          awq_dir: str | None, save_path_gen: str,
                          script_root: Path):
    """Construct Lance with our memory-frugal loader + (optionally) AWQ swap.
    Returns a dict holding model, vae, tokenizer, training_args, image_token_id
    — everything `validate_on_fixed_batch` needs."""
    sys.path.insert(0, str(lance_src))
    sys.path.insert(0, str(script_root))
    _install_path_overrides()
    from modeling.lance import Lance
    from modeling.lance.qwen2_navit import Qwen2ForCausalLM
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
    import inference_lance as IL

    # Meta-init patches (same as scripts/run_baseline.py)
    _OQ, _OV, _OL = Qwen2ForCausalLM.__init__, \
        Qwen2_5_VisionTransformerPretrainedModel.__init__, Lance.__init__
    def _Q(self, c):
        with _meta_init(): _OQ(self, c)
    def _V(self, c):
        with _meta_init(): _OV(self, c)
    def _L(self, *a, **k):
        with _meta_init(): _OL(self, *a, **k)
    Qwen2ForCausalLM.__init__ = _Q
    Qwen2_5_VisionTransformerPretrainedModel.__init__ = _V
    Lance.__init__ = _L

    # Build sys.argv that inference_lance.main expects
    sys.argv = [
        "inference_lance.py",
        "--model_path", model_path, "--vit_path", vit_path,
        "--vit_type", "qwen_2_5_vl_original",
        "--llm_qk_norm", "true", "--llm_qk_norm_und", "true",
        "--llm_qk_norm_gen", "true", "--tie_word_embeddings", "false",
        "--validation_num_timesteps", "30", "--validation_timestep_shift", "3.5",
        "--copy_init_moe", "true", "--max_num_frames", "121",
        "--max_latent_size", "64", "--latent_patch_size", "1", "1", "1",
        "--visual_und", "true", "--visual_gen", "true",
        "--vae_model_type", "wan", "--apply_qwen_2_5_vl_pos_emb", "true",
        "--apply_chat_template", "false", "--cfg_type", "0",
        "--validation_data_seed", "42",
        "--video_height", "768", "--video_width", "768", "--num_frames", "50",
        "--task", "x2t_image", "--save_path_gen", save_path_gen,
        "--resolution", "image_768res", "--text_template", "true",
        "--cfg_text_scale", "4.0", "--use_KVcache", "true",
        "--val_dataset_config_file", "config/examples/x2t_image_example.json",
    ]

    # Replace loader with AWQ-swap or bf16-stream
    if awq_dir:
        # delayed import to avoid circulars
        from scripts.run_quant_eval import (
            swap_to_awq, stream_pass_through_weights,
            stream_awq_buffers, WQLinearINT4,
        )
        WQLinearINT4.MODE = "ondemand"
        def _loader(model, model_args):
            mods = swap_to_awq(model, Path(awq_dir))
            stream_pass_through_weights(model, Path(awq_dir))
            stream_awq_buffers(mods, Path(awq_dir))
            class _M:
                missing_keys, unexpected_keys = [], []
            return _M()
        IL.init_from_model_path_if_needed = _loader
    else:
        from scripts.run_baseline import _streaming_bf16_loader
        IL.init_from_model_path_if_needed = _streaming_bf16_loader

    # Now run the first ~530 lines of inference_lance.main() to build, but
    # break before the validation loop. We do this by running main() until
    # validate_on_fixed_batch is called for the first time, then capturing
    # the state.
    state = {}
    orig_validate = IL.validate_on_fixed_batch
    def _capture(*a, **kw):
        state.update({
            "fsdp_model": kw.get("fsdp_model") or a[0],
            "vae_model": kw.get("vae_model"),
            "vae_config": getattr(kw.get("vae_model"), "vae_config", None),
            "tokenizer": kw.get("tokenizer"),
            "training_args": kw.get("training_args"),
            "model_args": kw.get("model_args"),
            "inference_args": kw.get("inference_args"),
            "new_token_ids": kw.get("new_token_ids"),
            "image_token_id": kw.get("image_token_id"),
            "device": kw.get("device"),
        })
        raise _BootstrapReady()
    IL.validate_on_fixed_batch = _capture

    print("[worker] building Lance...", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        IL.main()
    except _BootstrapReady:
        pass
    print(f"[worker] build+bootstrap in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    # Restore the real validate for subsequent calls
    IL.validate_on_fixed_batch = orig_validate
    state["IL"] = IL
    return state


def serve(state: dict):
    """Read one JSON request per line from stdin; respond on stdout."""
    IL = state["IL"]
    from data.dataset_base import DataConfig, simple_custom_collate
    from data.datasets_custom import ValidationDataset
    from torch.utils.data import DataLoader

    print("READY", flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            task = req["task"]
            manifest_path = _normalise_manifest_for_worker(task, req["manifest_path"])
            print(f"[worker-debug] task={task} manifest_path={manifest_path}", file=sys.stderr, flush=True)
            print(Path(manifest_path).read_text(encoding="utf-8")[:1000], file=sys.stderr, flush=True)
            save_dir = req["save_dir"]
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Override save_dir + task in inference_args
            ia = state["inference_args"]
            ia.task = task
            ia.save_path_gen = save_dir
            ia.validation_num_timesteps = req.get("num_steps", getattr(ia, "validation_num_timesteps", 30))
            ia.num_frames = req.get("num_frames", getattr(ia, "num_frames", 1))
            ia.video_height = req.get("height", getattr(ia, "video_height", 768))
            ia.video_width = req.get("width", getattr(ia, "video_width", 768))
            ia.cfg_text_scale = req.get("cfg_scale", getattr(ia, "cfg_text_scale", 4.0))
            ia.validation_data_seed = req.get("seed", getattr(ia, "validation_data_seed", 42))
            ia.prompt_data_dict = {}

            # Build a one-shot ValidationDataset on the user's manifest
            dataset_config = _configure_dataset_config(
                DataConfig.from_yaml(manifest_path),
                state["model_args"],
                ia,
                state.get("vae_config"),
            )
            ds = ValidationDataset(
                jsonl_path=manifest_path,
                tokenizer=state["tokenizer"],
                data_args=type("DA", (), {"val_dataset_config_file": manifest_path})(),
                model_args=state["model_args"],
                training_args=state["training_args"],
                new_token_ids=state["new_token_ids"],
                dataset_config=dataset_config,
                local_rank=0, world_size=1,
            )
            loader = DataLoader(ds, batch_size=1, num_workers=0,
                                  collate_fn=simple_custom_collate)
            for batch in loader:
                IL.validate_on_fixed_batch(
                    fsdp_model=state["fsdp_model"],
                    vae_model=state["vae_model"],
                    tokenizer=state["tokenizer"],
                    val_data_cpu=batch,
                    training_args=state["training_args"],
                    model_args=state["model_args"],
                    inference_args=ia,
                    new_token_ids=state["new_token_ids"],
                    image_token_id=state["image_token_id"],
                    device=state["device"],
                    save_source_video=False, save_path_gen=save_dir,
                    save_path_gt="",
                )
            outputs = dict(ia.prompt_data_dict)
            print(json.dumps({"ok": True, "outputs": outputs}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False,
                              "error": str(e),
                              "trace": traceback.format_exc()[-2000:]}),
                  flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance_src", type=Path, required=True)
    ap.add_argument("--script_root", type=Path, required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vit_path", required=True)
    ap.add_argument("--awq_dir", default=None)
    ap.add_argument("--save_path_gen", default="/tmp/lance_worker_results")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    state = build_lance_resident(
        lance_src=args.lance_src,
        model_path=args.model_path, vit_path=args.vit_path,
        awq_dir=args.awq_dir, save_path_gen=args.save_path_gen,
        script_root=args.script_root,
    )
    serve(state)


if __name__ == "__main__":
    main()
