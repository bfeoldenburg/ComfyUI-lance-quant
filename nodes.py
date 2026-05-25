"""ComfyUI nodes for ByteDance Lance + AWQ-INT4 / NVFP4 quantized variants.

v2 design: the default backend starts a resident Lance worker per
checkpoint/precision combo and sends requests over line-delimited JSON. The
older subprocess backend remains available for debugging and as a fallback.

Setup:
  - Clone github.com/bytedance/Lance to either:
      * `<this dir>/Lance/`  (default search path)
      * or set LANCE_SRC_PATH env var
  - Place model weights under `ComfyUI/models/lance/`:
        Lance_3B/                          (or Lance_3B_Video/)
        Qwen2.5-VL-ViT/
        Wan2.2_VAE.pth
        Lance_3B-AWQ-INT4/                 (optional, for 4-bit inference)
        Lance_3B-NVFP4/                    (optional)
  - The Lance conda env must be activated in the ComfyUI process; see top-level
    README of lance-quant for the env recipe.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Paths & helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _lance_src_path() -> Path:
    env = os.environ.get("LANCE_SRC_PATH")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).parent
    for cand in (here / "Lance", here.parent / "Lance",
                  Path.home() / "lance-quant" / "src"):
        if (cand / "inference_lance.py").exists():
            return cand
    raise RuntimeError(
        "Lance source not found. Either set LANCE_SRC_PATH env var to your "
        "git clone of github.com/bytedance/Lance, or place the repo at "
        f"{here / 'Lance'} or {here.parent / 'Lance'}."
    )


def _model_root() -> Path:
    try:
        import folder_paths
        root = Path(folder_paths.models_dir) / "lance"
    except Exception:
        root = Path(os.environ.get("LANCE_MODELS_DIR", "models/lance"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _script_path(script_name: str) -> Path:
    """Find lance-quant helper scripts whether installed as a repo or copied."""
    env = os.environ.get("LANCE_QUANT_PATH")
    candidates = []
    if env:
        env_path = Path(env)
        candidates.extend([
            env_path / "scripts" / script_name,
            env_path / script_name,
        ])
    candidates.extend([
        _repo_root() / "scripts" / script_name,
        Path(__file__).resolve().parent / "scripts" / script_name,
        Path(__file__).resolve().parent / script_name,
        _lance_src_path() / "scripts" / script_name,
        _lance_src_path() / script_name,
    ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise RuntimeError(
        f"Cannot find {script_name}. Set LANCE_QUANT_PATH to the lance-quant "
        "checkout, or keep the ComfyUI node inside the repo."
    )


def _resolve_quant_dir(root: Path, flavor: str, precision: str) -> Path | None:
    if precision == "bf16":
        return None

    suffix = "AWQ-INT4" if precision == "awq_int4" else "NVFP4"
    candidates = [
        root / f"{flavor}-{suffix}",
        root / f"{flavor.replace('_', '-')}-{suffix}",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE: [B, H, W, C] float 0..1 -> PIL."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI IMAGE: [1, H, W, C] float 0..1."""
    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _normalise_video_frames(video_frames) -> np.ndarray:
    """ComfyUI video frames -> uint8 numpy array [frames, H, W, 3]."""
    if video_frames is None:
        raise ValueError("video_frames cannot be empty")

    frames = video_frames.detach().cpu() if hasattr(video_frames, "detach") else video_frames
    if hasattr(frames, "dim"):
        dims = frames.dim()
    else:
        dims = getattr(frames, "ndim", None)

    if dims == 3:
        frames = frames.unsqueeze(0) if hasattr(frames, "unsqueeze") else np.expand_dims(frames, axis=0)
        dims = 4
    if dims != 4:
        raise ValueError("video_frames must be a 4D IMAGE tensor")

    shape = tuple(frames.shape)
    if shape[1] in (3, 4) and shape[-1] not in (3, 4):
        frames = frames.permute(0, 2, 3, 1) if hasattr(frames, "permute") else np.transpose(frames, (0, 2, 3, 1))
        shape = tuple(frames.shape)
    if shape[-1] not in (3, 4):
        raise ValueError("video_frames must use RGB/RGBA channels")

    arr = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _normalise_audio(audio):
    if audio is None:
        return None

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        raise ValueError("audio must contain waveform and sample_rate")

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    dims = waveform.dim() if hasattr(waveform, "dim") else getattr(waveform, "ndim", None)
    if dims == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif dims == 2:
        waveform = waveform.unsqueeze(0)
    elif dims != 3:
        raise ValueError("audio waveform must be 1D, 2D, or 3D")

    if hasattr(waveform, "contiguous"):
        waveform = waveform.contiguous()
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _video_request_shape(video_frames) -> tuple[int, int, int]:
    frames = _normalise_video_frames(video_frames)
    return int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])


def _save_video_input(video_frames, video_audio, video_fps, video_path: str) -> str:
    frames = _normalise_video_frames(video_frames)
    audio = _normalise_audio(video_audio)
    fps = max(float(video_fps or 0), 1.0)
    video_path = str(Path(video_path).resolve())
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)

    frame_dir = Path(tempfile.mkdtemp(prefix="lance_video_frames_"))
    wav_path = None
    try:
        for index, frame in enumerate(frames):
            Image.fromarray(frame, mode="RGB").save(frame_dir / f"{index:06d}.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "%06d.png"),
        ]

        if audio is not None:
            try:
                import torchaudio
            except Exception:
                torchaudio = None
            if torchaudio is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name
                torchaudio.save(wav_path, audio["waveform"][0], int(audio["sample_rate"]))
                cmd.extend(["-i", wav_path])

        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        if wav_path is not None:
            cmd.extend(["-c:a", "aac", "-shortest"])
        cmd.append(video_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg failed to save video input")
        return video_path
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def _extract_audio_from_video(video_path: str):
    try:
        import torchaudio
    except Exception:
        return None

    try:
        waveform, sample_rate = torchaudio.load(video_path)
    except Exception:
        return None
    if waveform.numel() == 0:
        return None
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _extract_frames_from_video(video_path: str) -> tuple[torch.Tensor, float]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for Lance video nodes") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames found in {video_path}")
    return torch.from_numpy(np.stack(frames)).to(torch.float32) / 255.0, fps


def _video_to_comfy_outputs(video_path: str):
    frames, fps = _extract_frames_from_video(video_path)
    audio = _extract_audio_from_video(video_path)
    return frames, audio, fps


def _build_video_edit_manifest(video_path: str, instruction: str) -> dict:
    return {
        "0001": {
            "interleave_array": [instruction, video_path, video_path],
            "element_dtype_array": ["text", "video", "video"],
            "istarget_in_interleave": [0, 0, 1],
        }
    }


def _build_x2t_video_manifest(video_path: str, question: str) -> dict:
    return {
        "0001": {
            "interleave_array": [
                video_path,
                ["Watch the video carefully and answer the question.", question, ""],
            ],
            "element_dtype_array": ["video", "text"],
            "istarget_in_interleave": [0, 1],
        }
    }


def _find_output_video(save_dir: Path) -> Path:
    for candidate in ("0001.mp4", "000000.mp4"):
        path = save_dir / candidate
        if path.exists():
            return path
    matches = sorted(save_dir.glob("*.mp4"))
    if matches:
        return matches[0]
    raise RuntimeError(f"no output video found in {save_dir}")


# ---------------------------------------------------------------------------
# Backends: v1 subprocess runner and v2 persistent worker
# ---------------------------------------------------------------------------


class _WorkerClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

    def _start(self):
        if self.proc and self.proc.poll() is None:
            return
        src = _lance_src_path()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "lance_worker.py"),
            "--lance_src", str(src),
            "--script_root", str(_repo_root()),
            "--model_path", self.cfg["bf16_path"],
            "--vit_path", self.cfg["vit_path"],
            "--save_path_gen", tempfile.mkdtemp(prefix="lance_worker_boot_"),
        ]
        if self.cfg.get("awq_dir"):
            cmd.extend(["--awq_dir", self.cfg["awq_dir"]])

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        if self.cfg.get("vae_path"):
            env["LANCE_VAE_PATH"] = self.cfg["vae_path"]
        self.proc = subprocess.Popen(
            cmd, cwd=str(src), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        def _drain_stderr():
            assert self.proc and self.proc.stderr
            for line in self.proc.stderr:
                self._stderr_q.put(line)
                sys.stderr.write(line)

        threading.Thread(target=_drain_stderr, daemon=True).start()
        assert self.proc.stdout
        deadline = time.time() + int(os.environ.get("LANCE_WORKER_START_TIMEOUT", "900"))
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "" and self.proc.poll() is not None:
                raise RuntimeError(self._worker_error("worker exited before READY"))
            if line.strip() == "READY":
                return
            if line:
                sys.stderr.write(f"[lance-worker] {line}")
        raise TimeoutError(self._worker_error("timed out waiting for READY"))

    def _worker_error(self, prefix: str) -> str:
        tail = []
        while not self._stderr_q.empty():
            tail.append(self._stderr_q.get_nowait())
        return prefix + ("\n" + "".join(tail[-40:]) if tail else "")

    def request(self, payload: dict) -> dict:
        with self._lock:
            self._start()
            assert self.proc and self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if line == "" and self.proc.poll() is not None:
                    raise RuntimeError(self._worker_error("worker exited during request"))
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    sys.stderr.write(f"[lance-worker] {line}\n")
                    continue
                if not res.get("ok"):
                    raise RuntimeError(res.get("error", "Lance worker failed") + "\n" + res.get("trace", ""))
                return res


_WORKERS: dict[str, _WorkerClient] = {}


def _worker_for(cfg: dict) -> _WorkerClient:
    key = json.dumps({
        "src": str(_lance_src_path()),
        "model": cfg["bf16_path"],
        "vit": cfg["vit_path"],
        "precision": cfg["precision"],
        "awq": cfg.get("awq_dir"),
    }, sort_keys=True)
    if key not in _WORKERS:
        _WORKERS[key] = _WorkerClient(cfg)
    return _WORKERS[key]


def _run_lance_cli(*, task: str, model_path: str, vit_path: str,
                   awq_dir: str | None, vae_path: str | None,
                   example_json: str, save_dir: str,
                   num_steps: int, num_frames: int, height: int, width: int,
                   cfg_scale: float, seed: int) -> dict:
    """Call our run_baseline.py / run_quant_eval.py and return the parsed result."""
    py = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    if vae_path:
        env["LANCE_VAE_PATH"] = vae_path

    if awq_dir:
        script = "run_quant_eval.py"
        cmd = [py, str(_script_path(script)), "--task", task,
                "--model_path", model_path, "--vit_path", vit_path,
                "--awq_dir", awq_dir, "--example_json", example_json,
                "--save_path_gen", save_dir,
                "--validation_num_timesteps", str(num_steps),
                "--cfg_scale", str(cfg_scale), "--seed", str(seed),
                "--video_height", str(height), "--video_width", str(width),
                "--num_frames", str(num_frames), "--mode", "ondemand"]
    else:
        script = "run_baseline.py"
        cmd = [py, str(_script_path(script)), "--task", task,
                "--model_path", model_path, "--vit_path", vit_path,
                "--example_json", example_json,
                "--save_path_gen", save_dir,
                "--validation_num_timesteps", str(num_steps),
                "--cfg_scale", str(cfg_scale), "--seed", str(seed),
                "--video_height", str(height), "--video_width", str(width),
                "--num_frames", str(num_frames)]

    print(f"[lance-comfy] running {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(_lance_src_path()), env=env, capture_output=True, text=True)
    print(f"[lance-comfy] returned {res.returncode} in {time.time()-t0:.1f}s")
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        raise RuntimeError(f"Lance CLI failed: {res.stderr[-500:]}")

    prompt_json = Path(save_dir) / "prompt.json"
    if prompt_json.exists():
        return json.loads(prompt_json.read_text())
    return {}


def _run_lance(*, lance_model: dict, task: str, example_json: str, save_dir: str,
               num_steps: int, num_frames: int, height: int, width: int,
               cfg_scale: float, seed: int) -> dict:
    backend = lance_model.get("backend", "resident_worker")
    if backend == "resident_worker":
        try:
            return _worker_for(lance_model).request({
                "task": task,
                "manifest_path": example_json,
                "save_dir": save_dir,
                "num_steps": num_steps,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "cfg_scale": cfg_scale,
                "seed": seed,
            }).get("outputs", {})
        except Exception:
            if not lance_model.get("fallback_to_subprocess", True):
                raise
            print("[lance-comfy] resident worker failed; falling back to subprocess", file=sys.stderr)

    return _run_lance_cli(
        task=task,
        model_path=lance_model["bf16_path"],
        vit_path=lance_model["vit_path"],
        awq_dir=lance_model["awq_dir"],
        vae_path=lance_model.get("vae_path"),
        example_json=example_json,
        save_dir=save_dir,
        num_steps=num_steps,
        num_frames=num_frames,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


class LanceModelLoader:
    """Selects which Lance checkpoint + precision to use; returns a config dict.
    The actual model is reloaded per inference call (see top-level docstring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flavor": (["Lance_3B", "Lance_3B_Video"], {"default": "Lance_3B"}),
                "precision": (["bf16", "awq_int4", "nvfp4"], {"default": "awq_int4"}),
                "backend": (["resident_worker", "subprocess"], {"default": "resident_worker"}),
                "fallback_to_subprocess": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("LANCE_MODEL",)
    RETURN_NAMES = ("lance_model",)
    FUNCTION = "load"
    CATEGORY = "Lance"

    def load(self, flavor, precision, backend, fallback_to_subprocess):
        root = _model_root()
        vit = root / "Qwen2.5-VL-ViT"
        vae = root / "Wan2.2_VAE.pth"
        # the bf16 source dir is needed for tokenizer/config in all modes
        bf16 = root / flavor
        awq_dir = _resolve_quant_dir(root, flavor, precision)

        for p, name in [(vit, "ViT"), (vae, "VAE"), (bf16, flavor)]:
            if not p.exists():
                raise RuntimeError(
                    f"missing {name} at {p}. Download "
                    f"bytedance-research/Lance into {root}.")
        if awq_dir and not awq_dir.exists():
            raise RuntimeError(
                f"missing quantized weights at {awq_dir}. Download "
                f"Reza2kn/{awq_dir.name} into {root}.")

        cfg = {
            "flavor": flavor, "precision": precision,
            "backend": backend,
            "fallback_to_subprocess": fallback_to_subprocess,
            "bf16_path": str(bf16), "vit_path": str(vit), "vae_path": str(vae),
            "awq_dir": str(awq_dir) if awq_dir else None,
        }
        print(f"[lance-comfy] loaded config: {cfg}")
        return (cfg,)


class _BaseLanceTask:
    CATEGORY = "Lance"
    TASK_NAME: str = ""

    def _save_example_json(self, save_dir: Path, payload: dict) -> str:
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / "_input_manifest.json"
        p.write_text(json.dumps(payload))
        return str(p)


class LanceT2I(_BaseLanceTask):
    TASK_NAME = "t2i"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "prompt": ("STRING", {"multiline": True,
                                    "default": "A beautiful landscape painting."}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
            "size": (["768x768", "1024x1024", "512x512"], {"default": "768x768"}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, lance_model, prompt, num_steps, cfg_scale, seed, size):
        h, w = map(int, size.split("x"))
        save = Path(tempfile.mkdtemp(prefix="lance_t2i_"))
        # Lance's example JSON for t2i is {filename: prompt}
        manifest = {"000000.png": prompt}
        (save / "_input.json").write_text(json.dumps(manifest))

        _run_lance(
            lance_model=lance_model, task="t2i",
            example_json=str(save / "_input.json"),
            save_dir=str(save), num_steps=num_steps,
            num_frames=1, height=h, width=w,
            cfg_scale=cfg_scale, seed=seed,
        )

        # Lance saves PNG/MP4 as 000000.{png,mp4}
        for fname in ("000000.png", "000000.mp4"):
            p = save / fname
            if p.exists():
                if fname.endswith(".png"):
                    return (_pil_to_tensor(Image.open(p)),)
        raise RuntimeError(f"no output in {save}")


class LanceX2TImage(_BaseLanceTask):
    TASK_NAME = "x2t_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "image": ("IMAGE",),
            "question": ("STRING", {"multiline": True,
                                      "default": "Describe this image."}),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self, lance_model, image, question):
        save = Path(tempfile.mkdtemp(prefix="lance_x2t_"))
        pil = _tensor_to_pil(image)
        ipath = save / "input.png"
        pil.save(ipath)

        # Lance's x2t_image expects an interleave_array with image + Q
        manifest = {"0001": {
            "interleave_array": [str(ipath),
                                  ["Look at the image carefully and answer.",
                                   question, ""]],
            "element_dtype_array": ["image", "text"],
            "istarget_in_interleave": [0, 1],
        }}
        (save / "_input.json").write_text(json.dumps(manifest))

        results = _run_lance(
            lance_model=lance_model, task="x2t_image",
            example_json=str(save / "_input.json"),
            save_dir=str(save), num_steps=1, num_frames=1,
            height=768, width=768, cfg_scale=4.0, seed=42,
        )
        # results: {filename: answer_text}
        if results:
            return (next(iter(results.values())).replace("<|im_end|>", "").strip(),)
        return ("(no output)",)


class LanceImageEdit(_BaseLanceTask):
    TASK_NAME = "image_edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "image": ("IMAGE",),
            "instruction": ("STRING", {"multiline": True,
                                         "default": "Make it look like a watercolor painting."}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, lance_model, image, instruction, num_steps, cfg_scale, seed):
        save = Path(tempfile.mkdtemp(prefix="lance_edit_"))
        pil = _tensor_to_pil(image)
        ipath = save / "input.png"
        pil.save(ipath)

        manifest = {"0001": {
            "interleave_array": [instruction, str(ipath), str(ipath)],
            "element_dtype_array": ["text", "image", "image"],
            "istarget_in_interleave": [0, 0, 1],
        }}
        (save / "_input.json").write_text(json.dumps(manifest))

        _run_lance(
            lance_model=lance_model, task="image_edit",
            example_json=str(save / "_input.json"),
            save_dir=str(save), num_steps=num_steps, num_frames=1,
            height=pil.height, width=pil.width,
            cfg_scale=cfg_scale, seed=seed,
        )
        out = save / "0001.png"
        if out.exists():
            return (_pil_to_tensor(Image.open(out)),)
        raise RuntimeError(f"no edited image found in {save}")


# t2v / video_edit / x2t_video follow the same pattern; keeping them simple

class LanceT2V(_BaseLanceTask):
    TASK_NAME = "t2v"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "prompt": ("STRING", {"multiline": True}),
            "num_frames": ("INT", {"default": 50, "min": 5, "max": 121}),
            "height": ("INT", {"default": 480, "min": 256, "max": 768, "step": 32}),
            "width": ("INT", {"default": 832, "min": 256, "max": 1280, "step": 32}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }}
    RETURN_TYPES = ("STRING",)   # returns path to generated video file
    FUNCTION = "run"
    def run(self, lance_model, prompt, num_frames, height, width, num_steps, cfg_scale, seed):
        if lance_model["flavor"] != "Lance_3B_Video":
            raise RuntimeError("t2v requires the Lance_3B_Video checkpoint")
        save = Path(tempfile.mkdtemp(prefix="lance_t2v_"))
        manifest = {"000000.mp4": prompt}
        (save / "_input.json").write_text(json.dumps(manifest))
        _run_lance(lance_model=lance_model, task="t2v",
                       example_json=str(save / "_input.json"),
                       save_dir=str(save), num_steps=num_steps,
                       num_frames=num_frames, height=height, width=width,
                       cfg_scale=cfg_scale, seed=seed)
        return (str(save / "000000.mp4"),)


class LanceVideoEdit(_BaseLanceTask):
    TASK_NAME = "video_edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "video_frames": ("IMAGE",),
            "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            "instruction": ("STRING", {"multiline": True, "default": "Change the background."}),
            "num_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 15.0, "step": 0.1}),
            "seed": ("INT", {"default": 42}),
        }, "optional": {
            "video_audio": ("AUDIO",),
        }}
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "run"

    def run(self, lance_model, video_frames, video_fps, instruction, num_steps, cfg_scale, seed, video_audio=None):
        if lance_model["flavor"] != "Lance_3B_Video":
            raise RuntimeError("video_edit requires the Lance_3B_Video checkpoint")

        save = Path(tempfile.mkdtemp(prefix="lance_video_edit_"))
        source_video = _save_video_input(video_frames, video_audio, video_fps, save / "source.mp4")
        num_frames, height, width = _video_request_shape(video_frames)
        manifest = _build_video_edit_manifest(source_video, instruction)
        (save / "_input.json").write_text(json.dumps(manifest), encoding="utf-8")

        _run_lance(
            lance_model=lance_model,
            task="video_edit",
            example_json=str(save / "_input.json"),
            save_dir=str(save),
            num_steps=num_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        return _video_to_comfy_outputs(str(_find_output_video(save)))


class LanceX2TVideo(_BaseLanceTask):
    TASK_NAME = "x2t_video"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lance_model": ("LANCE_MODEL",),
            "video_frames": ("IMAGE",),
            "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            "question": ("STRING", {"multiline": True, "default": "Describe this video."}),
        }, "optional": {
            "video_audio": ("AUDIO",),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self, lance_model, video_frames, video_fps, question, video_audio=None):
        if lance_model["flavor"] != "Lance_3B_Video":
            raise RuntimeError("x2t_video requires the Lance_3B_Video checkpoint")

        save = Path(tempfile.mkdtemp(prefix="lance_x2t_video_"))
        source_video = _save_video_input(video_frames, video_audio, video_fps, save / "source.mp4")
        num_frames, height, width = _video_request_shape(video_frames)
        manifest = _build_x2t_video_manifest(source_video, question)
        (save / "_input.json").write_text(json.dumps(manifest), encoding="utf-8")

        results = _run_lance(
            lance_model=lance_model,
            task="x2t_video",
            example_json=str(save / "_input.json"),
            save_dir=str(save),
            num_steps=1,
            num_frames=num_frames,
            height=height,
            width=width,
            cfg_scale=4.0,
            seed=42,
        )
        if results:
            return (next(iter(results.values())).replace("<|im_end|>", "").strip(),)
        return ("(no output)",)


# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LanceModelLoader": LanceModelLoader,
    "LanceT2I":         LanceT2I,
    "LanceT2V":         LanceT2V,
    "LanceVideoEdit":   LanceVideoEdit,
    "LanceImageEdit":   LanceImageEdit,
    "LanceX2TImage":    LanceX2TImage,
    "LanceX2TVideo":    LanceX2TVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LanceModelLoader": "Lance: Model Loader",
    "LanceT2I":         "Lance: Text → Image",
    "LanceT2V":         "Lance: Text → Video",
    "LanceVideoEdit":   "Lance: Video Edit",
    "LanceImageEdit":   "Lance: Image Edit",
    "LanceX2TImage":    "Lance: Image Understanding (VQA)",
    "LanceX2TVideo":    "Lance: Video Understanding (VQA)",
}
