"""
Production FastAPI server: SDXL Base + Refiner + IP-Adapter for dog image editing.
Single global model load at startup, async-safe, img2img with identity preservation.
"""
from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
# Plus ViT-H는 전용 image_encoder 필요 → 차원 불일치(514×1664 vs 1280) 방지를 위해 기본 SDXL용 사용
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.safetensors"

NUM_STEPS_BASE = 28
NUM_STEPS_REFINER = 12
GUIDANCE_SCALE = 6.5
STRENGTH_BASE = 0.65
STRENGTH_REFINER = 0.30
IP_ADAPTER_SCALE = 0.7
DTYPE = torch.bfloat16
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# -----------------------------------------------------------------------------
# Global pipelines (loaded once at startup)
# -----------------------------------------------------------------------------
base_pipe = None
refiner_pipe = None
device = None


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch.device("cuda")


def load_pipelines():
    global base_pipe, refiner_pipe, device
    device = get_device()

    # 서브모듈에서만 import해 auto_pipeline(HunyuanDiT/MT5Tokenizer) 로드 방지
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
        StableDiffusionXLImg2ImgPipeline,
    )
    from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Base (SDXL img2img only — avoids AutoPipeline pulling in HunyuanDiT/MT5Tokenizer)
    base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )
    base_pipe.scheduler = EulerDiscreteScheduler.from_config(base_pipe.scheduler.config)
    base_pipe = base_pipe.to(device)
    if hasattr(base_pipe, "enable_xformers_memory_efficient_attention"):
        try:
            base_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    base_pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT,
    )
    base_pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

    # Refiner
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_MODEL,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )
    refiner_pipe.scheduler = EulerDiscreteScheduler.from_config(refiner_pipe.scheduler.config)
    refiner_pipe = refiner_pipe.to(device)
    if hasattr(refiner_pipe, "enable_xformers_memory_efficient_attention"):
        try:
            refiner_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass


def run_inference(image: Image.Image, prompt: str, seed: int) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    w, h = image.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    if w != image.width or h != image.height:
        image = image.resize((w, h), Image.Resampling.LANCZOS)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=DTYPE):
        # Base img2img + IP-Adapter
        out = base_pipe(
            prompt=prompt,
            image=image,
            strength=STRENGTH_BASE,
            num_inference_steps=NUM_STEPS_BASE,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            output_type="latent",
            ip_adapter_image=image,
        )
        latents = out.latents

        # Decode for refiner (refiner expects image in pixel space for img2img)
        decoded = base_pipe.vae.decode(latents / base_pipe.vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        refiner_input = Image.fromarray((decoded * 255).round().astype("uint8"))

        # Refiner
        gen_ref = torch.Generator(device=device).manual_seed(seed + 1)
        refiner_out = refiner_pipe(
            prompt=prompt,
            image=refiner_input,
            strength=STRENGTH_REFINER,
            num_inference_steps=NUM_STEPS_REFINER,
            guidance_scale=GUIDANCE_SCALE,
            generator=gen_ref,
        )
        return refiner_out.images[0]


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="SDXL Dog Image Editor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.on_event("startup")
def startup():
    load_pipelines()


@app.post("/generate")
async def generate(
    file: UploadFile = File(..., description="Dog image"),
    prompt: str = Form(..., description="Edit prompt"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    seed = random.randint(0, 2**31 - 1)
    try:
        out_pil = run_inference(image, prompt.strip(), seed)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    filename = f"out_{seed}_{int(time.time())}.png"
    path = OUTPUTS_DIR / filename
    out_pil.save(path, format="PNG")
    image_url = f"/outputs/{filename}"

    return JSONResponse(
        status_code=200,
        content={"seed": seed, "image_path": str(path), "image_url": image_url},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "cuda": torch.cuda.is_available()}
