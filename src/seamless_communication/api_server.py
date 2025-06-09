#!/usr/bin/env python3
"""
FastAPI server for SeamlessExpressive translation service.
Provides REST API endpoints for speech-to-speech translation with prosody preservation.
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter
from seamless_communication.cli.expressivity.predict.pretssel_generator import PretsselGenerator
from seamless_communication.cli.m4t.predict import set_generation_opts
from seamless_communication.inference import Translator
from seamless_communication.models.unity import load_gcmvn_stats, load_unity_unit_tokenizer
from seamless_communication.store import add_gated_assets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Job storage (in production, use Redis or a database)
jobs_store: Dict[str, dict] = {}

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)

# Initialize FastAPI app
app = FastAPI(
    title="SeamlessExpressive API",
    description="REST API for speech-to-speech translation with prosody preservation",
    version="1.0.0"
)

# Global model instances (initialized on startup)
translator = None
pretssel_generator = None
fbank_extractor = None
gcmvn_mean = None
gcmvn_std = None
unit_tokenizer = None

# Device and dtype configuration
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


class TranslationRequest(BaseModel):
    """Translation request parameters"""
    target_lang: str
    expression: Optional[str] = "default"
    duration_factor: Optional[float] = 1.0
    
    
class TranslationJob(BaseModel):
    """Translation job status"""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str
    updated_at: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    target_lang: Optional[str] = None
    expression: Optional[str] = None


def remove_prosody_tokens_from_text(text: str) -> str:
    """Remove prosody tokens from text output"""
    text = text.replace("*", "").replace("=", "")
    text = " ".join(text.split())
    return text


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global translator, pretssel_generator, fbank_extractor
    global gcmvn_mean, gcmvn_std, unit_tokenizer
    
    logger.info("Initializing models...")
    
    # Add gated assets if configured
    gated_model_dir = os.getenv("MODEL_DIR", "/app/models/SeamlessExpressive")
    if gated_model_dir and Path(gated_model_dir).exists():
        add_gated_assets(Path(gated_model_dir))
    
    # Initialize models
    model_name = "seamless_expressivity"
    vocoder_name = "vocoder_pretssel"
    
    unit_tokenizer = load_unity_unit_tokenizer(model_name)
    
    translator = Translator(
        model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )
    
    pretssel_generator = PretsselGenerator(
        vocoder_name,
        vocab_info=unit_tokenizer.vocab_info,
        device=device,
        dtype=dtype,
    )
    
    fbank_extractor = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )
    
    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(vocoder_name)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)
    
    logger.info(f"Models initialized on {device=} with {dtype=}")


def process_translation(job_id: str, input_path: str, output_path: str,
                       target_lang: str, expression: str, duration_factor: float):
    """Process translation in background thread"""
    try:
        # Update job status
        jobs_store[job_id]["status"] = "processing"
        jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        # Load and resample audio
        wav, sample_rate = torchaudio.load(input_path)
        wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)
        wav = wav.transpose(0, 1)
        
        # Extract features
        data = fbank_extractor({"waveform": wav, "sample_rate": 16000})
        fbank = data["fbank"]
        gcmvn_fbank = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
        std, mean = torch.std_mean(fbank, dim=0)
        fbank = fbank.subtract(mean).divide(std)
        
        src = SequenceData(
            seqs=fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([fbank.shape[0]]),
            is_ragged=False,
        )
        src_gcmvn = SequenceData(
            seqs=gcmvn_fbank.unsqueeze(0),
            seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
            is_ragged=False,
        )
        
        # Set generation options
        class Args:
            beam_size = 5
            text_generation_beam_size = 5
            text_generation_max_len_a = 0
            text_generation_max_len_b = 200
            text_unk_blocking = False
            text_generation_ngram_blocking = False
            unit_generation_beam_size = 5
            no_repeat_ngram_size = 4
            unit_generation_max_len_a = 25
            unit_generation_max_len_b = 50
            unit_generation_ngram_blocking = False
            unit_generation_ngram_filtering = False
            algo = "beam"
            max_len_a = 0
            max_len_b = 200
            max_len = None
            min_len = 1
            beam_search_soft_max = False
            beam_search_hard_max = False
            beam_search_stop_on_eos = True
            unit_generation_ngram_filtering = False
            
        text_generation_opts, unit_generation_opts = set_generation_opts(Args())
        
        # Translate
        text_output, unit_output = translator.predict(
            src,
            "s2st",
            target_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=False,
            duration_factor=duration_factor,
            prosody_encoder_input=src_gcmvn,
        )
        
        # Generate speech
        speech_output = pretssel_generator.predict(
            unit_output.units,
            tgt_lang=target_lang,
            prosody_encoder_input=src_gcmvn,
        )
        
        # Log audio properties before saving
        audio_tensor_to_save = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
        logger.info(f"Job {job_id}: Saving audio with sample_rate={speech_output.sample_rate}, shape={audio_tensor_to_save.shape}, dtype={audio_tensor_to_save.dtype}")
        
        # Save output
        torchaudio.save(
            output_path,
            audio_tensor_to_save, # Use the already prepared tensor
            sample_rate=speech_output.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        
        # Update job with success
        jobs_store[job_id]["status"] = "completed"
        jobs_store[job_id]["output_path"] = output_path
        jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs_store[job_id]["status"] = "failed"
        jobs_store[job_id]["error"] = str(e)
        jobs_store[job_id]["updated_at"] = datetime.utcnow().isoformat()


@app.post("/translate", response_model=TranslationJob)
async def create_translation(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    target_lang: str = Form(...),
    expression: str = Form("default"),
    duration_factor: float = Form(1.0)
):
    """
    Create a new translation job.
    
    - **audio**: Audio file to translate (WAV format)
    - **target_lang**: Target language code (e.g., 'spa', 'fra', 'deu')
    - **expression**: Expression style (default, whisper, happy, sad, etc.)
    - **duration_factor**: Duration factor for speech generation
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_dir = Path("/shared_data/api_jobs/input")
    output_dir = Path("/shared_data/api_jobs/output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = input_dir / f"{job_id}.wav"
    output_path = output_dir / f"{job_id}_translated.wav"
    
    # Save uploaded audio
    content = await audio.read()
    with open(input_path, "wb") as f:
        f.write(content)
    
    # Create job entry
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_lang": target_lang,
        "expression": expression,
        "error": None
    }
    jobs_store[job_id] = job
    
    # Submit to background processing
    background_tasks.add_task(
        process_translation,
        job_id, str(input_path), str(output_path),
        target_lang, expression, duration_factor
    )
    
    return TranslationJob(**job)


@app.get("/translate/{job_id}", response_model=TranslationJob)
async def get_translation_status(job_id: str):
    """Get the status of a translation job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TranslationJob(**jobs_store[job_id])


@app.get("/translate/{job_id}/download")
async def download_translation(job_id: str):
    """Download the translated audio file"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_store[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Translation not completed")
    
    output_path = job["output_path"]
    if not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"translated_{job_id}.wav"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": translator is not None,
        "device": str(device),
        "jobs_count": len(jobs_store)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)