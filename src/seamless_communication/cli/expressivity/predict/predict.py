# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter

from seamless_communication.cli.expressivity.predict.pretssel_generator import (
    PretsselGenerator,
)
from seamless_communication.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)
from seamless_communication.inference import Translator
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.store import add_gated_assets


AUDIO_SAMPLE_RATE = 16000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def remove_prosody_tokens_from_text(text: str) -> str:
    # filter out prosody tokens, there is only emphasis '*', and pause '='
    text = text.replace("*", "").replace("=", "")
    text = " ".join(text.split())
    return text


def process_audio_file(
    input_path: str,
    output_path: str,
    tgt_lang: str,
    translator: Translator,
    pretssel_generator: PretsselGenerator,
    fbank_extractor: WaveformToFbankConverter,
    gcmvn_mean: torch.Tensor,
    gcmvn_std: torch.Tensor,
    text_generation_opts: Dict,
    unit_generation_opts: Dict,
    unit_generation_ngram_filtering: bool,
    duration_factor: float,
) -> Tuple[str, float]:
    """Process a single audio file and return the translated text and duration."""
    wav, sample_rate = torchaudio.load(input_path)
    wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)
    wav = wav.transpose(0, 1)

    data = fbank_extractor(
        {
            "waveform": wav,
            "sample_rate": 16000,
        }
    )
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

    text_output, unit_output = translator.predict(
        src,
        "s2st",
        tgt_lang,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=unit_generation_opts,
        unit_generation_ngram_filtering=unit_generation_ngram_filtering,
        duration_factor=duration_factor,
        prosody_encoder_input=src_gcmvn,
    )

    assert unit_output is not None
    speech_output = pretssel_generator.predict(
        unit_output.units,
        tgt_lang=tgt_lang,
        prosody_encoder_input=src_gcmvn,
    )

    logger.info(f"Saving expressive translated audio in {tgt_lang} to {output_path}")
    torchaudio.save(
        output_path,
        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
        sample_rate=speech_output.sample_rate,
    )

    text_out = remove_prosody_tokens_from_text(str(text_output[0]))
    logger.info(f"Translated text in {tgt_lang}: {text_out}")
    
    output_wav, _ = torchaudio.load(output_path)
    output_duration = output_wav.shape[1] / AUDIO_SAMPLE_RATE
    
    return text_out, output_duration


def process_audio_segments(
    segments_json_path: str,
    output_dir: str,
    tgt_lang: str,
    translator: Translator,
    pretssel_generator: PretsselGenerator,
    fbank_extractor: WaveformToFbankConverter,
    gcmvn_mean: torch.Tensor,
    gcmvn_std: torch.Tensor,
    text_generation_opts: Dict,
    unit_generation_opts: Dict,
    unit_generation_ngram_filtering: bool,
    duration_factor: float,
) -> List[Dict]:
    """Process audio segments from a JSON file and return metadata about the processed segments."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(segments_json_path, 'r') as f:
        segments = json.load(f)
    
    processed_segments = []
    
    for segment in segments:
        track_id = segment["track_id"]
        input_path = segment["audio_file"]
        output_filename = f"translated_audio_{track_id:06d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.info(f"Processing segment {track_id} from {input_path}")
        
        try:
            text, duration = process_audio_file(
                input_path=input_path,
                output_path=output_path,
                tgt_lang=tgt_lang,
                translator=translator,
                pretssel_generator=pretssel_generator,
                fbank_extractor=fbank_extractor,
                gcmvn_mean=gcmvn_mean,
                gcmvn_std=gcmvn_std,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                unit_generation_ngram_filtering=unit_generation_ngram_filtering,
                duration_factor=duration_factor,
            )
            
            processed_segment = segment.copy()
            processed_segment.update({
                "translated_audio_file": output_path,
                "translated_text": text,
                "translated_duration": duration,
                "original_duration": segment["end_time"] - segment["start_time"],
                "duration_ratio": duration / (segment["end_time"] - segment["start_time"]),
            })
            processed_segments.append(processed_segment)
            
        except Exception as e:
            logger.error(f"Error processing segment {track_id}: {e}")
    
    metadata_path = os.path.join(output_dir, "translated_segments.json")
    with open(metadata_path, 'w') as f:
        json.dump(processed_segments, f, indent=2)
    
    logger.info(f"Saved translated segments metadata to {metadata_path}")
    
    return processed_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Running SeamlessExpressive inference.")
    parser.add_argument("input", type=str, help="Audio WAV file path or segments JSON path.")
    
    parser = add_inference_arguments(parser)
    parser.add_argument(
        "--gated-model-dir",
        type=Path,
        required=False,
        help="SeamlessExpressive model directory.",
    )
    parser.add_argument(
        "--duration_factor",
        type=float,
        help="The duration factor for NAR T2U model.",
        default=1.0,
    )
    parser.add_argument(
        "--segments_mode",
        action="store_true",
        help="Process audio segments from a JSON file instead of a single audio file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for translated audio segments (required in segments mode).",
    )
    args = parser.parse_args()

    if args.segments_mode:
        if not args.output_dir:
            raise Exception("--output_dir must be provided in segments mode.")
        if not args.tgt_lang:
            raise Exception("--tgt_lang must be provided in segments mode.")
    else:
        if not args.tgt_lang or args.output_path is None:
            raise Exception(
                "--tgt_lang, --output_path must be provided for single file inference."
            )
        
    if args.gated_model_dir:
        add_gated_assets(args.gated_model_dir)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running inference on {device=} with {dtype=}.")

    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    translator = Translator(
        args.model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )

    pretssel_generator = PretsselGenerator(
        args.vocoder_name,
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

    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(args.vocoder_name)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    if args.segments_mode:
        logger.info(f"Processing audio segments from {args.input}")
        process_audio_segments(
            segments_json_path=args.input,
            output_dir=args.output_dir,
            tgt_lang=args.tgt_lang,
            translator=translator,
            pretssel_generator=pretssel_generator,
            fbank_extractor=fbank_extractor,
            gcmvn_mean=gcmvn_mean,
            gcmvn_std=gcmvn_std,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            duration_factor=args.duration_factor,
        )
    else:
        logger.info(f"Processing single audio file {args.input}")
        process_audio_file(
            input_path=args.input,
            output_path=args.output_path,
            tgt_lang=args.tgt_lang,
            translator=translator,
            pretssel_generator=pretssel_generator,
            fbank_extractor=fbank_extractor,
            gcmvn_mean=gcmvn_mean,
            gcmvn_std=gcmvn_std,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            duration_factor=args.duration_factor,
        )


if __name__ == "__main__":
    main()
