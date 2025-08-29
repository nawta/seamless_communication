# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import json
import torch
import torchaudio
from pathlib import Path
from typing import Optional

from fairseq2.data import SequenceData, Collater
from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.nn.padding import get_seqs_and_padding_mask

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Running SeamlessExpressive inference.")
    parser.add_argument("input", type=str, help="Audio WAV file path.")

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
        "--extract_text",
        action="store_true",
        help="Extract only ASR and translation text without audio synthesis",
    )
    parser.add_argument(
        "--inject_text",
        type=str,
        default=None,
        help="Inject custom text for synthesis while preserving speaker characteristics",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.tgt_lang:
        raise Exception("--tgt_lang must be provided for SeamlessExpressive inference.")
    
    if not args.extract_text and args.output_path is None:
        raise Exception("--output_path must be provided when not using --extract_text.")
    
    if args.extract_text and args.inject_text:
        raise Exception("--extract_text and --inject_text cannot be used simultaneously.")
        
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

    wav, sample_rate = torchaudio.load(args.input)
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

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    text_output, unit_output = translator.predict(
        src,
        "s2st",
        args.tgt_lang,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=unit_generation_opts,
        unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
        duration_factor=args.duration_factor,
        prosody_encoder_input=src_gcmvn,
    )

    # Handle extract_text option - only extract text without synthesis
    if args.extract_text:
        # Get ASR result (source text)
        asr_lang = args.src_lang if args.src_lang else "eng"
        logger.info(f"Extracting source text using language: {asr_lang}")
        
        asr_output, _ = translator.predict(
            src,
            "s2tt",  # Using S2TT for ASR
            asr_lang,
            text_generation_opts=text_generation_opts,
        )
        
        # Remove prosody tokens
        source_text = remove_prosody_tokens_from_text(str(asr_output[0]))
        translated_text = remove_prosody_tokens_from_text(str(text_output[0]))
        
        # Output results
        logger.info(f"Source text ({asr_lang}): {source_text}")
        logger.info(f"Translated text ({args.tgt_lang}): {translated_text}")
        
        # Save to JSON file if output_path is provided
        if args.output_path:
            output_data = {
                "source_text": source_text,
                "source_lang": asr_lang,
                "target_text": translated_text,
                "target_lang": args.tgt_lang
            }
            json_path = args.output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Text extraction results saved to: {json_path}")
        
        return  # Exit without audio synthesis
    
    # Handle inject_text option - replace text while preserving voice characteristics
    if args.inject_text:
        original_text = remove_prosody_tokens_from_text(str(text_output[0]))
        logger.info(f"Original translation: {original_text}")
        logger.info(f"Injecting text: {args.inject_text}")
        
        # Create text encoder for the target language
        text_encoder = translator.text_tokenizer.create_encoder(
            task="translation",
            lang=args.tgt_lang,
            mode="source",
            device=translator.device
        )
        
        # Create collater if not exists
        collate = Collater(
            pad_value=translator.text_tokenizer.vocab_info.pad_idx or 0,
            pad_to_multiple=2
        )
        
        # Encode the injected text
        injected_text_data = collate(text_encoder(args.inject_text))
        injected_seqs, injected_padding_mask = get_seqs_and_padding_mask(injected_text_data)
        
        # Generate units from injected text using T2ST (Text to Speech Translation)
        # Import necessary modules for modality
        from seamless_communication.inference.translator import Modality
        
        # Get prediction with injected text
        _, injected_unit_output = translator.get_prediction(
            translator.model,
            translator.text_tokenizer,
            translator.unit_tokenizer,
            injected_seqs,
            injected_padding_mask,
            input_modality=Modality.TEXT,
            output_modality=Modality.SPEECH,
            tgt_lang=args.tgt_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            duration_factor=args.duration_factor,
            prosody_encoder_input=src_gcmvn,  # Preserve original prosody
        )
        
        # Use injected units for synthesis
        unit_output = injected_unit_output
        logger.info(f"Generated units from injected text")
    
    assert unit_output is not None
    speech_output = pretssel_generator.predict(
        unit_output.units,
        tgt_lang=args.tgt_lang,
        prosody_encoder_input=src_gcmvn,
    )

    logger.info(f"Saving expressive translated audio in {args.tgt_lang}")
    torchaudio.save(
        args.output_path,
        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
        sample_rate=speech_output.sample_rate,
    )

    # Display the text that was synthesized
    if args.inject_text:
        logger.info(f"Synthesized injected text: {args.inject_text}")
    else:
        text_out = remove_prosody_tokens_from_text(str(text_output[0]))
        logger.info(f"Translated text in {args.tgt_lang}: {text_out}")


if __name__ == "__main__":
    main()
