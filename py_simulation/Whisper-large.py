import os
import time

import torch

from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    BitsAndBytesConfig
)

from quantize_toolkit import (
    count_params_bnb, 
    inpsect_bnb, 
    apply_custom_quantization_bnb
    )

from evaluate_toolkit import evaluate_asr

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'openai/whisper-large-v3'
    os.chdir(CWD)

    ASR_TASK_CONFIG = {
        'librispeech_clean': {
            'hf_name': 'librispeech_asr',
            'config': 'clean',
            'split': 'test'
        },
        'librispeech_other': {
            'hf_name': 'librispeech_asr',
            'config': 'other',
            'split': 'test'
        },
        'voxpopuli_en': {
            'hf_name': 'facebook/voxpopuli',
            'config': 'en',
            'split': 'validation'
        },
        'ami_ihm': {
            'hf_name': 'edinburghcstr/ami',
            'config': 'ihm',
            'split': 'validation'
        },
        'ami_sdm': {
            'hf_name': 'edinburghcstr/ami',
            'config': 'sdm',
            'split': 'validation'
        }
    }

    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    Whisper = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    ).to(DEVICE)

    time_1 = time.perf_counter()
    fp_score = evaluate_asr(
        model=Whisper,
        processor=processor,
        tasks=ASR_TASK_CONFIG
    )*100
    time_2 = time.perf_counter()
    print(f'Mean WER: {fp_score:.4f} (in { time_2 - time_1} seconds)%')


    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None
    )
    ptq_net = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config
    )
    count_params_bnb(ptq_net, processor)

    time_1 = time.perf_counter()
    ptq_score = evaluate_asr(
        model=ptq_net,
        processor=processor,
        tasks=ASR_TASK_CONFIG
    )*100
    time_2 = time.perf_counter()
    print(f'Mean WER: {ptq_score:.4f} (in { time_2 - time_1} seconds)%')


    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_bnb(ptq_net, processor, prints=True)
    inpsect_bnb(ptq_net, processor)

    time_1 = time.perf_counter()
    aptq_score = evaluate_asr(
        model=ptq_net,
        processor=processor,
        tasks=ASR_TASK_CONFIG
    )*100
    time_2 = time.perf_counter()
    print(f'Mean WER: {aptq_score:.4f} (in { time_2 - time_1} seconds)%')