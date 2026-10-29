import os
import sys
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

from evaluate_toolkit import evaluate_asr, store_results

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'openai/whisper-large-v3'
    os.chdir(CWD)

##### Setup Result Reporting ############################################################################################################
    model_name = 'Whisper-large'
    result_data = {}
    try:
        assert(len(sys.argv) == 2)
        run_id = 'run_' + sys.argv[1]
        torch.manual_seed(hash(sys.argv[1]))
    except:
        print('ERROR: wrong system arguments, provide a \"run id\" only.')
        exit()

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

    time_start = time.perf_counter()
##### Pretrained Network ################################################################################################################
    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    Whisper = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    ).to(DEVICE)

    time_1 = time.perf_counter()
    result_data['STD'] = evaluate_asr(
        model=Whisper,
        processor=processor,
        tasks=ASR_TASK_CONFIG
    )
    time_2 = time.perf_counter()
    print(f'Mean WER: {result_data['STD']['TOTAL']:.4f} (in { time_2 - time_1} seconds)%')

##### PTQ ###############################################################################################################################
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
    count_params_bnb(ptq_net, processor, seed=hash(sys.argv[1]))

    time_1 = time.perf_counter()
    result_data['PTQ'] = evaluate_asr(
        model=ptq_net,
        processor=processor,
        tasks=ASR_TASK_CONFIG
    )
    time_2 = time.perf_counter()
    print(f'Mean WER: {result_data['PTQ']['TOTAL']:.4f} (in { time_2 - time_1} seconds)%')

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_bnb(ptq_net, processor, seed=hash(sys.argv[1]), prints=True)
    inpsect_bnb(ptq_net, processor, seed=hash(sys.argv[1]))

    time_1 = time.perf_counter()
    result_data['AxCQ_PTQ'] = evaluate_asr(
        model=ptq_net,
        processor=processor,
        tasks=ASR_TASK_CONFIG
    )
    time_2 = time.perf_counter()
    print(f'Mean WER: {result_data['AxCQ_PTQ']['TOTAL']:.4f} (in { time_2 - time_1} seconds)%')

##### Finish ############################################################################################################################
    time_end = time.perf_counter()
    store_results(result_data, model_name, run_id)
    print('Done! (in', time_end - time_start,'seconds)')