import os
import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from quantize_toolkit import (
    count_params_bnb, 
    inpsect_bnb, 
    apply_custom_quantization_bnb
    )

from evaluate_toolkit import (
    eval_lm_eval_existing,
    eval_lm_eval_custom
)

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'bigscience/bloom-7b1'
    os.chdir(CWD)

    NLP_TASK_CONFIG = [
        'piqa',
        'hellaswag',
        'winogrande',
        'arc_easy',
        'arc_challenge',
        'wic',
        'rte',
        'multirc',
    ] 

    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    time_1 = time.perf_counter()
    fp_score = eval_lm_eval_existing(MODEL_NAME, NLP_TASK_CONFIG, DEVICE)
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {fp_score} in { time_2 - time_1} seconds)\n')

    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    ptq_net = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto'
    )

    time_1 = time.perf_counter()
    ptq_score = eval_lm_eval_custom(ptq_net, NLP_TASK_CONFIG, DEVICE)
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {ptq_score} (in { time_2 - time_1} seconds)\n')

    count_params_bnb(ptq_net, tokenizer)

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_bnb(ptq_net, tokenizer, prints=True)
    inpsect_bnb(ptq_net, tokenizer)

    time_1 = time.perf_counter()
    aptq_score = eval_lm_eval_custom(ptq_net, NLP_TASK_CONFIG, DEVICE)
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {aptq_score}  (in { time_2 - time_1} seconds)\n')