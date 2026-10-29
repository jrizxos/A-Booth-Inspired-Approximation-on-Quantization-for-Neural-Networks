import os
import sys
import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from lm_eval.models.huggingface import HFLM

from quantize_toolkit import (
    count_params_bnb, 
    inpsect_bnb, 
    apply_custom_quantization_bnb
)

from evaluate_toolkit import (
    eval_lm_eval_existing,
    eval_lm_eval_custom, 
    store_results
)

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'openai-community/gpt2-xl'
    os.chdir(CWD)

##### Setup Result Reporting ############################################################################################################
    model_name = 'GPT2-xl'
    result_data = {}
    try:
        assert(len(sys.argv) == 2)
        run_id = 'run_' + sys.argv[1]
        torch.manual_seed(hash(sys.argv[1]))
    except:
        print('ERROR: wrong system arguments, provide a \"run id\" only.')
        exit()

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

    time_start = time.perf_counter()
##### Pretrained Network ################################################################################################################
    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    time_1 = time.perf_counter()
    result_data['STD'] = eval_lm_eval_existing(MODEL_NAME, NLP_TASK_CONFIG, DEVICE, seed=hash(sys.argv[1]))
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {result_data['STD']['TOTAL']} (in { time_2 - time_1} seconds)\n')

##### PTQ ###############################################################################################################################
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
    model_hflm = HFLM(pretrained=ptq_net)

    time_1 = time.perf_counter()
    result_data['PTQ'] = eval_lm_eval_custom(model_hflm, NLP_TASK_CONFIG, DEVICE, seed=hash(sys.argv[1]))
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {result_data['PTQ']['TOTAL']} (in { time_2 - time_1} seconds)\n')

    count_params_bnb(ptq_net, tokenizer, seed=hash(sys.argv[1]))

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_bnb(ptq_net, tokenizer, seed=hash(sys.argv[1]), prints=True)
    inpsect_bnb(ptq_net, tokenizer, seed=hash(sys.argv[1]))
    model_hflm = HFLM(pretrained=ptq_net)

    time_1 = time.perf_counter()
    result_data['AxCQ_PTQ'] = eval_lm_eval_custom(model_hflm, NLP_TASK_CONFIG, DEVICE, seed=hash(sys.argv[1]))
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {result_data['AxCQ_PTQ']['TOTAL']}  (in { time_2 - time_1} seconds)\n')

##### Finish ############################################################################################################################
    time_end = time.perf_counter()
    store_results(result_data, model_name, run_id)
    print('Done! (in', time_end - time_start,'seconds)')