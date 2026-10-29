import os
import sys
import time

import torch

from transformers import AutoModelForCausalLM, TorchAoConfig

from torchao.quantization import  Int8DynamicActivationInt8WeightConfig, FqnToConfig

from lm_eval.models.huggingface import HFLM

from quantize_toolkit import (
    inspect_torchao, 
    count_params_torchao, 
    apply_custom_quantization_torchao
)

from evaluate_toolkit import (
    eval_lm_eval_existing,
    eval_lm_eval_custom, 
    store_results
)

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'Corianas/Tiny-Moe'
    os.chdir(CWD)

##### Setup Result Reporting ############################################################################################################
    model_name = 'Tiny-Moe'
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
    qconfig = TorchAoConfig(quant_type=Int8DynamicActivationInt8WeightConfig())
    ptq_net = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                    dtype="auto",
                                                    device_map="auto",
                                                    quantization_config=qconfig)
    model_hflm = HFLM(pretrained=ptq_net)

    time_1 = time.perf_counter()
    result_data['PTQ'] = eval_lm_eval_custom(model_hflm, NLP_TASK_CONFIG, DEVICE, seed=hash(sys.argv[1]))
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {result_data['PTQ']['TOTAL']} (in { time_2 - time_1} seconds)\n')

    count_params_torchao(ptq_net)

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_torchao(ptq_net, DEVICE, prints=True)
    model_hflm = HFLM(pretrained=ptq_net)

    time_1 = time.perf_counter()
    result_data['AxCQ_PTQ'] = eval_lm_eval_custom(model_hflm, NLP_TASK_CONFIG, DEVICE, seed=hash(sys.argv[1]))
    time_2 = time.perf_counter()
    print(f'\nAverage accuracy: {result_data['AxCQ_PTQ']['TOTAL']}  (in { time_2 - time_1} seconds)\n')

    inspect_torchao(ptq_net, DEVICE, out_mode=0)

##### Finish ############################################################################################################################
    time_end = time.perf_counter()
    store_results(result_data, model_name, run_id)
    print('Done! (in', time_end - time_start,'seconds)')