import os
import sys
import time

import torch

from transformers import AutoModelForSequenceClassification

from torchao.quantization import quantize_
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig

from quantize_toolkit import (
    inspect_torchao, 
    count_params_torchao, 
    apply_custom_quantization_torchao
)

from evaluate_toolkit import evaluate_glue, store_results

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.chdir(CWD)

##### Setup Result Reporting ############################################################################################################
    model_name = 'BERT-large'
    result_data = {}
    try:
        assert(len(sys.argv) == 2)
        run_id = 'run_' + sys.argv[1]
        torch.manual_seed(hash(sys.argv[1]))
    except:
        print('ERROR: wrong system arguments, provide a \"run id\" only.')
        exit()

    TASKS_NUM = 6
    GLUE_TASK_CONFIG = {
        # single-sentence classification
        'cola': {
            'repo' : 'SarielSinLuo/bert-large-uncased-finetuned-cola',
            'text_columns': ('sentence',),
            'eval_func': 'matthews',
        },
        'sst2': {
            'repo' : 'pranav4205/bert-large-uncased-finetuned-sst2',
            'text_columns': ('sentence',),
            'eval_func': 'accuracy',
        },

        # sentence-pair classification
        'mrpc': {
            'repo' : 'SarielSinLuo/bert-large-uncased-finetuned-mrpc',
            'text_columns': ('sentence1', 'sentence2'),
            'eval_func': 'accuracy',
        },
        'qqp': {
            'repo' : None,
            'text_columns': ('question1', 'question2'),
            'eval_func': 'accuracy',
        },
        'qnli': {
            'repo' : None,
            'text_columns': ('question', 'sentence'),
            'eval_func': 'accuracy',
        },
        'rte': {
            'repo' : 'SarielSinLuo/bert-large-uncased-finetuned-rte',
            'text_columns': ('sentence1', 'sentence2'),
            'eval_func': 'accuracy',
        },
        'wnli': {
            'repo' : 'gchhablani/bert-large-cased-finetuned-wnli',
            'text_columns': ('sentence1', 'sentence2'),
            'eval_func': 'accuracy',
        },

        # regression
        'stsb': {
            'repo' : 'SarielSinLuo/bert-large-uncased-finetuned-stsb',
            'text_columns': ('sentence1', 'sentence2'),
            'eval_func': 'pearson_spearman',
        },
    }

    time_start = time.perf_counter()
##### Pretrained Network ################################################################################################################
    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    fp_score = 0
    time_1 = time.perf_counter()
    result_data['STD'] = {}
    for task_name in GLUE_TASK_CONFIG:
        task_cfg = GLUE_TASK_CONFIG[task_name]
        if task_cfg['repo']:
            print(f'{task_name}')
            BERT = AutoModelForSequenceClassification.from_pretrained(task_cfg['repo']).to(DEVICE)
            BERT.eval()
            fp_metric = evaluate_glue(BERT, task_name, task_cfg, DEVICE)
            fp_score += fp_metric
            result_data['STD'][task_name] = fp_metric
            print(f'\t -> {task_cfg['eval_func']}: {fp_metric:.4f}')
    time_2 = time.perf_counter()
    total = fp_score/TASKS_NUM
    result_data['STD']['TOTAL'] = total
    print(f'\nTotal score: {total:.4f} (in { time_2 - time_1} seconds)\n')
    count_params_torchao(BERT)
    print()

##### PTQ ###############################################################################################################################
    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    ptq_score = 0
    time_1 = time.perf_counter()
    result_data['PTQ'] = {}
    for task_name in GLUE_TASK_CONFIG:
        task_cfg = GLUE_TASK_CONFIG[task_name]
        if task_cfg['repo']:
            print(f'{task_name}')
            ptq_net = AutoModelForSequenceClassification.from_pretrained(task_cfg['repo']).to(DEVICE)
            config = Int8DynamicActivationInt8WeightConfig(version=2)
            quantize_(ptq_net, config)
            ptq_net.eval()
            ptq_metric = evaluate_glue(ptq_net, task_name, task_cfg, DEVICE)
            ptq_score += ptq_metric
            result_data['PTQ'][task_name] = ptq_metric
            print(f'\t -> {task_cfg['eval_func']}: {ptq_metric:.4f}')
    time_2 = time.perf_counter()
    total = ptq_score/TASKS_NUM
    result_data['PTQ']['TOTAL'] = total
    print(f'\nTotal score: {total:.4f}  (in { time_2 - time_1} seconds)\n')
    count_params_torchao(ptq_net)
    print()

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime())
    ptq_score = 0
    time_1 = time.perf_counter()
    result_data['AxCQ_PTQ'] = {}
    for task_name in GLUE_TASK_CONFIG:
        task_cfg = GLUE_TASK_CONFIG[task_name]
        if task_cfg['repo']:
            print(f'{task_name}')
            ptq_net = AutoModelForSequenceClassification.from_pretrained(task_cfg['repo']).to(DEVICE)
            config = Int8DynamicActivationInt8WeightConfig(version=2)
            quantize_(ptq_net, config)
            apply_custom_quantization_torchao(ptq_net, DEVICE, prints=True)
            inspect_torchao(ptq_net, DEVICE, out_mode=0)
            ptq_net.eval()
            ptq_metric = evaluate_glue(ptq_net, task_name, task_cfg, DEVICE)
            ptq_score += ptq_metric
            result_data['AxCQ_PTQ'][task_name] = ptq_metric
            print(f'\t -> {task_cfg['eval_func']}: {ptq_metric:.4f}')
    time_2 = time.perf_counter()
    total = ptq_score/TASKS_NUM
    result_data['AxCQ_PTQ']['TOTAL'] = total
    print(f'\nTotal score: {total:.4f}  (in { time_2 - time_1} seconds)\n')

##### Finish ############################################################################################################################
    time_end = time.perf_counter()
    store_results(result_data, model_name, run_id)
    print('Done! (in', time_end - time_start,'seconds)')