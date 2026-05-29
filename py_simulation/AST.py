import os
import time
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoFeatureExtractor, ASTForAudioClassification, BitsAndBytesConfig

from quantize_toolkit import (
    count_params_bnb, 
    inpsect_bnb, 
    apply_custom_quantization_bnb
    )

from evaluate_toolkit import (
    get_audioset,
    eval_audioset
)

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'MIT/ast-finetuned-audioset-10-10-0.448'
    os.chdir(CWD)

    audioset = get_audioset()

    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    AST = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    ).to(DEVICE).eval()

    time_1 = time.perf_counter()
    # fp_score = eval_audioset(AST, feature_extractor, audioset)
    time_2 = time.perf_counter()
    print(f'\n(in { time_2 - time_1} seconds)\n')

    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    ptq_net = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config
    ).eval()

    time_1 = time.perf_counter()
    # ptq_score = eval_audioset(ptq_net, feature_extractor, audioset)
    time_2 = time.perf_counter()
    print(f'\n(in { time_2 - time_1} seconds)\n')

    count_params_bnb(ptq_net, feature_extractor)

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_bnb(ptq_net, feature_extractor, prints=True)
    inpsect_bnb(ptq_net, feature_extractor)

    time_1 = time.perf_counter()
    aptq_score = eval_audioset(ptq_net, feature_extractor, audioset)
    time_2 = time.perf_counter()
    print(f'\n(in { time_2 - time_1} seconds)\n')