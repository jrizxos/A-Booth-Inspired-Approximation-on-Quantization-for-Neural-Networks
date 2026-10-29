import os
import sys
import time

import torch

from torchvision import transforms

from transformers import ViTImageProcessor, ViTForImageClassification, BitsAndBytesConfig

from quantize_toolkit import (
    count_params_bnb, 
    inpsect_bnb, 
    apply_custom_quantization_bnb
)

from evaluate_toolkit import get_ImageNet, test_vit, store_results

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.chdir(CWD)
    MODEL_NAME = 'google/vit-base-patch16-224'

##### Setup Result Reporting ############################################################################################################
    model_name = 'ViT'
    result_data = {}
    try:
        assert(len(sys.argv) == 2)
        run_id = 'run_' + sys.argv[1]
        torch.manual_seed(hash(sys.argv[1]))
    except:
        print('ERROR: wrong system arguments, provide a \"run id\" only.')
        exit()

##### Get Data ##########################################################################################################################
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_loader, validation_loader = get_ImageNet(transform)

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    time_start = time.perf_counter()
##### Pretrained Network ################################################################################################################
    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    ViT = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        device_map='auto'
    ).to(DEVICE)
    time_1 = time.perf_counter()
    result_data['STD'] = test_vit(ViT, validation_loader, processor, device=DEVICE)
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

##### PTQ ###############################################################################################################################
    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None
        )
    ptq_net = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto'
    )
    count_params_bnb(ptq_net, processor, seed=hash(sys.argv[1]))

    time_1 = time.perf_counter()
    result_data['PTQ'] = test_vit(ptq_net, validation_loader, processor, device=DEVICE)
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

    print('\n3.  Custom PTQ: ///////////////////////////////////////////////////////////\n',time.asctime()) 
    apply_custom_quantization_bnb(ptq_net, processor, seed=hash(sys.argv[1]), prints=True)
    inpsect_bnb(ptq_net, processor, seed=hash(sys.argv[1]))

    time_1 = time.perf_counter()
    result_data['AxCQ_PTQ'] = test_vit(ptq_net, validation_loader, processor, device=DEVICE)
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

##### Finish ############################################################################################################################
    time_end = time.perf_counter()
    store_results(result_data, model_name, run_id)
    print('Done! (in', time_end - time_start,'seconds)')