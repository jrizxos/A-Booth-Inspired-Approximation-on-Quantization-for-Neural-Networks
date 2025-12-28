import os
import time
import torch
from copy import deepcopy
from torch import nn
from torch import optim
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models.quantization import googlenet as q_googlenet

from quantize_toolkit import get_ImageNet, get_CIFAR, CustomWeightFakeQuantize, train, test, PTQ_quantize, apply_custom_quantization, QAT_quantize

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.chdir(CWD)

    IMAGENET_DATASET = False

##### Get Data ##########################################################################################################################
    if(IMAGENET_DATASET):
        train_loader, validation_loader = get_ImageNet()
    else:
        train_loader, validation_loader = get_CIFAR()

    time_start = time.perf_counter()
##### Pretrained Network ################################################################################################################
    print('1. Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    GoogLeNet = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    time_1 = time.perf_counter()
    if(IMAGENET_DATASET):
        model_name = 'GoogLeNet'
    else:
        print('Finetuning for CIFAR dataset...')
        model_name = 'GoogLeNet_CIFAR'
        GoogLeNet.fc = nn.Linear(GoogLeNet.fc.in_features, 10)
        GoogLeNet = GoogLeNet.to(DEVICE)
        optimizer = optim.SGD(GoogLeNet.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        GoogLeNet.train()
        train(GoogLeNet, train_loader, optimizer, criterion, epochs=10, name=[model_name,'main'], load=False, device=DEVICE)
    GoogLeNet.eval()
    test(GoogLeNet, validation_loader, device=DEVICE)
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

##### PTQ ###############################################################################################################################
    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        ptq_net = q_googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    else:
        print('Finetuning for CIFAR dataset...')
        q_GoogLeNet = q_googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        q_GoogLeNet.fc = nn.Linear(q_GoogLeNet.fc.in_features, 10)
        q_GoogLeNet = q_GoogLeNet.to(DEVICE)
        optimizer = optim.SGD(q_GoogLeNet.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        q_GoogLeNet.train()
        train(q_GoogLeNet, train_loader, optimizer, criterion, epochs=10, name=[model_name,'quantizable'], load=False, device=DEVICE)
        ptq_net = deepcopy(q_GoogLeNet)
    ptq_qconfig = torch.quantization.default_qconfig
    time_1 = time.perf_counter()
    PTQ_quantize(ptq_net, validation_loader, observer_limit=10, name=[model_name,'PTQ'], qconfig=ptq_qconfig, load=False, device='cpu')
    test(ptq_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

    print('\n3. Custom PTQ: ////////////////////////////////////////////////////////////\n',time.asctime())    
    time_1 = time.perf_counter()       
    apply_custom_quantization(ptq_net, name=[model_name,'PTQ_cus'], load=False)
    test(ptq_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

##### QAT ###############################################################################################################################
    print('\n4. QAT Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        qat_net = q_googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    else:
        qat_net = deepcopy(q_GoogLeNet)
    optimizer = optim.SGD(qat_net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    qat_qconfig_dict = {'activation' : FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=0,
                                        quant_max=255,
                                        dtype=torch.quint8,
                                        qscheme=torch.per_tensor_affine),
                        'weight' : FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-128,
                                        quant_max=127,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric),
                        'custom' : None}

    time_1 = time.perf_counter()
    QAT_quantize(qat_net, train_loader, optimizer, criterion, epochs=10, name=[model_name,'QAT'], qconfig_dict=qat_qconfig_dict, load=False, device=DEVICE)
    test(qat_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

    print('\n5. Custom post-QAT: ///////////////////////////////////////////////////////\n',time.asctime())
    time_1 = time.perf_counter()
    apply_custom_quantization(qat_net, name=[model_name,'QAT_post'], load=False)
    test(qat_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

    print('\n6. Custom in-QAT: ///////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        qat_net_2 = q_googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    else:
        qat_net_2 = deepcopy(q_GoogLeNet)
    optimizer = optim.SGD(qat_net_2.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    qat_qconfig_dict = {'activation' : FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=0,
                                        quant_max=255,
                                        dtype=torch.quint8,
                                        qscheme=torch.per_tensor_affine),
                        'weight' : FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-128,
                                        quant_max=127,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric),
                        'custom' : CustomWeightFakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-128,
                                        quant_max=127,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric)}
    
    time_1 = time.perf_counter()
    QAT_quantize(qat_net_2, train_loader, optimizer, criterion, epochs=3, name=[model_name,'QAT_in_pre'], qconfig_dict=qat_qconfig_dict, load=False, device=DEVICE)
    test(qat_net_2, validation_loader, device='cpu')
    apply_custom_quantization(qat_net_2, name=[model_name,'QAT_in'], load=False)
    test(qat_net_2, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

##### Finish ############################################################################################################################
    time_end = time.perf_counter()
    print('Done! (in', time_end - time_start,'seconds)')