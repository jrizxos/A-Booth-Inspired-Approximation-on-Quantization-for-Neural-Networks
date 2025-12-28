import os
import time
import torch
from copy import deepcopy
from torch import nn
from torch import optim
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights

from quantize_toolkit import get_ImageNet, get_CIFAR, QuantizableModel, CustomWeightFakeQuantize, train, test, PTQ_quantize, apply_custom_quantization, QAT_quantize

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
    print('1.1 Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    time_1 = time.perf_counter()
    if(IMAGENET_DATASET):
        model_name = 'VGG16'
    else:
        print('Finetuning for CIFAR dataset...')
        model_name = 'VGG16_CIFAR'
        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, 10)
        vgg = vgg.to(DEVICE)
        optimizer = optim.SGD(vgg.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        vgg.train()
        train(vgg, train_loader, optimizer, criterion, epochs=10, name=[model_name,'main'], load=False, device=DEVICE)
    vgg.eval()
    test(vgg, validation_loader, device=DEVICE)
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

##### PTQ ###############################################################################################################################
    print('\n1.2 PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        ptq_net = QuantizableModel(vgg16(weights=VGG16_Weights.IMAGENET1K_V1))
    else:
        ptq_net = QuantizableModel(deepcopy(vgg))
        ptq_net = ptq_net.to('cpu')
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
    print('\n1.4 QAT Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        qat_net = QuantizableModel(vgg16(weights=VGG16_Weights.IMAGENET1K_V1))
    else:
        qat_net = QuantizableModel(deepcopy(vgg))
        qat_net = qat_net.to('cpu')
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

    print('\n1.5 Custom post-QAT: ///////////////////////////////////////////////////////\n',time.asctime())
    time_1 = time.perf_counter()
    apply_custom_quantization(qat_net, name=[model_name,'QAT_post'], load=False)
    test(qat_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

    print('\n1.6 Custom in-QAT: ///////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        qat_net_2 = QuantizableModel(vgg16(weights=VGG16_Weights.IMAGENET1K_V1))
    else:
        qat_net_2 = QuantizableModel(deepcopy(vgg))
        qat_net_2 = qat_net_2.to('cpu')
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


##### Batch Norm Version ################################################################################################################
##### Pretrained Network ################################################################################################################
    print('2.1 Trained: ///////////////////////////////////////////////////////////////\n',time.asctime())
    vgg = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    time_1 = time.perf_counter()
    if(IMAGENET_DATASET):
        model_name = 'VGG16_BN'
    else:
        print('Finetuning for CIFAR dataset...')
        model_name = 'VGG16_BN_CIFAR'
        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, 10)
        vgg = vgg.to(DEVICE)
        optimizer = optim.SGD(vgg.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        vgg.train()
        train(vgg, train_loader, optimizer, criterion, epochs=10, name=[model_name,'main'], load=False, device=DEVICE)
    vgg.eval()
    time_1 = time.perf_counter()
    test(vgg, validation_loader, device=DEVICE)
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

##### PTQ ###############################################################################################################################
    print('\n2.2 PTQ Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        ptq_net = QuantizableModel(vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1))
    else:
        ptq_net = QuantizableModel(deepcopy(vgg))
        ptq_net = ptq_net.to('cpu')
    ptq_qconfig = torch.quantization.default_qconfig
    time_1 = time.perf_counter()
    PTQ_quantize(ptq_net, validation_loader, observer_limit=10, name=[model_name,'PTQ'], qconfig=ptq_qconfig, load=False, device='cpu')
    test(ptq_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')

    print('\n2.3 Custom PTQ: ////////////////////////////////////////////////////////////\n',time.asctime())    
    time_1 = time.perf_counter()       
    apply_custom_quantization(ptq_net, name=[model_name,'PTQ_cus'], load=False)
    test(ptq_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

##### QAT ###############################################################################################################################
    print('\n2.4 QAT Quantized: /////////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        qat_net = QuantizableModel(vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1))
    else:
        qat_net = QuantizableModel(deepcopy(vgg))
        qat_net = qat_net.to('cpu')
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

    print('\n2.5 Custom post-QAT: ///////////////////////////////////////////////////////\n',time.asctime())
    time_1 = time.perf_counter()
    apply_custom_quantization(qat_net, name=[model_name,'QAT_post'], load=False)
    test(qat_net, validation_loader, device='cpu')
    time_2 = time.perf_counter()
    print('(in', time_2 - time_1,'seconds)\n')    

    print('\n2.6 Custom in-QAT: ///////////////////////////////////////////////////////\n',time.asctime())
    if(IMAGENET_DATASET):
        qat_net_2 = QuantizableModel(vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1))
    else:
        qat_net_2 = QuantizableModel(deepcopy(vgg))
        qat_net_2 = qat_net_2.to('cpu')
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