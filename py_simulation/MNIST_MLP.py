import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torchvision.datasets import MNIST
from torchvision import transforms

from quantize_toolkit import (
    CustomWeightFakeQuantize, 
    train, 
    PTQ_quantize, 
    count_params, 
    apply_custom_quantization, 
    QAT_quantize
)

from evaluate_toolkit import get_MNIST, test_nn

class MLP(nn.Module):
    def __init__(self, q = False):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
                        torch.nn.Linear(28*28,30),
                        torch.nn.ReLU(),
                        torch.nn.Linear(30,10))
        self.q = q
        if q:
          self.quant = QuantStub()
          self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        x = self.layers(x)
        if self.q:
            x = self.dequant(x)
        return x

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.chdir(CWD)

    # Get Data ##########################################################################################################################
    transform = transforms.Compose([
        transforms.ToTensor(),
        torch.flatten])

    train_loader, validation_loader = get_MNIST(transform)

    time_start = time.perf_counter()
    # Train Normally ####################################################################################################################
    print('0. Untrained: /////////////////////////////////////////////////////////////')
    mlp_net = MLP()
    optimizer = optim.SGD(mlp_net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    test_nn(mlp_net, validation_loader)
    
    print('\n1. Trained: ///////////////////////////////////////////////////////////////')
    train(mlp_net, train_loader, optimizer, criterion, epochs=10, name=['MNIST_MLP','main'], load=False, device=DEVICE)
    test_nn(mlp_net, validation_loader, device=DEVICE)

    # PTQ ###############################################################################################################################
    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////')
    ptq_net = MLP(q=True)
    state_dict = mlp_net.state_dict()
    ptq_net.load_state_dict(state_dict, strict=False)
    ptq_qconfig = torch.quantization.default_qconfig
    PTQ_quantize(ptq_net, validation_loader, observer_limit=10, name=['MNIST_MLP','PTQ'], qconfig=ptq_qconfig, load=False, device='cpu')
    test_nn(ptq_net, validation_loader, device='cpu')
    count_params(ptq_net)

    print('\n3. Custom PTQ: ////////////////////////////////////////////////////////////')
    apply_custom_quantization(ptq_net, name=['MNIST_MLP','PTQ_cus'], load=False)
    test_nn(ptq_net, validation_loader, device='cpu')

    # QAT ###############################################################################################################################
    print('\n4. QAT Quantized: /////////////////////////////////////////////////////////')
    qat_net = MLP(q=True)
    optimizer = optim.SGD(qat_net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    qat_qconfig_dict = {'activation' : FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=0,
                                        quant_max=255,
                                        dtype=torch.quint8,
                                        qscheme=torch.per_tensor_affine, reduce_range=False),
                        'weight' : FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-128,
                                        quant_max=127,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric, reduce_range=False),
                        'custom' : None}


    QAT_quantize(qat_net, train_loader, optimizer, criterion, epochs=10, name=['MNIST_MLP','QAT'], qconfig_dict=qat_qconfig_dict, load=False, device=DEVICE)
    test_nn(qat_net, validation_loader, device='cpu')

    print('\n5. Custom post-QAT: ///////////////////////////////////////////////////////')
    apply_custom_quantization(qat_net, name=['MNIST_MLP','QAT_post'], load=False) 
    test_nn(qat_net, validation_loader, device='cpu')

    print('\n6. Custom in-QAT: ///////////////////////////////////////////////////////')
    qat_net_2 = MLP(q=True)
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
    
    QAT_quantize(qat_net_2, train_loader, optimizer, criterion, epochs=3, name=['MNIST_MLP','QAT_in_pre'], qconfig_dict=qat_qconfig_dict, load=False, device=DEVICE)
    test_nn(qat_net_2, validation_loader, device='cpu')
    apply_custom_quantization(qat_net_2, name=['MNIST_MLP','QAT_in'], load=False)
    test_nn(qat_net_2, validation_loader, device='cpu')

    # Finish ############################################################################################################################
    time_end = time.perf_counter()
    print('Done! (in', time_end - time_start,'seconds)')
