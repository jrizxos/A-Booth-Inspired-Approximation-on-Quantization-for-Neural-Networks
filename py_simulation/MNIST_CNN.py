import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
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

class CNN(nn.Module):
    def __init__(self, q = False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout_fc = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.q = q
        if q:
          self.quant = QuantStub()
          self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.reshape(x.shape[0], 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        if self.q:
            x = self.dequant(x)

        x = self.logsoftmax(x)
        return x

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.chdir(CWD)

    # Get Data ##########################################################################################################################
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_loader, validation_loader = get_MNIST(transform)

    time_start = time.perf_counter()
    # Train Normally ####################################################################################################################
    print('0. Untrained: /////////////////////////////////////////////////////////////')
    cnn_net = CNN()
    optimizer = optim.SGD(cnn_net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    test_nn(cnn_net, validation_loader)
    
    print('\n1. Trained: ///////////////////////////////////////////////////////////////')
    train(cnn_net, train_loader, optimizer, criterion, epochs=3, name=['MNIST_CNN','main'], load=False, device=DEVICE)
    test_nn(cnn_net, validation_loader, device=DEVICE)

    # PTQ ###############################################################################################################################
    print('\n2. PTQ Quantized: /////////////////////////////////////////////////////////')
    ptq_net = CNN(q=True)
    state_dict = cnn_net.state_dict()
    ptq_net.load_state_dict(state_dict, strict=False)
    ptq_qconfig = torch.quantization.default_qconfig
    PTQ_quantize(ptq_net, validation_loader, observer_limit=10, name=['MNIST_CNN','PTQ'], qconfig=ptq_qconfig, load=False, device='cpu')
    test_nn(ptq_net, validation_loader, device='cpu')
    count_params(ptq_net)

    print('\n3. Custom PTQ: ////////////////////////////////////////////////////////////')
    apply_custom_quantization(ptq_net, name=['MNIST_CNN','PTQ_cus'], load=False)
    test_nn(ptq_net, validation_loader, device='cpu')

    # QAT ###############################################################################################################################
    print('\n4. QAT Quantized: /////////////////////////////////////////////////////////')
    qat_net = CNN(q=True)
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


    QAT_quantize(qat_net, train_loader, optimizer, criterion, epochs=3, name=['MNIST_CNN','QAT'], qconfig_dict=qat_qconfig_dict, load=False, device=DEVICE)
    test_nn(qat_net, validation_loader, device='cpu')

    print('\n5. Custom post-QAT: ///////////////////////////////////////////////////////')
    apply_custom_quantization(qat_net, name=['MNIST_CNN','QAT_post'], load=False) 
    test_nn(qat_net, validation_loader, device='cpu')

    print('\n6. Custom in-QAT: ///////////////////////////////////////////////////////')
    qat_net_2 = CNN(q=True)
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
    
    QAT_quantize(qat_net_2, train_loader, optimizer, criterion, epochs=3, name=['MNIST_CNN','QAT_in_pre'], qconfig_dict=qat_qconfig_dict, load=False, device=DEVICE)
    test_nn(qat_net_2, validation_loader, device='cpu')
    apply_custom_quantization(qat_net_2, name=['MNIST_CNN','QAT_in'], load=False)
    test_nn(qat_net_2, validation_loader, device='cpu')

    # Finish ############################################################################################################################
    time_end = time.perf_counter()
    print('Done! (in', time_end - time_start,'seconds)')
