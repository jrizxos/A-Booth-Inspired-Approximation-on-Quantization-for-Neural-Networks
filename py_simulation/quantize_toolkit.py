import os

import numpy as np

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize

from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.datasets import CIFAR10

#### Constants ##########################################################################################################################

CWD = os.path.dirname(os.path.realpath(__file__))
TRAIN_RES = os.path.join(CWD, 'results')
TRACE_FILE = 'trace.txt'

BOOTH_QUANT_VALUE_DICT = {-128: -127, -127: -127, -126: -126, -125: -126, -124: -124, -123: -124, -122: -124, -121: -120, -120: -120, -119: -120, -118: -120, -117: -120, -116: -120, -115: -112, -114: -112, -113: -112, -112: -112, -111: -112, -110: -112, -109: -112, -108: -112, -107: -112, -106: -112, -105: -112, -104: -112, -103: -96, -102: -96, -101: -96, -100: -96, -99: -96, -98: -96, -97: -96, -96: -96, -95: -96, -94: -96, -93: -96, -92: -96, -91: -96, -90: -96, -89: -96, -88: -96, -87: -96, -86: -96, -85: -96, -84: -96, -83: -96, -82: -96, -81: -96, -80: -96, -79: -64, -78: -64, -77: -64, -76: -64, -75: -64, -74: -64, -73: -64, -72: -64, -71: -64, -70: -64, -69: -64, -68: -64, -67: -64, -66: -64, -65: -64, -64: -64, -63: -63, -62: -62, -61: -62, -60: -60, -59: -60, -58: -60, -57: -56, -56: -56, -55: -56, -54: -56, -53: -56, -52: -56, -51: -48, -50: -48, -49: -48, -48: -48, -47: -48, -46: -48, -45: -48, -44: -48, -43: -48, -42: -48, -41: -48, -40: -48, -39: -32, -38: -32, -37: -32, -36: -32, -35: 
        -32, -34: -32, -33: -32, -32: -32, -31: -31, -30: -30, -29: -30, -28: -28, -27: -28, -26: -28, -25: -24, -24: -24, -23: -24, -22: -24, -21: -24, -20: -24, -19: -16, -18: -16, -17: -16, -16: -16, -15: -15, -14: -14, -13: -14, -12: -12, -11: -12, -10: -12, -9: -8, -8: -8, -7: -7, -6: -6, -5: -6, -4: -4, -3: -3, -2: -2, -1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 6, 7: 7, 8: 8, 9: 8, 10: 8, 11: 12, 12: 12, 13: 12, 14: 14, 15: 15, 16: 16, 17: 16, 18: 16, 19: 16, 20: 16, 21: 24, 22: 24, 23: 24, 24: 24, 25: 24, 26: 24, 27: 28, 28: 28, 29: 28, 30: 30, 31: 31, 32: 32, 33: 32, 34: 32, 35: 32, 
        36: 32, 37: 32, 38: 32, 39: 32, 40: 32, 41: 48, 42: 48, 43: 48, 44: 48, 45: 48, 46: 48, 47: 48, 48: 48, 49: 48, 50: 48, 51: 48, 52: 48, 53: 56, 54: 56, 55: 56, 56: 56, 57: 56, 58: 56, 59: 60, 60: 60, 61: 60, 62: 62, 63: 63, 64: 64, 65: 64, 66: 64, 67: 64, 68: 64, 69: 64, 70: 64, 71: 64, 72: 64, 73: 64, 74: 64, 75: 64, 76: 64, 77: 64, 78: 64, 79: 64, 80: 64, 81: 96, 82: 96, 83: 96, 84: 96, 85: 
        96, 86: 96, 87: 96, 88: 96, 89: 96, 90: 96, 91: 96, 92: 96, 93: 96, 94: 96, 95: 96, 96: 96, 97: 96, 98: 96, 99: 96, 100: 96, 101: 96, 102: 96, 103: 96, 104: 96, 105: 112, 106: 112, 107: 112, 108: 112, 109: 112, 110: 112, 111: 112, 112: 112, 113: 112, 114: 112, 115: 112, 116: 112, 117: 120, 118: 120, 119: 120, 120: 120, 121: 120, 122: 120, 123: 124, 124: 124, 125: 124, 126: 126, 127: 127}

BOOTH_QUANT_VALUE_NP = np.array([-127, -127, -126, -126, -124, -124, -124, -120, -120, -120, -120, -120,
        -120, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,
        -112,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,
         -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,
         -96,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,
         -64,  -64,  -64,  -64,  -64,  -63,  -62,  -62,  -60,  -60,  -60,  -56,
         -56,  -56,  -56,  -56,  -56,  -48,  -48,  -48,  -48,  -48,  -48,  -48,
         -48,  -48,  -48,  -48,  -48,  -32,  -32,  -32,  -32,  -32,  -32,  -32,
         -32,  -31,  -30,  -30,  -28,  -28,  -28,  -24,  -24,  -24,  -24,  -24,
         -24,  -16,  -16,  -16,  -16,  -15,  -14,  -14,  -12,  -12,  -12,   -8,
          -8,   -7,   -6,   -6,   -4,   -3,   -2,   -1,    0,    1,    2,    3,
           4,    4,    6,    7,    8,    8,    8,   12,   12,   12,   14,   15,
          16,   16,   16,   16,   16,   24,   24,   24,   24,   24,   24,   28,
          28,   28,   30,   31,   32,   32,   32,   32,   32,   32,   32,   32,
          32,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,
          48,   56,   56,   56,   56,   56,   56,   60,   60,   60,   62,   63,
          64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,
          64,   64,   64,   64,   64,   96,   96,   96,   96,   96,   96,   96,
          96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,
          96,   96,   96,   96,   96,  112,  112,  112,  112,  112,  112,  112,
         112,  112,  112,  112,  112,  120,  120,  120,  120,  120,  120,  124,
         124,  124,  126,  127], dtype='int8')

BOOTH_QUANT_VALUE_TORCH = torch.tensor(BOOTH_QUANT_VALUE_NP, dtype=torch.int8)

BOOTH_QUANT_TRANSLATE = {-128: 7, -127: 7, -126: 15, -125: 15, -124: 23, -123: 23, -122: 23, -121: 31, -120: 31, -119: 31, -118: 31, -117: 31, -116: 31, -115: 39, -114: 39, -113: 39, -112: 39, -111: 39, -110: 39, -109: 39, -108: 39, -107: 39, -106: 39, -105: 39, -104: 39, -103: 47, -102: 47, -101: 47, -100: 47, -99: 47, -98: 47, -97: 47, -96: 47, -95: 47, -94: 47, -93: 47, -92: 47, -91: 47, -90: 47, -89: 47, -88: 47, -87: 47, -86: 47, -85: 47, -84: 47, -83: 47, -82: 47, -81: 47, -80: 47, -79: 55, -78: 55, -77: 55, -76: 55, -75: 55, -74: 55, -73: 55, -72: 55, -71: 55, -70: 55, -69: 55, -68: 55, -67: 55, -66: 55, -65: 55, -64: 55, -63: 6, -62: 14, -61: 14, -60: 22, -59: 22, -58: 22, -57: 30, -56: 30, -55: 30, -54: 30, -53: 30, -52: 30, -51: 38, -50: 38, -49: 38, -48: 38, -47: 38, -46: 38, -45: 38, -44: 38, 
-43: 38, -42: 38, -41: 38, -40: 38, -39: 46, -38: 46, -37: 46, -36: 46, -35: 46, -34: 46, -33: 46, -32: 46, -31: 5, -30: 13, -29: 13, -28: 21, -27: 21, -26: 21, -25: 29, -24: 29, -23: 29, -22: 29, -21: 29, -20: 29, -19: 37, -18: 37, -17: 37, -16: 37, -15: 4, -14: 12, -13: 12, -12: 20, -11: 20, -10: 20, -9: 28, -8: 28, -7: 3, -6: 11, -5: 11, -4: 19, -3: 2, -2: 10, -1: 1, 0: 0, 1: 8, 2: 17, 3: 16, 4: 26, 5: 26, 6: 25, 7: 24, 8: 35, 9: 35, 10: 35, 11: 34, 12: 34, 13: 34, 14: 33, 15: 32, 16: 44, 17: 44, 18: 44, 19: 44, 20: 44, 21: 43, 22: 43, 23: 43, 24: 43, 25: 43, 26: 43, 27: 42, 28: 42, 29: 42, 30: 41, 31: 40, 32: 53, 33: 53, 34: 53, 35: 53, 36: 53, 37: 53, 38: 53, 39: 53, 40: 53, 41: 52, 42: 52, 43: 52, 44: 52, 45: 52, 46: 52, 47: 52, 48: 52, 49: 52, 50: 52, 51: 52, 52: 52, 53: 51, 54: 51, 55: 51, 56: 51, 57: 51, 58: 51, 59: 50, 60: 50, 61: 50, 62: 49, 63: 48, 64: 62, 65: 62, 66: 62, 67: 62, 68: 62, 69: 62, 70: 62, 71: 62, 72: 62, 73: 62, 74: 62, 75: 62, 76: 62, 77: 62, 78: 62, 79: 62, 80: 62, 81: 61, 82: 61, 83: 61, 84: 61, 85: 61, 86: 61, 87: 61, 88: 61, 89: 61, 90: 61, 91: 61, 92: 61, 93: 61, 94: 61, 95: 61, 96: 61, 97: 61, 98: 61, 99: 61, 100: 61, 101: 61, 102: 61, 
103: 61, 104: 61, 105: 60, 106: 60, 107: 60, 108: 60, 109: 60, 110: 60, 111: 60, 112: 60, 113: 60, 114: 60, 115: 60, 116: 60, 117: 59, 118: 59, 119: 59, 120: 59, 121: 59, 122: 59, 123: 58, 124: 58, 
125: 58, 126: 57, 127: 56}

BOOTH_QUANT_TRANS_MAP = {-127: 7, -126: 15, -124: 23, -120: 31, -112: 39, -96: 47, -64: 55, -63: 6, -62: 14, -60: 22, -56: 30, -48: 38, -32: 46, -31: 5, -30: 13, -28: 21, -24: 29, -16: 37, -15: 4, -14: 12, -12: 20, -8: 28, -7: 3, -6: 11, -4: 19, -3: 2, -2: 10, -1: 1, 0: 0, 1: 8, 2: 17, 3: 16, 4: 26, 6: 25, 7: 24, 8: 35, 12: 34, 14: 33, 15: 32, 16: 44, 24: 43, 28: 42, 30: 41, 31: 40, 32: 53, 48: 52, 56: 51, 60: 50, 62: 49, 63: 48, 64: 62, 96: 61, 112: 60, 120: 59, 124: 58, 126: 57, 127: 56}

##### Torch datasets ####################################################################################################################

def get_ImageNet(transform:transforms.Compose=None) -> tuple[DataLoader,DataLoader]:
    if(transform==None):
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
    
    root = os.path.join(CWD, 'training_data', 'ILSVRC2012')
    train_dataset = ImageNet(root=root, split='train', transform=transform)
    validation_dataset = ImageNet(root=root, split='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=12) 
    return train_loader, validation_loader

def get_CIFAR(transform:transforms.Compose=None) -> tuple[DataLoader,DataLoader]:
    if(transform==None):
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], 
                        std=[0.5, 0.5, 0.5])])
    
    root = os.path.join(CWD, 'training_data', 'CIFAR10')
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    validation_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=8)
    return train_loader, validation_loader


##### Torch functions ###################################################################################################################

class QuantizableModel(torch.nn.Module):
    def __init__(self, model_fp32):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def inspect_quantized_weights(model: nn.Module, prints: bool = True) -> int:
    invalid_total = 0
    for m_name, param in model.state_dict().items():
        if isinstance(param, torch.Tensor) and param.is_quantized:
            int_repr = param.int_repr()
            dequant = param.dequantize()
            mask = torch.isin(int_repr, BOOTH_QUANT_VALUE_TORCH)
            invalid = int_repr[~mask].unique()
            invalid_total += len(invalid)
            if(prints):
                print(f'Layer: {m_name} | {param.shape} : {param.dtype}')
                print(f'  Scale: {param.q_scale()}, Zero Point: {param.q_zero_point()}')
                print(f'  Stored INT min/max: {int_repr.min().item()} / {int_repr.max().item()} ({dequant.min().item()} / {dequant.max().item()})')
                print('Invalid values: '+str(invalid) if(len(invalid)>0) else 'No invalid values!')
                print()
    for m_name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight, bias = module._packed_params._packed_params.unpack()
            int_repr = weight.int_repr()
            dequant = weight.dequantize()
            mask = torch.isin(int_repr, BOOTH_QUANT_VALUE_TORCH)
            invalid = int_repr[~mask].unique()
            invalid_total += len(invalid)
            if(prints):
                print(f'Linear Layer: {m_name} | {weight.shape} : {weight.dtype}:')
                print(f'  Scale: {module.scale}, Zero Point: {module.zero_point}')
                print(f'  Stored INT min/max: {int_repr.min().item()} / {int_repr.max().item()} ({dequant.min().item()} / {dequant.max().item()})')
                print('Invalid values: '+str(invalid) if(len(invalid)>0) else 'No invalid values!')
                print()
    return invalid_total

def apply_custom_quantization(network: nn.Module, name: list, load: bool=False) -> None:
    save_dir = os.path.join(TRAIN_RES, name[0])

    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, '_'.join(name) + '.pth')

    if load and os.path.exists(model_path):
        print('Loading saved model...')
        network.load_state_dict(torch.load(model_path))
        print('Model loaded! Skipping PTQ custom quantization.')
    
    else:
        with torch.no_grad():
            for module_name, module in network.named_modules():
                if isinstance(module, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
                    print(f'Applying custom quantization to: {module_name}', end='')

                    weights = module.weight()
                    qscheme = weights.qscheme()
                    bias = module.bias()

                    if not weights.is_quantized or weights.dtype != torch.qint8:
                        print(f' Skipped! (not qint8 weights, got dtype: {weights.dtype})')
                        continue

                    int_vals = weights.int_repr()

                    if qscheme == torch.per_tensor_affine:
                        scale = weights.q_scale()
                        zero_point = weights.q_zero_point()

                        indexed = (int_vals.to(torch.int16) + 128)
                        processed = BOOTH_QUANT_VALUE_TORCH[indexed.to(torch.int)]

                        new_qtensor = torch._make_per_tensor_quantized_tensor(
                            processed.reshape_as(int_vals),
                            scale=scale,
                            zero_point=zero_point)

                    elif qscheme == torch.per_channel_affine:
                        scales = weights.q_per_channel_scales()
                        zero_points = weights.q_per_channel_zero_points()
                        axis = weights.q_per_channel_axis()

                        indexed = (int_vals.to(torch.int16) + 128)
                        processed = BOOTH_QUANT_VALUE_TORCH[indexed.to(torch.int)]

                        new_qtensor = torch._make_per_channel_quantized_tensor(
                            processed.reshape_as(int_vals),
                            scale=scales,
                            zero_point=zero_points,
                            axis=axis)

                    else:
                        print(f' Skipped! (unsupported qscheme: {qscheme})')
                        continue

                    module.set_weight_bias(new_qtensor, bias)
                    print(' Done!')

        torch.save(network.state_dict(), model_path)
    return

def test(network: nn.Module, test_loader: DataLoader, max_batches: int=0, device: str='cpu') -> None:
    network = network.to(device)
    network.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    batch_idx = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            if max_batches>0 and batch_idx>max_batches:
                break
            
            outputs = network(data)
            if hasattr(outputs, "logits"):  # Inception v3 case
                outputs = outputs.logits
            vals, top1_preds = outputs.topk(1, dim=1)
            top1_correct += (top1_preds.squeeze(1) == labels).sum().item()

            vals, top5_preds = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in top5_preds[i] for i in range(labels.size(0))])

            batch_idx += 1

            total += labels.size(0)

    print(f'Test set Top-1 Acc: {(top1_correct / total)*100:.0f}%, Top-5 Acc: {(top5_correct / total)*100:.0f}%')
    return

def train(network: nn.Module, 
          train_loader: DataLoader, 
          optimizer: Optimizer, 
          criterion: _Loss, 
          epochs: int, 
          name: str, 
          load: bool=False,
          device: str='cpu') -> None:
    save_dir = os.path.join(TRAIN_RES, name[0])

    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, '_'.join(name) + '.pth')
    optimizer_path = os.path.join(save_dir, '_'.join(name) +'_optimizer.pth')

    if load and os.path.exists(model_path) and os.path.exists(optimizer_path):
        print('Loading saved model and optimizer...')
        network.load_state_dict(torch.load(model_path))
        print('Model loaded! Skipping training.')
    
    else:
        network.to(device)
        print(('No saved model found. ' if load else '') + 'Training model from scratch...')
        for epoch in range(1, epochs + 1):
            network.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = network(data)
                if hasattr(outputs, "logits"):  # Inception v3 case
                    outputs, aux_output = outputs.logits, outputs.aux_logits
                    loss = criterion(outputs, labels) + 0.4 * criterion(aux_output, labels)
                else:                           # Standard models
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * data.size(0)
                vals, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            print(f'Epoch {epoch}/{epochs}, Loss: {running_loss/total:.4f}, Acc: {(correct/total)*100:.0f}%')

        torch.save(network.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
    return

# TODO fix/check device usage in training
def PTQ_quantize(network: nn.Module, 
                 test_loader: DataLoader, 
                 observer_limit: int, 
                 name: str, 
                 qconfig: QConfig, 
                 load: bool=False,
                 device: str='cpu') -> None:
    save_dir = os.path.join(TRAIN_RES, name[0])

    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, '_'.join(name) + '.pth')

    network.to(device)
    network.eval()
    network.qconfig = qconfig

    torch.quantization.prepare(network, inplace=True)

    if load and os.path.exists(model_path):
        print('Loading saved model...')
        torch.quantization.convert(network, inplace=True)
        network.load_state_dict(torch.load(model_path))
        print('Model loaded! Skipping PTQ quantization.')
        return
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            network.to(device)
            if(i >= observer_limit): 
                break
            network(data)

    torch.quantization.convert(network, inplace=True)
    torch.save(network.state_dict(), model_path)
    return

def QAT_quantize(network: nn.Module, 
                 train_loader: DataLoader, 
                 optimizer: Optimizer, 
                 criterion: _Loss, 
                 epochs: int, 
                 name:str, 
                 qconfig_dict: dict, 
                 load: bool=False,
                 device: str='cpu') -> None:
    save_dir = os.path.join(TRAIN_RES, name[0])

    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, '_'.join(name) + '.pth')

    qat_qconfig = QConfig(activation=qconfig_dict['activation'],
                          weight=qconfig_dict['weight'])

    if(qconfig_dict['custom']):
        qat_qconfig_custom = QConfig(activation=qconfig_dict['activation'],
                                     weight=qconfig_dict['custom'])
        for m_name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.qconfig = qat_qconfig_custom
            else:
                module.qconfig = qat_qconfig
    else:
        network.qconfig = qat_qconfig
    
    if load and os.path.exists(model_path):
        print('Loading saved model...')
        network.to('cpu')
        network.train()
        torch.quantization.prepare_qat(network, inplace=True)
        torch.quantization.convert(network, inplace=True)
        network.load_state_dict(torch.load(model_path))
        print('Model loaded! Skipping QAT quantization.')

    else:
        network.to(device)
        network.train()
        torch.quantization.prepare_qat(network, inplace=True)
        train(network, train_loader, optimizer, criterion, epochs, name=[name[0], 'QAT_main'], load=False, device=device)
        network.to('cpu')
        network.eval()
        torch.quantization.convert(network, inplace=True)
        torch.save(network.state_dict(), model_path)
    return

class CustomFakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zp, quant_min, quant_max, lookup_table):
        inv_scale = 1.0 / scale
        transformed = (x * inv_scale).round() + zp
        clamped = torch.clamp(transformed, quant_min, quant_max)

        lookup_table = lookup_table.to(x.device)

        indexed = (clamped.to(torch.int16) + 128).to(torch.int64)
        looked_up = lookup_table[indexed]

        dequant = (looked_up.to(x.dtype) - zp) * scale
        return dequant

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None, None

class CustomWeightFakeQuantize(FakeQuantize):
    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            quant_min = self.activation_post_process.quant_min
            quant_max = self.activation_post_process.quant_max

            if self.is_per_channel:
                # Per-channel broadcasting
                scale = self.scale.reshape([-1 if i == self.ch_axis else 1 for i in range(X.dim())])
                zp = self.zero_point.reshape([-1 if i == self.ch_axis else 1 for i in range(X.dim())])
            else:
                scale = self.scale
                zp = self.zero_point

            # Call custom quantization function with STE
            X = CustomFakeQuantFunction.apply(X, scale, zp, quant_min, quant_max, BOOTH_QUANT_VALUE_TORCH)

        return X