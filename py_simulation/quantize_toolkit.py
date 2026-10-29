import os

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize

import torchao

from transformers import ViTImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.whisper.processing_whisper import WhisperProcessor
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
from transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer import ASTFeatureExtractor

import bitsandbytes

#### Constants ##########################################################################################################################

CWD = os.path.dirname(os.path.realpath(__file__))
TRAIN_RES = os.path.join(CWD, 'results')
TRACE_FILE = 'trace.txt'

# these are the allowed values
QUANT_LUT_ALLOWED = [-127, -126, -124, -120, -112,  -96,  -64,  -63,  -62,  -60,  -56,  -48,  -32,  -31,
                      -30,  -28,  -24,  -16,  -15,  -14,  -12,   -8,   -7,   -6,   -4,   -3,   -2,   -1,
                        0,    1,    2,    3,    4,    6,    7,    8,   12,   14,   15,   16,   24,   28,
                       30,   31,   32,   48,   56,   60,   62,   63,   64,   96,  112,  120,  124,  126,
                      127]

# moves centered values to smaller closest
QUANT_LUT_SMALLER = [-127, -127, -126, -126, -124, -124, -124, -120, -120, -120, -120, -120, -120, -112,
                     -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,  -96,  -96,  -96,
                      -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,
                      -96,  -96,  -96,  -96,  -96,  -96,  -96,  -64,  -64,  -64,  -64,  -64,  -64,  -64,
                      -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -63,  -62,  -62,  -60,  -60,
                      -60,  -56,  -56,  -56,  -56,  -56,  -56,  -48,  -48,  -48,  -48,  -48,  -48,  -48,
                      -48,  -48,  -48,  -48,  -48,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -31,
                      -30,  -30,  -28,  -28,  -28,  -24,  -24,  -24,  -24,  -24,  -24,  -16,  -16,  -16,
                      -16,  -15,  -14,  -14,  -12,  -12,  -12,   -8,   -8,   -7,   -6,   -6,   -4,   -3,
                       -2,   -1,    0,    1,    2,    3,    4,    4,    6,    7,    8,    8,    8,   12,
                       12,   12,   14,   15,   16,   16,   16,   16,   16,   24,   24,   24,   24,   24,
                       24,   28,   28,   28,   30,   31,   32,   32,   32,   32,   32,   32,   32,   32,
                       32,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   56,
                       56,   56,   56,   56,   56,   60,   60,   60,   62,   63,   64,   64,   64,   64,
                       64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   96,
                       96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,
                       96,   96,   96,   96,   96,   96,   96,   96,   96,  112,  112,  112,  112,  112,
                      112,  112,  112,  112,  112,  112,  112,  120,  120,  120,  120,  120,  120,  124,
                      124,  124,  126,  127]

# moves centered values to bigger closest
QUANT_LUT_BIGGER = [ -127, -127, -126, -124, -124, -124, -120, -120, -120, -120, -120, -120, -112, -112,
                     -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,  -96,  -96,  -96,  -96,
                      -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,
                      -96,  -96,  -96,  -96,  -96,  -96,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,
                      -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -63,  -62,  -60,  -60,  -60,
                      -56,  -56,  -56,  -56,  -56,  -56,  -48,  -48,  -48,  -48,  -48,  -48,  -48,  -48,
                      -48,  -48,  -48,  -48,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -31,
                      -30,  -28,  -28,  -28,  -24,  -24,  -24,  -24,  -24,  -24,  -16,  -16,  -16,  -16,
                      -16,  -15,  -14,  -12,  -12,  -12,   -8,   -8,   -8,   -7,   -6,   -4,   -4,   -3,
                       -2,   -1,    0,    1,    2,    3,    4,    6,    6,    7,    8,    8,   12,   12,
                       12,   14,   14,   15,   16,   16,   16,   16,   24,   24,   24,   24,   24,   24,
                       28,   28,   28,   30,   30,   31,   32,   32,   32,   32,   32,   32,   32,   32,
                       48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   56,   56,
                       56,   56,   56,   56,   60,   60,   60,   62,   62,   63,   64,   64,   64,   64,
                       64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   96,   96,
                       96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,
                       96,   96,   96,   96,   96,   96,   96,   96,  112,  112,  112,  112,  112,  112,
                      112,  112,  112,  112,  112,  112,  120,  120,  120,  120,  120,  120,  124,  124,
                      124,  126,  126,  127]

# moves centered values to the closest towards zero
QUANT_LUT_TO_ZERO = [-127, -127, -126, -124, -124, -124, -120, -120, -120, -120, -120, -120, -112, -112,
                     -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,  -96,  -96,  -96,  -96,
                      -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,
                      -96,  -96,  -96,  -96,  -96,  -96,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,
                      -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -63,  -62,  -60,  -60,  -60,
                      -56,  -56,  -56,  -56,  -56,  -56,  -48,  -48,  -48,  -48,  -48,  -48,  -48,  -48,
                      -48,  -48,  -48,  -48,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -31,
                      -30,  -28,  -28,  -28,  -24,  -24,  -24,  -24,  -24,  -24,  -16,  -16,  -16,  -16,
                      -16,  -15,  -14,  -12,  -12,  -12,   -8,   -8,   -8,   -7,   -6,   -4,   -4,   -3,
                       -2,   -1,    0,    1,    2,    3,    4,    4,    6,    7,    8,    8,    8,   12,
                       12,   12,   14,   15,   16,   16,   16,   16,   16,   24,   24,   24,   24,   24,
                       24,   28,   28,   28,   30,   31,   32,   32,   32,   32,   32,   32,   32,   32,
                       32,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   56,
                       56,   56,   56,   56,   56,   60,   60,   60,   62,   63,   64,   64,   64,   64,
                       64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   96,
                       96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,
                       96,   96,   96,   96,   96,   96,   96,   96,   96,  112,  112,  112,  112,  112,
                      112,  112,  112,  112,  112,  112,  112,  120,  120,  120,  120,  120,  120,  124,
                      124,  124,  126,  127]

# moves centered values to the closest away from zero              
QUANT_LUT_AWAY_ZERO = [-127, -127, -126, -126, -124, -124, -124, -120, -120, -120, -120, -120, -120, -112,
                       -112, -112, -112, -112, -112, -112, -112, -112, -112, -112, -112,  -96,  -96,  -96,
                        -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,  -96,
                        -96,  -96,  -96,  -96,  -96,  -96,  -96,  -64,  -64,  -64,  -64,  -64,  -64,  -64,
                        -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -64,  -63,  -62,  -62,  -60,  -60,
                        -60,  -56,  -56,  -56,  -56,  -56,  -56,  -48,  -48,  -48,  -48,  -48,  -48,  -48,
                        -48,  -48,  -48,  -48,  -48,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -31,
                        -30,  -30,  -28,  -28,  -28,  -24,  -24,  -24,  -24,  -24,  -24,  -16,  -16,  -16,
                        -16,  -15,  -14,  -14,  -12,  -12,  -12,   -8,   -8,   -7,   -6,   -6,   -4,   -3,
                         -2,   -1,    0,    1,    2,    3,    4,    6,    6,    7,    8,    8,   12,   12,
                         12,   14,   14,   15,   16,   16,   16,   16,   24,   24,   24,   24,   24,   24,
                         28,   28,   28,   30,   30,   31,   32,   32,   32,   32,   32,   32,   32,   32,
                         48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   48,   56,   56,
                         56,   56,   56,   56,   60,   60,   60,   62,   62,   63,   64,   64,   64,   64,
                         64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   96,   96,
                         96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,   96,
                         96,   96,   96,   96,   96,   96,   96,   96,  112,  112,  112,  112,  112,  112,
                        112,  112,  112,  112,  112,  112,  120,  120,  120,  120,  120,  120,  124,  124,
                        124,  126,  126,  127]

BOOTH_QUANT_VALUE_TORCH = torch.tensor(QUANT_LUT_AWAY_ZERO, dtype=torch.int8)

# maps an ASP2 (6-bits) to its corresponding int8
ASP2_MAPPING = {(0, 0):   0,  (0, 1):  -1,  (0, 2):  -3,  (0, 3):  -7,  (0, 4): -15,  (0, 5): -31,  (0, 6): -63,  (0, 7): -127,
                (1, 0):   1,                (1, 2):  -2,  (1, 3):  -6,  (1, 4): -14,  (1, 5): -30,  (1, 6): -62,  (1, 7): -126,
                (2, 0):   3,  (2, 1):   2,                (2, 3):  -4,  (2, 4): -12,  (2, 5): -28,  (2, 6): -60,  (2, 7): -124,
                (3, 0):   7,  (3, 1):   6,  (3, 2):   4,                (3, 4):  -8,  (3, 5): -24,  (3, 6): -56,  (3, 7): -120,
                (4, 0):  15,  (4, 1):  14,  (4, 2):  12,  (4, 3):   8,                (4, 5): -16,  (4, 6): -48,  (4, 7): -112,
                (5, 0):  31,  (5, 1):  30,  (5, 2):  28,  (5, 3):  24,  (5, 4):  16,                (5, 6): -32,  (5, 7):  -96,
                (6, 0):  63,  (6, 1):  62,  (6, 2):  60,  (6, 3):  56,  (6, 4):  48,  (6, 5):  32,                (6, 7):  -64,
                (7, 0): 127,  (7, 1): 126,  (7, 2): 124,  (7, 3): 120,  (7, 4): 112,  (7, 5):  96,  (7, 6):  64                }
# here -128 means an invalid key, the ASP2 numbers corresponding to those indexes do not exist
ASP2_MAPPING_LUT = torch.tensor([    0,   -1,   -3,   -7,  -15,  -31,  -63, -127,    
                                     1, -128,   -2,   -6,  -14,  -30,  -62, -126, 
                                     3,    2, -128,   -4,  -12,  -28,  -60, -124,    
                                     7,    6,    4, -128,   -8,  -24,  -56, -120, 
                                    15,   14,   12,    8, -128,  -16,  -48, -112,   
                                    31,   30,   28,   24,   16, -128,  -32,  -96, 
                                    63,   62,   60,   56,   48,   32, -128,  -64, 
                                   127,  126,  124,  120,  112,   96,   64, -128 ], dtype=torch.int8)

# maps an int8 to its corresponding ASP2 (6-bits)
ASP2_INV_MAPPING = {0  : (0, 0),  -1 : (0, 1),  -3 : (0, 2),  -7 : (0, 3),  -15: (0, 4),  -31: (0, 5),  -63: (0, 6),  -127: (0, 7),
                    1  : (1, 0),                -2 : (1, 2),  -6 : (1, 3),  -14: (1, 4),  -30: (1, 5),  -62: (1, 6),  -126: (1, 7),  
                    3  : (2, 0),  2  : (2, 1),                -4 : (2, 3),  -12: (2, 4),  -28: (2, 5),  -60: (2, 6),  -124: (2, 7),  
                    7  : (3, 0),  6  : (3, 1),  4  : (3, 2),                -8 : (3, 4),  -24: (3, 5),  -56: (3, 6),  -120: (3, 7),  
                    15 : (4, 0),  14 : (4, 1),  12 : (4, 2),  8  : (4, 3),                -16: (4, 5),  -48: (4, 6),  -112: (4, 7),  
                    31 : (5, 0),  30 : (5, 1),  28 : (5, 2),  24 : (5, 3),  16 : (5, 4),                -32: (5, 6),  -96 : (5, 7),  
                    63 : (6, 0),  62 : (6, 1),  60 : (6, 2),  56 : (6, 3),  48 : (6, 4),  32 : (6, 5),                -64 : (6, 7),  
                    127: (7, 0),  126: (7, 1),  124: (7, 2),  120: (7, 3),  112: (7, 4),  96 : (7, 5),  64  : (7, 6)               } 
# here 63 means an invalid key, the int8 numbers corresponding to those indexes should be replaced by the closest representable int8
ASP2_INV_MAPPING_LUT = torch.tensor([63,  7, 15, 63, 23, 63, 63, 63, 31, 63, 63, 63, 63, 63, 63, 63,
                                     39, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
                                     47, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
                                     63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
                                     55,  6, 14, 63, 22, 63, 63, 63, 30, 63, 63, 63, 63, 63, 63, 63,
                                     38, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
                                     46,  5, 13, 63, 21, 63, 63, 63, 29, 63, 63, 63, 63, 63, 63, 63,
                                     37,  4, 12, 63, 20, 63, 63, 63, 28,  3, 11, 63, 19,  2, 10,  1, 
                                      0,  8, 17, 16, 26, 63, 25, 24, 35, 63, 63, 63, 34, 63, 33, 32, 
                                     44, 63, 63, 63, 63, 63, 63, 63, 43, 63, 63, 63, 42, 63, 41, 40,
                                     53, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 
                                     52, 63, 63, 63, 63, 63, 63, 63, 51, 63, 63, 63, 50, 63, 49, 48, 
                                     62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 
                                     63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 
                                     61, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 
                                     60, 63, 63, 63, 63, 63, 63, 63, 59, 63, 63, 63, 58, 63, 57, 56], dtype=torch.int8)

##### General functions #################################################################################################################

def produce_quant_lut_allowed():
    mapping = {}
    inv_mapping = {}
    for i in range(0,8):
        for j in range(0,8):
            val = 2**i - 2**j
            if(val!=0 or i==j==0):
                mapping[(i,j)] = val
                inv_mapping[val] = (i,j)
    return mapping, inv_mapping

def closest_approx(x: int):
    if x in QUANT_LUT_ALLOWED:
        return x
    r_up = x
    r_dw = x

    while 1:
        if r_dw > -128:
            r_dw -= 1
            if r_dw in QUANT_LUT_ALLOWED:
                return r_dw
        if r_up < 128:
            r_up += 1
            if r_up in QUANT_LUT_ALLOWED:
                return r_up

def produce_quant_lut_closest():
    lut = []
    for x in range(-128,128):
        apx = closest_approx(x)
        lut.append(apx)
    return lut

def booth_transitions(x: int, width: int):

    mask = (1 << width) - 1

    # Force fixed-width 2's complement
    x = x & mask

    # Append virtual 0 at position -1
    x_ext = x << 1

    # Transitions between adjacent bits
    transitions = (x_ext ^ (x_ext >> 1))

    # Limit to width bits
    transitions &= mask

    # Extract bit positions MSB to LSB
    positions = []
    while transitions:
      i = transitions.bit_length() - 1
      positions.append(i)
      transitions ^= 1<<i

    return positions

def booth_approx(x: int, width: int, q: int):
    positions = booth_transitions(x, width)
    msb = ((x>> (width-1)) & 1) == 1
    sign = -1 if msb == True else 1
    approx = 0
    while positions and (q > 0):
        approx += sign * (1<<positions.pop(0))
        sign = -sign # invert sign
        q -= 1
    return approx

def booth_like_apporx(x: int, width: int, q: int):
    positions = booth_transitions(x, width)
    msb = ((x>> (width-1)) & 1) == 1
    sign = -1 if msb == True else 1
    approx = 0
    while positions and (q > 0):
        cur_pos  = positions.pop(0)
        if q == 1: # At the last power of two in the approximation
            # I use -2, if there are no items in positions, as the cur_pos could be 0 and I want to test if they are consecutive numbers!
            next_pos = next(iter(positions), -2)
            if cur_pos == next_pos+1: # consecutive transitions, so 010 or 101 in the original number
              cur_pos -= 1  # or next_pos it's the same...
        approx += sign * (1<<cur_pos)
        sign = -sign # invert sign
        q -= 1
    return approx

def produce_booth_like_apporx_lut():
    lut = []
    for x in range(-128,128):
        apx = booth_like_apporx(x,8,2)
        apx = apx if not apx==-128 else -127
        lut.append(apx)
    return lut

def produce_to_and_away_zero():
    to_zero = []
    away_zero = []
    for i in range(-128,128):
        if i<0:
            to_zero.append(QUANT_LUT_BIGGER[i+128])
            away_zero.append(QUANT_LUT_SMALLER[i+128])
        else:
            to_zero.append(QUANT_LUT_SMALLER[i+128])
            away_zero.append(QUANT_LUT_BIGGER[i+128])
    return to_zero, away_zero

##### Torch functions ###################################################################################################################

class QuantizableModel(nn.Module):
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

def count_params(model):
    total_params = 0
    quantized_params = 0
    quantized_axc = 0
    fp32_params = 0

    for p in model.parameters():
        n = p.numel() 
        total_params += n
        fp32_params += n

    for m in model.modules():

        if isinstance(m, torch.nn.quantized.Conv2d):
            w = m.weight()
            qn = w.numel() if w is not None else 0
            quantized_params += qn
            quantized_axc += qn
            total_params += qn

        elif isinstance(m, torch.nn.quantized.Linear):
            w = m.weight()
            qn = w.numel() if w is not None else 0
            quantized_params += qn
            quantized_axc += qn
            total_params += qn

    print(f'Total params: {total_params}')
    print(f'Quantized params: {quantized_params} ( {quantized_params * 100 / total_params:.2f}% )')
    print(f'AXC Quantized params: {quantized_axc} ( {quantized_axc * 100 / total_params:.2f}% )')
    print(f'Remaining FP params (unquantized): {fp32_params}')

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
            count = 0
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
                    indexed = (int_vals.to(torch.int16) + 128)
                    processed = BOOTH_QUANT_VALUE_TORCH[indexed.to(torch.int)]

                    if qscheme == torch.per_tensor_affine:
                        new_qtensor = torch._make_per_tensor_quantized_tensor(
                            processed.reshape_as(int_vals),
                            scale = weights.q_scale(),
                            zero_point = weights.q_zero_point())

                    elif qscheme == torch.per_channel_affine:
                        new_qtensor = torch._make_per_channel_quantized_tensor(
                            processed.reshape_as(int_vals),
                            scale = weights.q_per_channel_scales(),
                            zero_point = weights.q_per_channel_zero_points(),
                            axis = weights.q_per_channel_axis())

                    else:
                        print(f' Skipped! (unsupported qscheme: {qscheme})')
                        continue

                    module.set_weight_bias(new_qtensor, bias)
                    count += int_vals.numel()
                    print(' Done!')

        print(f'Approximated {count} total parameters')
        torch.save(network.state_dict(), model_path)
    return

def apply_custom_quantization_loose(network: nn.Module, name: list, load: bool=False) -> None:
    ''' Like apply_custom_quantization, but skips the depthwise and the pointwise porjection conv2 layers'''
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
            count = 0
            for module_name, module in network.named_modules():
                if not isinstance(module, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
                    continue

                if isinstance(module, torch.nn.quantized.Conv2d):
                    if (
                        module.groups == module.in_channels
                        and module.in_channels == module.out_channels
                    ):
                        print(f'Skipping depthwise layer: {module_name}')
                        continue
                
                    if (
                        module.kernel_size == (1, 1)
                        and module.out_channels < module.in_channels
                    ):
                        print(f'Skipping projection layer: {module_name}')
                        continue
                
                print(f'Applying custom quantization to: {module_name}', end='')
                
                
                weights = module.weight()
                qscheme = weights.qscheme()
                bias = module.bias()
                
                
                if not weights.is_quantized or weights.dtype != torch.qint8:
                    print(f' Skipped! (not qint8 weights, got dtype: {weights.dtype})')
                    continue
                
                int_vals = weights.int_repr()
                indexed = (int_vals.to(torch.int16) + 128)
                processed = BOOTH_QUANT_VALUE_TORCH[indexed.to(torch.int)]

                if qscheme == torch.per_tensor_affine:
                    new_qtensor = torch._make_per_tensor_quantized_tensor(
                        processed.reshape_as(int_vals),
                        scale = weights.q_scale(),
                        zero_point = weights.q_zero_point())

                elif qscheme == torch.per_channel_affine:
                    new_qtensor = torch._make_per_channel_quantized_tensor(
                        processed.reshape_as(int_vals),
                        scale = weights.q_per_channel_scales(),
                        zero_point = weights.q_per_channel_zero_points(),
                        axis = weights.q_per_channel_axis())

                else:
                    print(f' Skipped! (unsupported qscheme: {qscheme})')
                    continue

                module.set_weight_bias(new_qtensor, bias)
                count += int_vals.numel()
                print(' Done!')


        torch.save(network.state_dict(), model_path)
        print(f'Approximated {count} parameters')
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
                if hasattr(outputs, 'logits'):  # Inception v3 case
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

            print(f'Epoch {epoch}/{epochs}, Loss: {running_loss/total:.4f}, Acc: {(correct/total)*100:.2f}%')

        torch.save(network.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
    return

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

    network.to('cpu')
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
            if(i >= observer_limit): 
                break

            data, labels = data.to('cpu'), labels.to('cpu')
            network.to('cpu')
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

    network.qconfig = qat_qconfig

    if(qconfig_dict['custom']):
        qat_qconfig_custom = QConfig(activation=qconfig_dict['activation'],
                                     weight=qconfig_dict['custom'])
        for module in network.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.qconfig = qat_qconfig_custom
    
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


##### torchao functions #################################################################################################################

def inspect_torchao(model: nn.Module, device: str, out_mode: int = 2):
    '''
    out_mode arg selects which stats will be printed:
    0 : check all layers and report only total invalid values
    1 : print stats from all layers
    2 : print stats from all layers and invalid values per layer
    '''
    total_invalid = 0
    allowed = torch.tensor(QUANT_LUT_ALLOWED, device=device)
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            w = module.weight
            if isinstance(w, torchao.quantization.Int8Tensor):
                int_data = w.qdata
            if isinstance(w, torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor):
                int_data = w.original_weight_tensor.tensor_impl.__dict__['int_data'] 
            else:
                continue 
            mask = torch.isin(int_data, allowed)
            invalid = int_data[~mask].unique()
            total_invalid += len(invalid)
            if out_mode > 0 :
                print(f'\nLayer: {name}')
                print('\tint_data dtype:', int_data.dtype)
                print('\tmin:', int_data.min().item())
                print('\tmax:', int_data.max().item())
                if out_mode > 1:
                    print('Invalid values: '+str(invalid) if(len(invalid)>0) else 'No invalid values!')
    print(f'Total invalid values: {total_invalid}')

def count_params_torchao(model: nn.Module):
    int8_elems = 0
    total_elems = 0

    # total parameters (original FP parameters)
    for p in model.parameters():
        total_elems += p.numel()

    # walk modules to find Int8Tensor-backed weights
    for _, module in model.named_modules():
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
            except Exception:
                continue

            if isinstance(attr, (torchao.quantization.Int8Tensor, torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor)):
                int8_elems += attr.numel()

    print(f'Total params: {total_elems}\nQuantized params: {int8_elems} ( {int8_elems*100 / total_elems:.2f}% )')

def apply_custom_quantization_torchao(model: nn.Module, device: str, prints: bool = False):
    count_layers = 0
    count_params = 0
    quant_lut = BOOTH_QUANT_VALUE_TORCH.to(device)

    for _, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        w = module.weight
        if isinstance(w, torchao.quantization.Int8Tensor):
            int_data = w.qdata
        elif isinstance(w,torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor):
            int_data = w.original_weight_tensor.tensor_impl.__dict__['int_data']
        else:
            continue

        idx = (int_data.to(torch.int16) + 128).to(torch.long)
        new_int_data = quant_lut[idx]

        int_data.copy_(new_int_data)
        count_layers += 1
        count_params += len(int_data)

    if prints:
        print(f"LUT applied to {count_layers} quantized Linear layers")


##### bitsandbytes functions ############################################################################################################

def preheat_bnb(model: nn.Module, tokenizer: any, seed: int):
    ''' BNB quantized models require 'preheating' as initially all interesting parameters are None. 
    Giving it some input will materialize those parameters to not-None '''
    model.eval()
    with torch.no_grad():
        if isinstance(tokenizer, ViTImageProcessor):
            try:
                dummy_images = torch.randn(
                    1,
                    3,
                    tokenizer.size['height'],
                    tokenizer.size['width'],
                    device=model.device
                )
                dummy = {'pixel_values': dummy_images}
                model(**dummy)
            except Exception as e:
                print(f'Unexpected behaviour in preheat_bnb. Info: most likely the provided model or tokenizer in not compatible witht the baked dummy input here.')
                raise e
        elif isinstance(tokenizer, PreTrainedTokenizerBase):
            try:
                dummy = tokenizer(
                    f'Materialize bnb weights with seed: {str(seed)}',
                    return_tensors='pt'
                ).to(model.device)
                model(**dummy) 
            except Exception as e:
                print(f'Unexpected behaviour in preheat_bnb. Info: most likely the provided model or tokenizer in not compatible witht the baked dummy input here.')
                raise e                                            
        elif isinstance(tokenizer, WhisperProcessor):
            try:
                dummy_features = torch.randn(
                    1, 128, 3000, dtype=model.dtype, device=model.device
                )
                dummy = {'input_features': dummy_features}
                model.generate(**dummy)
            except Exception as e:
                print(f'Unexpected behaviour in preheat_bnb. Info: most likely the provided model or tokenizer in not compatible witht the baked dummy input here.')
                raise e
        elif isinstance(tokenizer, Wav2Vec2Processor):
            try:
                dummy_features = torch.randn(
                    1, 56080, dtype=model.dtype, device=model.device
                )
                dummy = { "input_values": dummy_features,
                        "attention_mask": None}
                model(**dummy)
            except Exception as e:
                print(f'Unexpected behaviour in preheat_bnb. Info: most likely the provided model or tokenizer in not compatible witht the baked dummy input here.')
                raise e
        elif isinstance(tokenizer,ASTFeatureExtractor):
            try:
                dummy_features = torch.randn(160000)
                dummy = tokenizer(
                    dummy_features,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                dummy = {k: v.to(model.dtype).to(model.device) for k, v in dummy.items()}
                model(**dummy)
            except Exception as e:
                print(f'Unexpected behaviour in preheat_bnb. Info: most likely the provided model or tokenizer in not compatible witht the baked dummy input here.')
                raise e
        else:
            print(f'Unknown tokenizer/processor in preheat_bnb: {type(tokenizer)}. Info: you can add your specific tokenizer/processor in this function as a separate \'elif isinstance\' statement.')
            raise ValueError
        
    return

def inpsect_bnb(model: nn.Module, tokenizer: any, seed: int):
    preheat_bnb(model, tokenizer, seed)

    total_invalid = 0
    allowed = torch.tensor(QUANT_LUT_ALLOWED, device=model.device)
    for _, module in model.named_modules():
        if isinstance(module, bitsandbytes.nn.modules.Linear8bitLt):
            state = module.state
            if state.CB is None:
                continue
            W = state.CB

            mask = ~torch.isin(W, allowed)
            total_invalid += mask.sum().item()
    print(f'Total invalid values: {total_invalid}')

def count_params_bnb(model: nn.Module, tokenizer: any, seed: int):
    preheat_bnb(model, tokenizer, seed)

    total_fp_params = 0
    remaining_fp_params = 0
    int8_params = 0
    scale_params = 0

    for p in model.parameters():
        total_fp_params += p.numel()
        if p.dtype != torch.int8:
            remaining_fp_params += p.numel()

    for _, module in model.named_modules():
        if isinstance(module, bitsandbytes.nn.modules.Linear8bitLt):
            state = module.state

            # 8-bit weights
            if getattr(state, 'CB', None) is not None:
                int8_params += state.CB.numel()

            # FP16 scales
            if getattr(state, 'SCB', None) is not None:
                scale_params += state.SCB.numel()

    print(f'Total params: {total_fp_params}')
    print(f'Quantized params: {int8_params} ( {int8_params * 100 / total_fp_params:.2f}% )')
    print(f'Scale params: {scale_params}')
    print(f'Remaining FP params (unquantized): {remaining_fp_params}')

    effective_total = int8_params + scale_params + remaining_fp_params
    print(f'\nEffective parameter count (after quant): {effective_total}')

def apply_custom_quantization_bnb(model: nn.Module, tokenizer: any, seed: int, prints:bool=False):
    preheat_bnb(model, tokenizer, seed)

    count = 0
    quant_lut = BOOTH_QUANT_VALUE_TORCH.to(model.device)
    for _, module in model.named_modules():
        if isinstance(module, bitsandbytes.nn.modules.Linear8bitLt):
            state = module.state
            if state.CB is None:
                continue
            W = state.CB
            idx = (W.to(torch.int16) + 128).to(torch.long)
            W_new = quant_lut[idx].to(torch.int8)
            state.CB.copy_(W_new)
            count += W.numel()
    if(prints):
        print(f'Approximated {count} total parameters')
