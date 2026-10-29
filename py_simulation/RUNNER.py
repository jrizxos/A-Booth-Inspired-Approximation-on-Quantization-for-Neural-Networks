import os
import subprocess

CIFAR_OR_IMAGENET={
    'Alexnet.py'      : {'I':5, 'C':5},
    'MNASNet.py'      : {'I':5, 'C':5},
    'SqueezeNet.py'   : {'I':5, 'C':5},
    'Mobilenet.py'    : {'I':5, 'C':5},
    'Resnet.py'       : {'I':5, 'C':5},
    'VGG16.py'        : {'I':5, 'C':5},
    'VGG16_BN.py'     : {'I':5, 'C':5},
    'InceptionV3.py'  : {'I':5, 'C':5},
    'GoogLeNet.py'    : {'I':5, 'C':5},
    'DenseNet.py'     : {'I':5, 'C':5},
}
OTHERS={
    'AST.py'            :1,
    'BERT-large.py'     :1,
    'BLOOM-7b1.py'      :1,
    'DistilBERT.py'     :1,
    'GPT2-xl.py'        :1,
    'jamba.py'          :1,
    'mamba.py'          :1,
    'mms-large.py'      :1,
    'MNIST_CNN.py'      :1,
    'MNIST_MLP.py'      :1,
    'OPT-13B.py'        :1,
    'qwen_moe.py'       :1,
    'tiny_moe.py'       :1,
    'ViT.py'            :1,
    'Whisper-large.py'  :1,
    'zamba2.py'         :1,
}

if __name__ == '__main__':
    CWD = os.path.dirname(os.path.realpath(__file__))
    TRAIN_RES = os.path.join(CWD, 'results')
    os.chdir(CWD)

    for model in OTHERS:
        for run_id in range(1, OTHERS[model]+1):
            log_file = f"{model}_{run_id}.log"

            print(f'Running {model} run {run_id}')

            with open(log_file, "w") as f:
                subprocess.run(['python', model, str(run_id)], stdout=f, check=True)

    for model in CIFAR_OR_IMAGENET:
        for dataset in ('C', 'I'):
            runs_num = CIFAR_OR_IMAGENET[model][dataset]
            for run_id in range(5, runs_num+5):
                log_file = f"{model}_{run_id}_{'Imagenet' if dataset=='I' else 'CIFAR10'}.log"

                print(f'Running {model} for {'Imagenet' if dataset=='I' else 'CIFAR10'}, run {run_id}')

                with open(log_file, "w") as f:
                    subprocess.run(['python', model, str(run_id), dataset], stdout=f, check=True)