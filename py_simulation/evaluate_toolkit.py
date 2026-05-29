import os
import gc
import re

import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import ImageNet
from torchvision.datasets import CIFAR10

from transformers import AutoTokenizer, ViTImageProcessor, DataCollatorWithPadding, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

from sklearn.metrics import matthews_corrcoef, accuracy_score, average_precision_score
from scipy.stats import pearsonr, spearmanr

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

import evaluate

#### Constants ##########################################################################################################################

CWD = os.path.dirname(os.path.realpath(__file__))

##### Torch datasets ####################################################################################################################

def get_MNIST(transform: transforms.Compose) -> tuple[DataLoader,DataLoader]:
    train_dataset = MNIST(root='training_data', train=True, download=True, transform=transform)
    validation_dataset = MNIST(root='training_data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=8)

    return train_loader, validation_loader

def get_ImageNet(transform: transforms.Compose=None) -> tuple[DataLoader,DataLoader]:
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

def get_CIFAR(transform: transforms.Compose=None) -> tuple[DataLoader,DataLoader]:
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

def test_nn(network: nn.Module, test_loader: DataLoader, max_batches: int=0, device: str='cpu') -> tuple[float]:
    network = network.to(device)
    network.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    batch_idx = 0

    with torch.no_grad():
        for data, labels in test_loader:
            if max_batches>0 and batch_idx>max_batches:
                break
            
            data, labels = data.to(device), labels.to(device)
            outputs = network(data)
            if hasattr(outputs, 'logits'):  # Inception v3 case
                outputs = outputs.logits
            vals, top1_preds = outputs.topk(1, dim=1)
            top1_correct += (top1_preds.squeeze(1) == labels).sum().item()

            vals, top5_preds = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in top5_preds[i] for i in range(labels.size(0))])

            batch_idx += 1

            total += labels.size(0)

    t1 = (top1_correct / total)*100
    t5 = (top5_correct / total)*100
    print(f'Test set Top-1 Acc: {t1:.2f}%, Top-5 Acc: {t5:.2f}%')
    return t1,t5

def test_vit(network: nn.Module, test_loader: DataLoader, processor: ViTImageProcessor, max_batches: int=0, device: str='cpu') -> tuple[float]:
    network.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    batch_idx = 0

    with torch.no_grad():
        for data, labels in test_loader:
            if max_batches>0 and batch_idx>max_batches:
                break
            
            data, labels = data.to(device), labels.to(device)
            inputs = processor(
                images=data,
                return_tensors='pt',
                do_rescale=False
            ).to(device)
            outputs = network(**inputs)
            logits = outputs.logits 
            
            top1_preds = logits.argmax(dim=1)
            top1_correct += (top1_preds == labels).sum().item()
    
            top5_preds = torch.topk(logits, k=5, dim=1).indices
            top5_correct += (
                top5_preds == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            batch_idx += 1

            total += labels.size(0)

    t1 = (top1_correct / total)*100
    t5 = (top5_correct / total)*100
    print(f'Test set Top-1 Acc: {t1:.2f}%, Top-5 Acc: {t5:.2f}%')
    return t1,t5

##### HuggingFace datasets ##############################################################################################################

# glue
def make_glue_val_loader(task_name: str, task_cfg: dict, batch_size: int = 32):
    max_length = task_cfg.get('max_len', 128)
    
    # Load dataset
    dataset = load_dataset('glue', task_name, split='validation')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(task_cfg['repo'], use_fast=True)

    text_cols = task_cfg['text_columns']

    def tokenize_fn(examples):
        if len(text_cols) == 1:
            return tokenizer(
                examples[text_cols[0]],
                truncation=True,
                max_length=max_length,
            )
        else:
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                truncation=True,
                max_length=max_length,
            )

    dataset = dataset.map(tokenize_fn, batched=True)

    if 'label' in dataset.column_names:
        dataset = dataset.rename_column('label', 'labels')
    
    dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels'],
    )

    collator = DataCollatorWithPadding(tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

def evaluate_glue(model: nn.Module, task_name: str, task_cfg: dict, device: str):
    model.eval()
    
    dataloader = make_glue_val_loader(task_name, task_cfg)

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            if task_cfg['eval_func'] == 'pearson_spearman':
                preds = logits.squeeze(-1)
            else:
                preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if task_cfg['eval_func'] == 'accuracy':
        return accuracy_score(all_labels, all_preds)
    elif task_cfg['eval_func'] == 'matthews':
        return matthews_corrcoef(all_labels, all_preds)
    elif task_cfg['eval_func'] == 'pearson_spearman':
        pearson = pearsonr(all_labels, all_preds)[0]
        spearman = spearmanr(all_labels, all_preds)[0]
        return (pearson + spearman) / 2
    else:
        raise ValueError(f'Unknown eval_func: {task_cfg['eval_func']}')

def get_dataset(name: str, max_samples: int = None):
    dataset_piqa = load_dataset(name, split='validation', trust_remote_code=True)
    if max_samples is not None:
        dataset_piqa = dataset_piqa.select(range(max_samples))
    return dataset_piqa

# LM eval harness
def eval_lm_eval_existing(model: str, tasks: list, device: str) -> float:
    total_acc = 0
    acc_tasks = 0

    for task in tasks:
        results = evaluator.simple_evaluate(
            model='hf',
            model_args={
                'pretrained':model,
                'trust_remote_code':model=='zai-org/codegeex4-all-9b'
                },
            tasks=[task],
            batch_size=1,
            device=device,
            confirm_run_unsafe_code=True
        )

        task_res = results['results'][task]

        if 'acc,none' in task_res:
            acc = task_res['acc,none']
            total_acc += acc
            acc_tasks += 1
            print(f'{task} | accuracy: {acc:.4f}')
        else:
            print(f'Warning: {task} acc,none not included in results\n{task_res}')
        
        torch.cuda.empty_cache()
        gc.collect()

    m_acc =  total_acc/acc_tasks if acc_tasks>0 else None

    return m_acc

def eval_lm_eval_custom(model: nn.Module, tasks: list, device: str) -> float:
    total_acc = 0
    acc_tasks = 0

    for task in tasks:
        results = evaluator.simple_evaluate(
            model=HFLM(pretrained=model),
            tasks=[task],
            batch_size=1,
            device=device,
            confirm_run_unsafe_code=True
        )

        task_res = results['results'][task]

        if 'acc,none' in task_res:
            acc = task_res['acc,none']
            total_acc += acc
            acc_tasks += 1
            print(f'{task} | accuracy: {acc:.4f}')
        else:
            print(f'Warning: {task} acc,none not included in results\n{task_res}')
        
        torch.cuda.empty_cache()
        gc.collect()

    m_acc =  total_acc/acc_tasks if acc_tasks>0 else None

    return m_acc

# ASR
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_asr(model, processor, tasks):
    wer_metric = evaluate.load('wer')

    total_wer = 0
    for task in tasks.keys():
        dataset = load_dataset(
            tasks[task]['hf_name'],
            tasks[task]['config'],
            split=tasks[task]['split']
        )

        predictions = []
        references = []

        for sample in tqdm(dataset):
            audio = sample['audio']
            inputs = processor(
                    audio['array'],
                    sampling_rate=audio['sampling_rate'],
                    return_tensors='pt'
                )

            if isinstance(model, WhisperForConditionalGeneration):
                inputs = {
                    k: v.to(device=model.device, dtype=model.dtype)
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    generated_ids = model.generate(**inputs)

                transcription = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            else:
                if len(audio['array']) < 4000: # here Wav2Vec2 for mms-large crashes for very sort samples
                    continue
                inputs = {
                    "input_values": inputs["input_values"].to(model.dtype).to(model.device),
                    "attention_mask": inputs.get("attention_mask", None)
                }
                if inputs["attention_mask"] is not None:
                    inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

                with torch.no_grad():
                    logits = model(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]


            predictions.append(normalize_text(transcription))
            sample_text = 'text' if 'text' in sample else 'normalized_text'
            references.append(normalize_text(sample[sample_text]))

        wer = wer_metric.compute(
            predictions=predictions,
            references=references
        )
        print(f'{task} WER: {wer:.4f}%')
        total_wer += wer

    return total_wer/len(tasks.keys())

# Speech classification
def ast_forward(model, feature_extractor, audio_array, sampling_rate):
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.dtype).to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    return outputs.logits

def get_audioset():
    audioset = load_dataset(
        "agkphysics/AudioSet",
        "balanced",
        split="test"
    )
    audioset = audioset.cast_column(
        "audio",
        Audio(sampling_rate=16000)
    )
    return audioset

def eval_audioset(model, feature_extractor, dataset):
    label2id = model.config.label2id
    num_classes = model.config.num_labels

    y_true = []
    y_scores = []

    with torch.no_grad():
        for sample in tqdm(dataset):
            
            labels = sample["human_labels"]
            target = np.zeros(num_classes, dtype=np.float32)
            skip = False
            for lbl in labels:
                if lbl in label2id:
                    target[label2id[lbl]] = 1.0
                else:
                    skip = True     # normaly unreachable
                    break
            if skip:
                continue
            y_true.append(target)

            audio = sample["audio"]
            logits = ast_forward(model, feature_extractor, audio["array"], audio["sampling_rate"])
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            y_scores.append(probs)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    mAP = average_precision_score(y_true, y_scores, average="macro")
    print(f"AudioSet mAP: {mAP:.4f}")
    return mAP
