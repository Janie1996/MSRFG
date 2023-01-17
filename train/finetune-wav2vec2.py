# -*- coding: UTF-8 -*-
"""
@file:wj_0803_wav2vec2.py
@author: Wei Jie
@date: 2022/8/3
@description: 使用wav2vec提取语音深度特征
"""
import pickle
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split
import torch
import os
import sys
import random
import warnings

warnings.filterwarnings('ignore')
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed=np.random.randint(10000)
seed_everything(seed)
print(seed)

def speech_file_to_array_fn(path):
    target_sampling_rate=16000
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):

    #label_list = ['ang','exc','fru','hap','neu','sad']
    label_list=[0,1,2,3,4,5]
    speech_list = [speech_file_to_array_fn(path) for path in examples['path']]
    target_list = [label_to_id(label, label_list) for label in examples["emotion"]]

    model_name_or_path = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'#"lighteternal/wav2vec2-large-xlsr-53-greek"
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    result = processor(speech_list, sampling_rate=16000)
    result["labels"] = list(target_list)

    return result



##model

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

# 模型训练返回数据类型
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense=nn.TransformerEncoderLayer(config.hidden_size,8)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.feas=[]

    def forward(self, features, **kwargs):
        x = features
        #x = self.dropout(x)
        x = self.dense(x)
        # self.feas.append(x)
        # pickle.dump(self.feas,open('den_train1.pickle','wb'))
        
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

from torch.nn import functional as F
class AttentionLSTM(nn.Module):
    """Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self):
        """
        LSTM with self-Attention model.
        :param cfg: Linguistic config object
        """
        super(AttentionLSTM, self).__init__()
        # self.batch_size = cfg.batch_size
        self.output_size = 6
        self.hidden_size = 300
        self.embedding_length = 1024

        self.dropout = torch.nn.Dropout(0.75)
        self.dropout2 = torch.nn.Dropout(0.4)

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """
        This method computes soft alignment scores for each of the hidden_states and the last hidden_state of the LSTM.
        Tensor Sizes :
            hidden.shape = (batch_size, hidden_size)
            attn_weights.shape = (batch_size, num_seq)
            soft_attn_weights.shape = (batch_size, num_seq)
            new_hidden_state.shape = (batch_size, hidden_size)
        :param lstm_output: Final output of the LSTM which contains hidden layer outputs for each sequence.
        :param final_state: Final time-step hidden state (h_n) of the LSTM
        :return: Context vector produced by performing weighted sum of all hidden states with attention weights
        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def extract(self, input):
        input = input.transpose(0, 1)
        input = self.dropout(input)

        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def classify(self, attn_output):
        attn_output = self.dropout2(attn_output)
        logits = self.label(attn_output)
        return logits.squeeze(1)

    def forward(self, input):
        attn_output = self.extract(input)
        logits = self.classify(attn_output)
        return logits

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.feas = []
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_values=input_values.cuda()
        attention_mask=attention_mask.cuda()
        labels=labels.cuda()
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        self.feas.append(hidden_states)
        #pickle.dump(self.feas,open('test.pickle','wb'))
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor,Wav2Vec2FeatureExtractor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


import numpy as np
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


from transformers import TrainingArguments


from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn

from transformers import (
    Trainer,
    is_apex_available,
)

if is_apex_available():
    print('-----')
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        self.use_amp=True
        self.use_apex=False
        inputs = self._prepare_inputs(inputs)
        # loss = self.compute_loss(model, inputs)
        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


if __name__ == "__main__":

    from datasets import load_dataset, load_metric


    data_files = {
        "train": "train.csv",
        "validation": "test.csv",
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(train_dataset)
    print(eval_dataset)
    # We need to specify the input and output column
    input_column = "path"
    output_column = "emotion"
    # we need to distinguish the unique labels in our SER dataset
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")
    from transformers import AutoConfig, Wav2Vec2Processor
    model_name_or_path='jonatasgrosman/wav2vec2-large-xlsr-53-english'
    # model_name_or_path = "lighteternal/wav2vec2-large-xlsr-53-greek"
    # model_name_or_path = './data/checkpoint-24050'
    pooling_mode = "max"
    # config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        #use_auth_token=True,
    )
    setattr(config, 'pooling_mode', pooling_mode)
    processor = Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english',padding=True)
    # processor = Wav2Vec2Processor.from_pretrained("lighteternal/wav2vec2-large-xlsr-53-greek",)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")
    print()


    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    model.freeze_feature_extractor()

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    is_regression = False
    training_args = TrainingArguments(
    output_dir="./data1",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    do_train=True,
    evaluation_strategy="epoch",  #"steps",
    num_train_epochs=20.0,
    fp16=True,
    save_strategy="epoch",
    save_steps=300,
    eval_steps=100,
    logging_steps=300,
    learning_rate=5e-5,
    save_total_limit=20,
    )
    if(training_args.do_train):
        train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=8,
            load_from_cache_file=False,
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=8,
        #remove_columns=['path','emotion','text']
    )
    # feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=feature_extractor,
        tokenizer=processor.feature_extractor,
    )
    if(training_args.do_train):
        trainer.train()
        print(seed)
    else:
        logits_cls, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="eval")
        #logits_ctc, logits_cls = predictions
        pred_ids = np.argmax(logits_cls, axis=-1)
        correct = np.sum(pred_ids == labels)
        acc = correct / len(pred_ids)
        print('correct:', correct, ', acc:', acc)



