#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ffmpeg')
get_ipython().system('pip install git+https://github.com/openai/whisper.git')


# In[2]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install librosa')
get_ipython().system('pip install evaluate')
get_ipython().system('pip install jiwer')
get_ipython().system('pip install gradio')


# In[3]:


get_ipython().system('pip install git+https://github.com/huggingface/transformers')


# In[4]:


from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# import the relavant libraries for loggin in
from huggingface_hub import HfApi, HfFolder


# In[5]:


def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    folder = HfFolder()
    folder.save_token(token)

    return None

# STEP 0. Loging to Hugging Face
# get your account token from https://huggingface.co/settings/tokens
token = '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'
login_hugging_face(token)
print('We are logged in to Hugging Face now!')


# In[6]:


# STEP 1. Download Dataset
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "bn", split="train+validation")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "bn", split="test")


# In[7]:


common_voice = common_voice.remove_columns(
    ["accent",
     "age",
     "client_id",
     "down_votes",
     "gender",
     "locale",
     "segment",
     "up_votes"]
    )

common_voice


# In[8]:


common_voice["train"][1]


# In[9]:


# STEP 2. Prepare: Feature Extractor, Tokenizer and Data
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny") #small whisper model, we can use either medium or large

# - Load Tokenizer: WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="bengali", task="transcribe")


# In[10]:


input_str = common_voice["train"][1]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


# In[11]:


# STEP 3. Combine elements with WhisperProcessor
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="bengali", task="transcribe")


# In[12]:


processor


# In[13]:


# STEP 4. Prepare Data
print('| Check the random audio example from Common Voice dataset to see what form the data is in:')
print(f'{common_voice["train"][0]}\n')

# -> (1): Downsample from 48kHZ to 16kHZ
from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print('| Check the effect of downsampling:')
print(f'{common_voice["train"][1]}\n')


def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# Prepare and use function to prepare our data ready for the Whisper AI model
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1
    )


# In[14]:


# STEP 5. Training and evaluation

# STEP 5.1. Initialize the Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

data_collator


# In[15]:


# STEP 5.2. Define evaluation metric
import evaluate
metric = evaluate.load("wer")


# In[16]:


metric


# In[17]:


# STEP 5.3. Load a pre-trained Checkpoint
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

"""
Overide generation arguments:
- no tokens are forced as decoder outputs: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids
- no tokens are suppressed during generation: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens
"""
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# In[18]:


get_ipython().system('pip install --upgrade accelerate')
# !pip install accelerate
# !pip install transformers[torch]


# In[19]:


get_ipython().system('pip show torch')


# In[20]:


print(torch.__version__)


# In[21]:


# STEP 5.4. Define the training configuration
"""
Check for Seq2SeqTrainingArguments here:
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
"""
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="output/whisper-tiny-bn",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # testing
)


# In[22]:


get_ipython().system('pip install tensorboard')


# In[23]:


# Initialize a trainer.
"""
Forward the training arguments to the Hugging Face trainer along with our model,
dataset, data collator and compute_metrics function.
"""

def compute_metrics(pred):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Save processor object before starting training
processor.save_pretrained(training_args.output_dir)


# In[ ]:


# STEP 5.5. Training
"""
Training will take appr. 5-10 hours depending on your GPU.
"""
print('Training is started.')
trainer.train()  # <-- !!! Here the training starting !!!
print('Training is finished.')


# In[ ]:




