#!/usr/bin/env python
# whisper_multigpu_bengali_clean.py — silence the pad/eos warning

import os, random, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np, torch

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
world_size = int(os.environ.get("WORLD_SIZE", 1))

from datasets import Audio, DatasetDict, load_dataset, load_from_disk
import evaluate
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
)

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed()

MODEL = "openai/whisper-medium"
LANG, TASK = "bengali", "transcribe"
DS_NAME, DS_CFG = "mozilla-foundation/common_voice_13_0", "bn"
FRACTION, SR = .25, 16_000
OUT = "whisper-bn"
CACHE = f"{OUT}/arrow_cache"
BS, STEPS = 16, 6_000
PROC = max(1, min(os.cpu_count() // world_size, 4))
PUSH = local_rank == 0

logging.basicConfig(
    level=logging.INFO if local_rank==0 else logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("whisper_bengali.log")],
)
log = logging.getLogger(__name__)

# ----------------------- processors --------------------------------
fe = WhisperFeatureExtractor.from_pretrained(MODEL)
tok = WhisperTokenizer.from_pretrained(MODEL, language=LANG, task=TASK)
proc = WhisperProcessor.from_pretrained(MODEL, language=LANG, task=TASK)

#  add a distinct pad token so mask is unambiguous
if tok.pad_token_id == tok.eos_token_id:
    tok.add_special_tokens({"pad_token": "<pad>"})
    new_pad = tok.pad_token_id
else:
    new_pad = tok.pad_token_id

# ----------------------- dataset slice -----------------------------
cv = DatasetDict()
cv["train_full"] = load_dataset(DS_NAME, DS_CFG, split="train+validation")
cv["test_full"]  = load_dataset(DS_NAME, DS_CFG, split="test")
n_tr = int(len(cv["train_full"])*FRACTION)
n_te = int(len(cv["test_full"])*FRACTION)
cv["train"] = cv["train_full"].shuffle(seed=42).select(range(n_tr))
cv["test"]  = cv["test_full"].shuffle(seed=42).select(range(n_te))
del cv["train_full"], cv["test_full"]
cv = cv.remove_columns(["accent","age","client_id","down_votes","gender","locale","path","segment","up_votes"])
cv = cv.cast_column("audio", Audio(sampling_rate=SR))

def prep(x):
    a = x["audio"]
    x["input_features"] = fe(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
    x["labels"] = tok(x["sentence"]).input_ids
    return x

if os.path.isdir(CACHE):
    vec = load_from_disk(CACHE)
else:
    if local_rank==0:
        vec = cv.map(prep, remove_columns=cv.column_names["train"], num_proc=PROC, desc="vectorising", load_from_cache_file=False)
        os.makedirs(OUT, exist_ok=True); vec.save_to_disk(CACHE)
    if world_size>1: torch.distributed.barrier()
    if local_rank!=0: vec = load_from_disk(CACHE)

# ----------------------- collator ----------------------------------
@dataclass
class Collate:
    processor: Any
    def __call__(self, batch: List[Dict[str,Union[List[int],torch.Tensor]]]) -> Dict[str,torch.Tensor]:
        ins = [{"input_features": b["input_features"]} for b in batch]
        out = self.processor.feature_extractor.pad(ins, return_tensors="pt", return_attention_mask=True)
        labs= [{"input_ids": b["labels"]}           for b in batch]
        labs= self.processor.tokenizer.pad(labs, return_tensors="pt")
        labels = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        if (labels[:,0]==self.processor.tokenizer.bos_token_id).all().cpu().item(): labels = labels[:,1:]
        out["labels"] = labels
        return out
coll = Collate(proc)

# ----------------------- model -------------------------------------
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
model.resize_token_embeddings(len(tok))               # accommodate new pad row
model.config.pad_token_id = new_pad                   # now pad ≠ eos
model.config.use_cache = False
model.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language=LANG, task=TASK)
if hasattr(model,"generation_config"):
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids

# ----------------------- metrics -----------------------------------
wer = evaluate.load("wer")
def m(pred):
    l = np.where(pred.label_ids==-100, tok.pad_token_id, pred.label_ids)
    hyp = tok.batch_decode(pred.predictions, skip_special_tokens=True)
    ref = tok.batch_decode(l, skip_special_tokens=True)
    return {"wer": 100*wer.compute(predictions=hyp, references=ref)}

# ----------------------- trainer args ------------------------------
args = Seq2SeqTrainingArguments(
    output_dir         = OUT,
    per_device_train_batch_size = BS,
    per_device_eval_batch_size  = BS,
    max_steps          = STEPS,
    learning_rate      = 1e-5,
    warmup_steps       = 500,
    gradient_checkpointing = True,
    fp16               = True,
    eval_strategy      = "steps",
    save_steps         = 500,
    eval_steps         = 500,
    logging_steps      = 25,
    predict_with_generate = True,
    generation_max_length = 225,
    load_best_model_at_end = True,
    metric_for_best_model = "wer",
    greater_is_better  = False,
    report_to          = ["tensorboard"],
    ddp_find_unused_parameters = False,
    push_to_hub        = PUSH,
    hub_model_id       = "ehzawad/whisper-medium-bn-25percent",
)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = args,
    train_dataset   = vec["train"],
    eval_dataset    = vec["test"],
    data_collator   = coll,
    compute_metrics = m,
    processing_class= proc,
)

if local_rank==0: proc.save_pretrained(OUT)
trainer.train()
if local_rank==0:
    trainer.save_model(OUT)
    if PUSH:
        trainer.push_to_hub(
            dataset_tags=DS_NAME,
            dataset="Common Voice 13.0",
            dataset_args=f"config:{DS_CFG}, split:test, {FRACTION*100:.0f}%",
            language=LANG,
            model_name=f"Whisper medium Bengali ({FRACTION*100:.0f}%)",
            finetuned_from=MODEL,
            tasks="automatic-speech-recognition",
            tags="hf-asr-leaderboard",
        )
