#!/usr/bin/env python
# whisper_multigpu_bengali.py
#
# Fine‑tunes Whisper‑Medium on Common Voice Bengali with two A100 GPUs.
# Launch:  torchrun --standalone --nproc_per_node=2 whisper_multigpu_bengali.py

import os, random, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ---------- environment quirks ----------
os.environ.setdefault("NCCL_P2P_DISABLE", "1")   # peer‑access race fix
os.environ.setdefault("NCCL_DEBUG", "WARN")      # tidy NCCL logs
os.environ.setdefault("PYTHONHASHSEED", "42")    # full determinism

import numpy as np
import torch

# ---------- pin rank to GPU -------------
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
world_size = int(os.environ.get("WORLD_SIZE", 1))

from datasets import Audio, DatasetDict, load_dataset, load_from_disk
import evaluate
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
)

# ---------- reproducibility -------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed()

# ---------- constants -------------------
MODEL_NAME     = "openai/whisper-medium"
LANGUAGE       = "bengali"
TASK           = "transcribe"
DATASET_NAME   = "mozilla-foundation/common_voice_13_0"
DATASET_CONFIG = "bn"
DATA_FRACTION  = 0.25
SAMPLING_RATE  = 16_000
OUTPUT_DIR     = "whisper-bn"
ARROW_CACHE    = os.path.join(OUTPUT_DIR, "arrow_cache")
BATCH_SIZE     = 16
MAX_STEPS      = 6_000
NUM_PROC       = max(1, min(os.cpu_count() // world_size, 4))
PUSH_TO_HUB    = local_rank == 0  # root rank only

# ---------- logging ---------------------
logging.basicConfig(
    level=logging.INFO if local_rank == 0 else logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("whisper_bengali.log")],
)
log = logging.getLogger(__name__)
if local_rank == 0:
    log.info(f"NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']}  rank={local_rank}/{world_size-1}")

# ---------- dataset ---------------------
cv = DatasetDict()
cv["train_full"] = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train+validation")
cv["test_full"]  = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
train_n = int(len(cv["train_full"]) * DATA_FRACTION)
test_n  = int(len(cv["test_full"])  * DATA_FRACTION)
cv["train"] = cv["train_full"].shuffle(seed=42).select(range(train_n))
cv["test"]  = cv["test_full"].shuffle(seed=42).select(range(test_n))
del cv["train_full"], cv["test_full"]
cv = cv.remove_columns(["accent","age","client_id","down_votes","gender","locale","path","segment","up_votes"])

# ---------- processor -------------------
fe  = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tok = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
proc= WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
cv  = cv.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def prepare(example):
    a = example["audio"]
    example["input_features"] = fe(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
    example["labels"] = tok(example["sentence"]).input_ids
    return example

# ---------- one‑time vectorise ----------
if os.path.isdir(ARROW_CACHE):
    vect = load_from_disk(ARROW_CACHE)
    if local_rank == 0: log.info("Arrow cache loaded.")
else:
    if local_rank == 0:
        log.info("Vectorising … first run only.")
        vect = cv.map(
            prepare,
            remove_columns=cv.column_names["train"],
            num_proc=NUM_PROC,
            desc="vectorising",
            load_from_cache_file=False,
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        vect.save_to_disk(ARROW_CACHE)
        log.info("Arrow cache saved.")
    if world_size > 1: torch.distributed.barrier()
    if local_rank != 0: vect = load_from_disk(ARROW_CACHE)

# ---------- collator --------------------
@dataclass
class Collator:
    processor: Any
    def __call__(self, feats: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        ins   = [{"input_features": f["input_features"]} for f in feats]
        batch = self.processor.feature_extractor.pad(ins, return_tensors="pt")
        labs  = [{"input_ids": f["labels"]} for f in feats]
        labs  = self.processor.tokenizer.pad(labs, return_tensors="pt")
        label_ids = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        if (label_ids[:,0]==self.processor.tokenizer.bos_token_id).all().cpu().item():
            label_ids = label_ids[:,1:]
        batch["labels"] = label_ids
        return batch
collator = Collator(proc)

# ---------- model -----------------------
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
forced = proc.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
model.config.forced_decoder_ids = forced
if hasattr(model, "generation_config"):
    model.generation_config.forced_decoder_ids = forced
log.info(f"forced_decoder_ids={forced}")

# ---------- metric ----------------------
wer_metric = evaluate.load("wer")
def compute_metrics(out):
    lab = np.where(out.label_ids == -100, tok.pad_token_id, out.label_ids)
    pred = tok.batch_decode(out.predictions, skip_special_tokens=True)
    ref  = tok.batch_decode(lab, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred, references=ref)}

# ---------- training args ---------------
args = Seq2SeqTrainingArguments(
    output_dir             = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    max_steps              = MAX_STEPS,
    learning_rate          = 1e-5,
    warmup_steps           = 500,
    gradient_checkpointing = True,
    fp16                   = True,
    eval_strategy          = "steps",          # <- compact name needed
    save_steps             = 500,
    eval_steps             = 500,
    logging_steps          = 25,
    predict_with_generate  = True,
    generation_max_length  = 225,
    load_best_model_at_end = True,
    metric_for_best_model  = "wer",
    greater_is_better      = False,
    report_to              = ["tensorboard"],
    ddp_find_unused_parameters = False,
    push_to_hub            = PUSH_TO_HUB,
    hub_model_id           = "ehzawad/whisper-medium-bn-25percent",
)

# ---------- trainer ---------------------
trainer = Seq2SeqTrainer(
    model           = model,
    args            = args,
    train_dataset   = vect["train"],
    eval_dataset    = vect["test"],
    data_collator   = collator,
    compute_metrics = compute_metrics,
    tokenizer       = fe,
)

# ---------- run -------------------------
if local_rank == 0: proc.save_pretrained(OUTPUT_DIR)
log.info("Training begins …")
trainer.train()
if local_rank == 0:
    trainer.save_model(OUTPUT_DIR)
    if PUSH_TO_HUB:
        trainer.push_to_hub(
            dataset_tags   = DATASET_NAME,
            dataset        = "Common Voice 13.0",
            dataset_args   = f"config:{DATASET_CONFIG}, split:test, {DATA_FRACTION*100:.0f}%",
            language       = LANGUAGE,
            model_name     = f"Whisper medium Bengali ({DATA_FRACTION*100:.0f}%)",
            finetuned_from = MODEL_NAME,
            tasks          = "automatic-speech-recognition",
            tags           = "hf-asr-leaderboard",
        )
    log.info("Training complete.")
