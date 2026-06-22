"""
Fine-tune KLUE-RoBERTa-base for development-relevance classification.

Binary classification:
  1 = article discusses a development-related situation in a developing country
  0 = domestic Korean news, other unrelated content

Input:  src/labels/sample_labeled.csv  (label_development_relevant column filled)
Output: models/devrel_classifier/      (HuggingFace model directory)
        src/processed/news/*_devrel.csv (full-corpus predictions, after --apply-all)

Usage:
  python train_devrel_classifier.py                        # train + eval
  python train_devrel_classifier.py --eval-only            # eval on held-out set only
  python train_devrel_classifier.py --apply-all            # apply to full corpus (needs GPU)
  python train_devrel_classifier.py --apply-all --year 2019
  python train_devrel_classifier.py --threshold 0.4        # lower threshold for more recall
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config

BASE_DIR      = Path(__file__).parent.parent
LABELS_DIR    = config.LABELS_DIR
MODELS_DIR    = config.MODELS_DIR
NEWS_DIR      = config.NEWS_DATA_DIR
PROCESSED_DIR = BASE_DIR / "src" / "processed" / "news"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME    = "klue/roberta-base"
SAVED_DIR     = MODELS_DIR / "devrel_classifier"
LABEL_COL     = "label_development_relevant"
MAX_LEN       = 128    # title + keywords fits in 128 tokens
BATCH_TRAIN   = 16
BATCH_INFER   = 64
EPOCHS        = 4
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
SEED          = 2025
VAL_SPLIT     = 0.20   # 20% held-out validation


# ── Text construction ──────────────────────────────────────────────────────────

def build_text(row: pd.Series) -> str:
    """Concatenate title + keywords for classification input."""
    parts = []
    for col in ["title", "keywords", "top_keywords", "제목", "키워드"]:
        v = str(row.get(col, "") or "").strip()
        if v:
            parts.append(v)
        if len(parts) == 2:
            break
    return " ".join(parts)[:512]


# ── Dataset ────────────────────────────────────────────────────────────────────

class ArticleDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    texts_train: list[str],
    labels_train: list[int],
    texts_val: list[str],
    labels_val: list[int],
    device: torch.device,
) -> tuple:
    """Fine-tune and return (model, tokenizer)."""
    logger.info("Loading %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    def encode(texts: list[str]) -> dict:
        return tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )

    enc_train = {k: v.numpy() for k, v in encode(texts_train).items()}
    enc_val   = {k: v.numpy() for k, v in encode(texts_val).items()}

    train_ds  = ArticleDataset(enc_train, labels_train)
    val_ds    = ArticleDataset(enc_val,   labels_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_INFER, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_state  = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        preds_all, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                lbl    = batch.pop("labels").cpu().numpy()
                out    = model(**batch)
                probs  = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                preds  = (probs >= 0.5).astype(int)
                preds_all.extend(preds)
                probs_all.extend(probs)
                labels_all.extend(lbl)

        val_f1  = f1_score(labels_all, preds_all, zero_division=0)
        val_acc = accuracy_score(labels_all, preds_all)
        try:
            val_auc = roc_auc_score(labels_all, probs_all)
        except ValueError:
            val_auc = float("nan")

        logger.info(
            "Epoch %d  loss=%.4f  val_F1=%.4f  val_Acc=%.4f  val_AUC=%.4f",
            epoch, avg_loss, val_f1, val_acc, val_auc,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  → New best model saved (F1=%.4f)", best_val_f1)

    # Restore best weights
    model.load_state_dict(best_state)
    return model, tokenizer


# ── Evaluation report ──────────────────────────────────────────────────────────

def eval_report(model, tokenizer, texts_val, labels_val, device, threshold: float = 0.5):
    model.eval()
    tokenizer_out = tokenizer(
        texts_val, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    ds     = ArticleDataset({k: v.numpy() for k, v in tokenizer_out.items()})
    loader = DataLoader(ds, batch_size=BATCH_INFER, shuffle=False)

    probs_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(probs)

    preds = (np.array(probs_all) >= threshold).astype(int)

    print("\n" + "=" * 55)
    print("DEVELOPMENT-RELEVANCE CLASSIFIER — VALIDATION RESULTS")
    print("=" * 55)
    print(f"Threshold: {threshold}   N={len(labels_val)}")
    print(f"Accuracy : {accuracy_score(labels_val, preds):.4f}")
    print(f"F1       : {f1_score(labels_val, preds, zero_division=0):.4f}")
    print(f"Precision: {precision_score(labels_val, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(labels_val, preds, zero_division=0):.4f}")
    try:
        print(f"AUC-ROC  : {roc_auc_score(labels_val, probs_all):.4f}")
    except ValueError:
        pass
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(labels_val, preds)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("\nClassification report:")
    print(classification_report(labels_val, preds, target_names=["not-dev", "dev-relevant"]))


# ── Apply to full corpus ───────────────────────────────────────────────────────

def apply_to_corpus(model, tokenizer, device, year: str | None, threshold: float) -> None:
    pattern = f"news_{year}_*.csv" if year else "news_*.csv"
    files   = sorted(
        p for p in NEWS_DIR.glob(pattern)
        if not any(s in p.stem for s in ["_classified", "_oda", "_devrel"])
    )
    if not files:
        logger.warning("No files found in %s matching %s", NEWS_DIR, pattern)
        return

    logger.info("Applying to %d files (threshold=%.2f) ...", len(files), threshold)
    model.eval()

    for csv_path in tqdm(files, unit="file", desc="Applying"):
        out_path = PROCESSED_DIR / (csv_path.stem + "_devrel.csv")
        if out_path.exists():
            continue

        df    = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", low_memory=False)
        texts = [build_text(row) for _, row in df.iterrows()]

        probs_all = []
        for i in range(0, len(texts), BATCH_INFER):
            chunk = texts[i : i + BATCH_INFER]
            enc   = tokenizer(chunk, padding=True, truncation=True,
                              max_length=MAX_LEN, return_tensors="pt")
            enc   = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(probs)

        df["devrel_prob"]    = np.array(probs_all).round(4)
        df["devrel_relevant"] = (df["devrel_prob"] >= threshold).astype(int)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        n_rel = df["devrel_relevant"].sum()
        logger.info("  %s  →  %d/%d development-relevant", csv_path.name, n_rel, len(df))


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--labels",     default="",    help="Path to labeled CSV (default: src/labels/sample_labeled.csv)")
@click.option("--threshold",  default=0.5,   show_default=True, type=float,
              help="Probability threshold for development-relevant label")
@click.option("--eval-only",  is_flag=True,  help="Only evaluate saved model — do not retrain")
@click.option("--apply-all",  is_flag=True,  help="Apply saved model to full corpus after training")
@click.option("--year",       default="",    help="Restrict --apply-all to one year")
@click.option("--force",      is_flag=True,  help="Retrain even if saved model exists")
def main(labels: str, threshold: float, eval_only: bool, apply_all: bool, year: str, force: bool) -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    labels_path = Path(labels) if labels else LABELS_DIR / "sample_labeled.csv"
    if not labels_path.exists():
        click.echo(f"Labels file not found: {labels_path}")
        click.echo("Run 'python sample_for_labeling.py' first, fill in the sheet, "
                   "save as sample_labeled.csv, then re-run.")
        sys.exit(1)

    # ── Load labels ───────────────────────────────────────────────────────────
    df_lab = pd.read_csv(labels_path, dtype=str, encoding="utf-8-sig")
    if LABEL_COL not in df_lab.columns:
        click.echo(f"Column '{LABEL_COL}' not found in {labels_path.name}.")
        click.echo(f"Available columns: {list(df_lab.columns)}")
        sys.exit(1)

    df_lab = df_lab[df_lab[LABEL_COL].isin(["0", "1", 0, 1])].copy()
    df_lab["_label"] = df_lab[LABEL_COL].astype(int)
    texts  = [build_text(row) for _, row in df_lab.iterrows()]
    labels_list = df_lab["_label"].tolist()

    logger.info(
        "Loaded %d labeled examples  (dev-relevant=%d  not=%d)",
        len(labels_list), sum(labels_list), len(labels_list) - sum(labels_list),
    )
    if len(labels_list) < 20:
        click.echo("Need at least 20 labeled examples. Keep labeling.")
        sys.exit(1)

    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels_list, test_size=VAL_SPLIT, random_state=SEED, stratify=labels_list
    )
    logger.info("Split: %d train / %d val", len(texts_train), len(texts_val))

    # ── Load or train ─────────────────────────────────────────────────────────
    if not eval_only and (force or not SAVED_DIR.exists()):
        logger.info("Training %s ...", MODEL_NAME)
        model, tokenizer = train(texts_train, labels_train, texts_val, labels_val, device)
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(SAVED_DIR))
        tokenizer.save_pretrained(str(SAVED_DIR))
        logger.info("Model saved → %s", SAVED_DIR)
    else:
        logger.info("Loading saved model from %s ...", SAVED_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(SAVED_DIR))
        model     = AutoModelForSequenceClassification.from_pretrained(str(SAVED_DIR)).to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_report(model, tokenizer, texts_val, labels_val, device, threshold=threshold)

    # ── Apply to full corpus ──────────────────────────────────────────────────
    if apply_all:
        apply_to_corpus(model, tokenizer, device, year or None, threshold)
        click.echo("\nDone. Run run_classify.py to continue with SDG + sentiment.")


if __name__ == "__main__":
    main()
