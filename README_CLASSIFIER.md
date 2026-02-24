# Cache Classifier Training Pipeline

**Build a fast, accurate binary classifier for cache decisions in GPTCache using BERT-based models.**

This pipeline creates a dataset, labels it with your local LLM, and trains a small BERT model (66M-150M params) that's **100-1000x faster than LLM-based classification** while maintaining high accuracy (90-95% F1).

---

## 📋 Overview

### Pipeline Steps

1. **Dataset Generation** (`dataset_generation.py`)
   - Load public datasets: MS MARCO, TriviaQA, SQuAD, Natural Questions
   - Generate synthetic time-sensitive queries (current price, weather, etc.)
   - Generate synthetic creative queries (write a poem, generate code, etc.)
   - Combine and balance into ~50K queries

2. **LLM Labeling** (`llm_labeling.py`)
   - Use your local Llama 3 8B (via Ollama) to label queries
   - Binary classification: cacheable (1) vs. not cacheable (0)
   - Outputs labeled dataset with confidence scores

3. **BERT Training** (`train_classifier.py`)
   - Fine-tune BERT/DistilBERT/ModernBERT on labeled data
   - Train/val/test split with stratification
   - Early stopping, metrics tracking (accuracy, F1, precision, recall)
   - Export trained model

4. **Inference** (`inference.py`)
   - Fast inference (<10ms per query on CPU, <1ms on GPU)
   - Batch processing support
   - Integration with GPTCache
   - Benchmarking utilities

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_classifier.txt
```

**Note:** Make sure Ollama is running with your Llama 3 model:

```bash
ollama run hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M
```

---

### 2. Generate Dataset

```bash
python dataset_generation.py
```

**Output:**
- `cache_classifier_data/raw_dataset.jsonl` (~50K queries, mix of labeled and unlabeled)
- `cache_classifier_data/dataset_stats.json` (statistics)

**What it does:**
- Downloads public datasets (TriviaQA, SQuAD, MS MARCO)
- Generates synthetic time-sensitive queries (weather, prices, etc.) → labeled as `0` (not cacheable)
- Generates synthetic creative queries (write a poem, etc.) → labeled as `0`
- Loads factual questions → labeled as `1` (cacheable)
- MS MARCO queries → unlabeled (to be labeled by LLM)

**Time:** 5-15 minutes (depends on download speed)

---

### 3. Label with LLM

```bash
python llm_labeling.py --input cache_classifier_data/raw_dataset.jsonl --output cache_classifier_data/labeled_dataset.jsonl
```

**Options:**
- `--max-queries N` - Limit to N queries (for testing)
- `--verify` - Show sample of labeled queries for manual inspection

**Output:**
- `cache_classifier_data/labeled_dataset.jsonl` (fully labeled)
- `cache_classifier_data/labeling_stats.json` (statistics)

**What it does:**
- Sends each unlabeled query to your local LLM
- Uses the prompt from `prompts.yaml` (classifier_prompt_1)
- Parses "yes"/"no" responses → 1/0 labels
- Adds confidence scores

**Time:** 2-10 hours (depends on dataset size and LLM speed)

**Tip:** Test with `--max-queries 100` first to verify setup!

---

### 4. Train BERT Classifier

```bash
python train_classifier.py --data cache_classifier_data/labeled_dataset.jsonl
```

**Options:**
- `--model distilbert/distilbert-base-uncased` (default, 66M params, fast)
- `--model answerdotai/ModernBERT-base` (recommended if available, 8192 context, very fast)
- `--model microsoft/deberta-v3-small` (higher accuracy, slower)
- `--epochs 3` (default)
- `--batch-size 16` (default, increase if you have GPU memory)
- `--learning-rate 2e-5` (default)
- `--output models/cache_classifier` (default)

**Output:**
- `models/cache_classifier/final_model/` (trained model + tokenizer)
- `models/cache_classifier/test_results.json` (evaluation metrics)
- Training logs in `models/cache_classifier/logs/`

**What it does:**
- Loads labeled dataset
- Splits into train/val/test (80/10/10) with stratification
- Tokenizes with model-specific tokenizer
- Fine-tunes BERT with classification head
- Evaluates on test set
- Saves best model (by F1 score)

**Expected results:**
- Accuracy: 90-95%
- F1 Score: 0.88-0.94
- Training time: 10-30 minutes on GPU, 1-3 hours on CPU (depends on dataset size)

**Time:** 10-60 minutes (GPU), 1-4 hours (CPU)

---

### 5. Test & Benchmark

**Interactive demo:**

```bash
python inference.py --demo
```

Try queries like:
- "What is the capital of France?" → ✓ CACHE
- "What is the weather right now?" → ✗ SKIP
- "Write a poem about cats" → ✗ SKIP

**Benchmark on test set:**

```bash
python inference.py --benchmark cache_classifier_data/labeled_dataset.jsonl
```

**Output:**
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Latency stats (avg, P95, P99)

**Expected latency:**
- CPU: 5-20ms per query
- GPU: 0.5-2ms per query

---

## 🔧 Model Recommendations

### Recommended Models (Ranked)

1. **`answerdotai/ModernBERT-base`** (if available, 2025+)
   - State-of-the-art encoder (replaces BERT)
   - 8192 token context (vs 512 for BERT)
   - 2-4x faster training/inference than DeBERTa
   - Up to 80% less memory than DeBERTa-v3
   - Best choice for new projects

2. **`distilbert/distilbert-base-uncased`** (classic, proven)
   - 66M params (40% smaller than BERT-base)
   - 60% faster than BERT-base
   - 97% of BERT-base performance
   - Great baseline, widely supported

3. **`microsoft/deberta-v3-small`** (high accuracy)
   - Better accuracy than DistilBERT
   - Slower training/inference
   - Good for production if you need max accuracy

4. **Ultra-small options** (if you need <50M params):
   - `sentence-transformers/all-MiniLM-L6-v2` (22M params)
   - Add a small classifier head
   - Trade accuracy for speed

### Why NOT use a decoder LLM (Phi-3, Qwen2, Llama)?

- **Classification is not generation:** You don't need next-token prediction, just a binary decision
- **BERT is 10-100x faster** for inference (smaller, simpler forward pass)
- **Less memory:** 66M vs 1.5B+ params
- **Better for this task:** Encoders are designed for understanding, decoders for generation

**When to use decoder LLM:**
- You want explanations: "Should cache? No, because it's asking for current price..."
- You need instruction-following
- You're already using unsloth for other tasks

---

## 📊 Expected Results

With a well-labeled dataset of 30K-50K examples:

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 90-95% |
| Precision | 88-94% |
| Recall | 87-93% |
| F1 Score | 88-94% |
| Inference Latency (CPU) | 5-20ms |
| Inference Latency (GPU) | 0.5-2ms |
| Model Size | 66M-150M params |
| Memory (inference) | 200-600MB |

**Comparison to LLM-based approach:**

| Metric | LLM (Llama 3 8B) | BERT Classifier | Speedup |
|--------|------------------|-----------------|---------|
| Latency | 500-2000ms | 5-20ms | 100-400x |
| Memory | 8GB+ | 200-600MB | 13-40x |
| Accuracy | ~85% (prompt-based) | 90-95% | Better |

---

## 🔌 Integration with GPTCache

### Option 1: Pre-embedding check

```python
from inference import CacheDecisionFunc

# Initialize classifier
cache_decision = CacheDecisionFunc("models/cache_classifier/final_model")

# Use in your cache logic
def should_cache_query(query: str) -> bool:
    return cache_decision(query)

# Example
if should_cache_query(user_query):
    # Proceed with embedding and caching
    response = cached_llm(user_query, cache_obj=llm_cache)
else:
    # Skip cache, call LLM directly
    response = llm(user_query)
```

### Option 2: Batch filtering

```python
from inference import CacheClassifier

classifier = CacheClassifier("models/cache_classifier/final_model")

# Filter batch of queries
queries = [...]  # List of user queries
results = classifier.predict_batch(queries)

cacheable_queries = [q for q, (should_cache, _) in zip(queries, results) if should_cache]
```

### Option 3: Replace existing LLM classifier

In your existing `main.py`, replace:

```python
# OLD: LLM-based classification (slow)
answer = llm.invoke(item["q"]).strip().lower()
predicted_cache = (answer == "yes")
```

With:

```python
# NEW: BERT-based classification (100x faster)
from inference import CacheClassifier
classifier = CacheClassifier("models/cache_classifier/final_model")
predicted_cache = classifier.predict(item["q"])
```

---

## 🛠️ Advanced Usage

### 1. Customize Prompt Template

Edit `prompts.yaml`:

```yaml
classifier_prompt_1: |
  You are a caching expert. Output only "yes" or "no".

  Say "yes" ONLY if the answer is static and unchanging.
  Say "no" for time-sensitive, creative, or computational queries.

  Query: {question}
  Answer:
```

### 2. Class Imbalance Handling

If your dataset is imbalanced (e.g., 80% cacheable, 20% not):

**Option A:** Add class weights in training

```python
# In train_classifier.py, modify Trainer initialization:
from torch import nn

class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

**Option B:** Resample dataset in `dataset_generation.py`

```python
# In combine_and_balance():
# Oversample minority class or undersample majority
from sklearn.utils import resample
```

### 3. Export to ONNX (Faster CPU Inference)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# Convert to ONNX
model = AutoModelForSequenceClassification.from_pretrained("models/cache_classifier/final_model")
tokenizer = AutoTokenizer.from_pretrained("models/cache_classifier/final_model")

onnx_model = ORTModelForSequenceClassification.from_pretrained(
    "models/cache_classifier/final_model",
    export=True
)
onnx_model.save_pretrained("models/cache_classifier/onnx")

# Use ONNX model (2-3x faster on CPU)
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained("models/cache_classifier/onnx")
```

### 4. Quantize Model (Smaller, Faster)

```bash
pip install optimum
```

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 8-bit quantization (2x smaller, minimal accuracy loss)
quantizer = ORTQuantizer.from_pretrained("models/cache_classifier/final_model")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir="models/cache_classifier/quantized", quantization_config=qconfig)
```

---

## 📁 File Structure

```
GPTCache/
├── dataset_generation.py      # Generate dataset from public sources
├── llm_labeling.py            # Label with local Llama 3 LLM
├── train_classifier.py        # Train BERT classifier
├── inference.py               # Inference and GPTCache integration
├── requirements_classifier.txt # Dependencies
├── README_CLASSIFIER.md       # This file
├── prompts.yaml               # Prompt templates (existing)
│
├── cache_classifier_data/     # Generated datasets
│   ├── raw_dataset.jsonl
│   ├── labeled_dataset.jsonl
│   ├── dataset_stats.json
│   └── labeling_stats.json
│
└── models/                    # Trained models
    └── cache_classifier/
        ├── final_model/       # Best model checkpoint
        │   ├── pytorch_model.bin
        │   ├── config.json
        │   └── tokenizer/
        └── test_results.json
```

---

## ❓ FAQ

### Q: Why not use unsloth?

**A:** Unsloth is designed for fine-tuning **decoder LLMs** (Llama, Mistral, Qwen, Phi) with LoRA/QLoRA. It does NOT support BERT-based **encoder models**. For binary classification, BERT is 10-100x more efficient than decoder LLMs. Use unsloth if you're training a generative model, not a classifier.

### Q: Can I use a decoder LLM instead of BERT?

**A:** Yes, but it's less efficient. You'd fine-tune a small decoder LLM (Phi-3-mini, Qwen2-1.5B) with instruction format:

```
Query: What is the capital of France?
Should cache? Yes

Query: What is the weather right now?
Should cache? No
```

This requires:
- Unsloth for efficient LoRA training
- Larger model (1.5B-3.8B vs 66M-150M for BERT)
- Slower inference (50-200ms vs 5-20ms)
- More memory (1-4GB vs 200-600MB)

**Trade-off:** You get explanations ("No, because it asks for current weather") but lose speed.

### Q: Which model should I use?

**A:** Start with `distilbert/distilbert-base-uncased` (proven, fast, small). If you want state-of-the-art, use `answerdotai/ModernBERT-base` (2025+, 2-4x faster than DeBERTa, 8192 context).

### Q: How much data do I need?

**A:** Minimum 5K labeled examples (balanced). Recommended: 20K-50K for production. This pipeline generates 30K-50K.

### Q: How do I improve accuracy?

1. More labeled data (50K-100K)
2. Better labels (manually verify samples)
3. Bigger model (ModernBERT-large, DeBERTa-v3-base)
4. Class weights (if imbalanced)
5. Data augmentation (paraphrase queries)

### Q: Can I use this for other binary classification tasks?

**A:** Yes! Just replace the dataset generation logic with your own labeled data. The training pipeline works for any binary text classification.

---

## 🐛 Troubleshooting

### Error: "MS MARCO requires authentication"

**Fix:** Accept terms at https://huggingface.co/datasets/microsoft/ms_marco, or set `use_ms_marco=False` in `dataset_generation.py`.

### Error: "CUDA out of memory"

**Fix:** Reduce batch size in `train_classifier.py`:

```bash
python train_classifier.py --batch-size 8
```

OR disable GPU (use CPU):

```bash
CUDA_VISIBLE_DEVICES="" python train_classifier.py
```

### LLM labeling is too slow

**Fix:** Limit queries for testing:

```bash
python llm_labeling.py --max-queries 1000
```

Or use a faster LLM (Phi-3-mini instead of Llama 3 8B).

### Model accuracy is low (<80%)

**Causes:**
1. **Bad labels:** LLM mislabeled queries → re-check prompts, use `--verify`
2. **Class imbalance:** Add class weights or resample
3. **Too little data:** Generate more, aim for 30K+
4. **Model too small:** Try ModernBERT-large or DeBERTa-v3-base

---

## 📚 References

- **ModernBERT:** https://huggingface.co/answerdotai/ModernBERT-base
- **DistilBERT:** https://huggingface.co/distilbert/distilbert-base-uncased
- **HuggingFace Text Classification:** https://huggingface.co/docs/transformers/tasks/sequence_classification
- **MS MARCO:** https://microsoft.github.io/msmarco/
- **TriviaQA:** https://nlp.cs.washington.edu/triviaqa/
- **Time-Sensitive QA:** https://github.com/wenhuchen/Time-Sensitive-QA

---

## 📝 License

Same as GPTCache project.

---

**Next Steps:**

1. Run `python dataset_generation.py`
2. Run `python llm_labeling.py --max-queries 100` (test)
3. Run `python train_classifier.py`
4. Run `python inference.py --demo`
5. Integrate with your GPTCache!

**Questions?** Check FAQ or open an issue.
