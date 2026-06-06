
### Plain language. Step by step. One thing at a time.

---

## Before You Read This

This guide builds the entire Infinite Context Engine from scratch, in the order things actually need to exist before you can build the next thing. It is written assuming you know very little, but are willing to look things up. Every section includes **where to learn** what you need before you do it.

One rule: **never skip ahead**. Each phase depends on the previous one being done and working. If something doesn't work, fix it before moving on.

---

## The Big Picture — What You're Building and In What Order

Here is the complete sequence, simplified:

1. **Phase 1 — Your Workspace** — Folder structure, Python environment, Git
2. **Phase 2 — The Classifier** — The brain that decides what kind of message was sent
3. **Phase 3 — The Database** — PostgreSQL + pgvector, the unified memory store
4. **Phase 4 — The FastAPI Proxy** — The middleware that sits between Open WebUI and Ollama
5. **Phase 5 — Episodic Memory** — Storing every conversation turn
6. **Phase 6 — Background Workers** — Celery + Redis, the async brain
7. **Phase 7 — Post-Flight Evaluator** — Rates each exchange after it finishes
8. **Phase 8 — Codex Extractor** — Builds the knowledge graph
9. **Phase 9 — Retrieval Orchestrator** — Pulls the right memories at the right time
10. **Phase 10 — Memory Slots** — Persistent per-session context always injected
11. **Phase 11 — Remaining Systems** — Procedural memory, decay, sentinel, reflection

Phases 1–4 are the "minimum viable ICE" — after those, you have a working proxy with a working classifier. Everything else is layered on top.

---

## What's Already Installed on Your Machine

Based on what you have set up (CachyOS, RTX 4070, Ollama, Open WebUI, Docker, pyenv, CUDA), you have most of the heavy infrastructure already. This guide will note when you're using something you already have vs. installing something new.

---

---

# PHASE 1 — Your Workspace

**What this phase is:** Creating the folders and Python environment that everything will live in.

**What you'll learn:** How Python virtual environments work, why they matter, and basic project structure.

**Where to learn:**

- Python virtual environments: https://realpython.com/python-virtual-environments-a-primer/
- What pyenv does: https://github.com/pyenv/pyenv#readme

---

### Step 1.1 — Create the project root folder

Open your terminal. Navigate to wherever you keep your projects. Then:

```
mkdir ice
cd ice
```

Now create the folder structure all at once:

```
mkdir -p src/classifier
mkdir -p src/api
mkdir -p src/workers
mkdir -p src/memory
mkdir -p src/retrieval
mkdir -p data/raw_logs
mkdir -p data/labeled
mkdir -p data/datasets
mkdir -p models/classifier
mkdir -p scripts
mkdir -p tests
mkdir -p docker
mkdir -p alembic/versions
mkdir -p ingest_inbox
mkdir -p codex_inject
```

You should now have a project folder with subfolders inside it. That's all this step is.

---

### Step 1.2 — Set the Python version with pyenv

You already have pyenv. ICE uses Python 3.11 (stable, widely compatible with all the libraries you'll need).

```
pyenv install 3.11.9
pyenv local 3.11.9
```

The second command creates a file called `.python-version` in your project folder. This makes pyenv automatically use 3.11.9 whenever you're in this folder.

To verify it worked:

```
python --version
```

It should print `Python 3.11.9`.

---

### Step 1.3 — Create a virtual environment

A virtual environment is a contained Python install just for this project. You install packages inside it and they don't affect anything else on your machine.

```
python -m venv .venv
```

This creates a hidden folder called `.venv` inside your project. Now activate it:

```
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)` at the start. Every time you open a new terminal and work on this project, you need to run this activate command first.

---

### Step 1.4 — Create a requirements file

Create a file called `requirements.txt` in your project root. Leave it empty for now — you'll add packages to it as you go through each phase. This file will be the running list of everything the project needs.

---

### Step 1.5 — Set up Git

```
git init
```

Create a file called `.gitignore` in your project root and put this inside it:

```
.venv/
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
data/raw_logs/
data/labeled/
models/
*.ckpt
*.pt
.python-version
```

Then:

```
git add .
git commit -m "initial project structure"
```

You don't need GitHub for this. Git is just local version control so you can go back if you break something.

---

### Step 1.6 — Create a .env file

Create a file called `.env` in your project root. This is where you'll put configuration values (database URLs, ports, etc.) so they're not hardcoded into your code. Leave it empty for now.

**Important:** `.env` is in your `.gitignore` already. Never put secrets or API keys in any file that isn't gitignored.

---

**Phase 1 is done.** You have a folder, a Python environment, and version control.

---

---

# PHASE 2 — The Classifier

**What this phase is:** Building the PyTorch neural network that reads a user's message and outputs what topic it's about, what the user's intent is, and whether the system needs to search memory.

This is the most ML-heavy part of the whole project. It has four stages:

- 2A: Collect and prepare data
- 2B: Label the data using a big model
- 2C: Train the small classifier
- 2D: Test and save it

**What you'll learn:** PyTorch basics, sentence embeddings, multi-label classification, training loops.

**Where to learn (read these first, in this order):**

1. What a neural network is (3Blue1Brown on YouTube — "Neural Networks" playlist, first 3 videos)
2. PyTorch basics: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
3. What sentence embeddings are: https://www.sbert.net/docs/quickstart.html
4. Multi-label classification explained: https://towardsdatascience.com/multi-label-text-classification-with-pytorch-7cd92a69d3d0

---

## PHASE 2A — Data Collection and Preparation

**What you're doing:** Building a dataset of 20,000 prompts from four sources. You already have some personal data started. This phase describes all four sources and how to get them ready.

---

### Step 2A.1 — Install the packages you need for data work

Make sure your `.venv` is active, then:

```
pip install pandas datasets tqdm
```

Add these to your `requirements.txt`:

```
pandas
datasets
tqdm
```

`pandas` is how you work with tables of data in Python. `datasets` is Hugging Face's library for downloading public datasets. `tqdm` shows progress bars.

**Where to learn pandas:** https://pandas.pydata.org/docs/getting_started/10min.html (read this before continuing)

---

### Step 2A.2 — Personal data — extract your prompts (the Amnesia Method)

Your personal chat logs are in files where your messages and the AI's responses are all mixed together. The architecture calls the extraction strategy the "Amnesia Method" — you process small chunks one at a time, closing and reopening the model connection between chunks so it has no memory of previous chunks.

First, install Ollama's Python client and sentence transformers (you'll need them later anyway):

```
pip install ollama
```

Add to `requirements.txt`:

```
ollama
```

Create the file `scripts/extract_personal_prompts.py`. Here is what this script needs to do (you'll write it or have an AI write it for you):

1. Read a raw chat log file from `data/raw_logs/`
2. Split it into 3,000-character chunks with 500-character overlap. Overlap means: chunk 1 covers characters 0–3000, chunk 2 covers characters 2500–5500, chunk 3 covers 5000–8000, etc. This prevents a message from being cut in half at a chunk boundary.
3. For each chunk, send it to your local Ollama model (use `qwen2.5:1.5b` — the small background model you'll have running) with this exact prompt: `"Identify all text written by the human user in this excerpt. Output ONLY a JSON list of strings. Each string is one complete human message. If nothing was written by a human user, output an empty list []."`
4. After each chunk, close and reopen the Ollama connection. This is the "amnesia" — the model has no memory of what it extracted before. This is intentional.
5. Parse the JSON response. Collect all extracted prompts into one big list.
6. After all chunks are done, put the list into a pandas DataFrame.
7. Drop duplicate rows where the prompt text is identical (use `df.drop_duplicates()`).
8. Save the result as `data/labeled/personal_prompts_raw.jsonl` — one prompt per line, as JSON.

You want about 5,000 unique personal prompts from this step. If you have fewer, that's fine — just note the actual number.

**Where to learn:** How to read files in Python: https://realpython.com/read-write-files-python/

---

### Step 2A.3 — Public data — WildChat

WildChat is a public dataset of real conversations with LLMs. You want 5,000 English human-authored turns from it.

Create the file `scripts/download_public_datasets.py`. This script needs to:

1. Use the `datasets` library to load WildChat. The Hugging Face dataset name is `allenai/WildChat-1M`.
2. Filter for English-language rows only (the dataset has a `language` column).
3. Extract the first human turn from each conversation (each conversation has a `conversation` field which is a list of message objects).
4. Take a random sample of 5,000 rows. Use `pandas.DataFrame.sample(n=5000, random_state=42)` — the `random_state=42` makes your sampling reproducible.
5. Save to `data/labeled/wildchat_prompts.jsonl`.

**Where to learn Hugging Face datasets:** https://huggingface.co/docs/datasets/quickstart

---

### Step 2A.4 — Public data — LMSYS Chatbot Arena

Same process as WildChat. The dataset name is `lmsys/lmsys-chat-1m`.

1. Load with `datasets`
2. Filter for English
3. Extract first human turn
4. Sample 5,000 rows with `random_state=42`
5. Save to `data/labeled/lmsys_prompts.jsonl`

---

### Step 2A.5 — Public data — ShareGPT

ShareGPT is trickier because the format varies. The dataset name is `anon8231489123/ShareGPT_Vicuna_unfiltered`.

1. Load with `datasets`
2. Each item has a `conversations` list. Each message in the list has a `from` field which is either `"human"` or `"gpt"`. Extract only messages where `from == "human"`.
3. Filter out very short prompts (less than 10 characters) and very long ones (more than 2,000 characters) — edge cases that hurt training.
4. Filter for English by checking if the text uses mostly ASCII characters (a simple proxy for English).
5. Sample 5,000 rows with `random_state=42`
6. Save to `data/labeled/sharegpt_prompts.jsonl`

---

### Step 2A.6 — Merge all four datasets

Create the file `scripts/merge_dataset.py`. This script:

1. Loads all four JSONL files
2. Combines them into one pandas DataFrame with two columns: `prompt` (the text) and `source` (which dataset it came from)
3. Deduplicates by content — if two prompts are byte-for-byte identical, keep only one
4. Shuffles the combined dataset with `random_state=42`
5. Saves to `data/labeled/prompts_unlabeled.jsonl`

You should end up with approximately 20,000 rows. If your personal dataset was smaller, it'll be slightly under that — that's fine.

---

## PHASE 2B — Labeling with a 70B Model

**What you're doing:** Running every prompt through a large locally-running model to get labels for it. Each prompt gets: a list of active topic labels, a list of active intent labels, and one context reliance class.

This is the most compute-intensive step in the whole project. It will take several hours. Your RTX 4070 is capable of running a quantized 70B model (slowly) or a 32B model at good speed. The architecture specifies a 70B model for best label quality, but a Qwen2.5-32B-Instruct Q4 will also work and run faster.

**Where to learn:** What is chain-of-thought prompting: https://learnprompting.org/docs/intermediate/chain_of_thought

---

### Step 2B.1 — Pull the labeling model in Ollama

You already have Ollama. In a terminal:

```
ollama pull qwen2.5:32b-instruct-q4_K_M
```

If you want to use 70B and have the patience (or a second machine), use:

```
ollama pull qwen2.5:72b-instruct-q4_K_M
```

Wait for it to download. This is a large file (20–40GB depending on which you chose).

---

### Step 2B.2 — Understand the labels

Before writing the labeling script, you need to deeply understand what each label means. Here are all 25 labels:

**Topic Labels (11 total) — what subject is this about:**

- `Software_&_Tech` — code, debugging, dev tools, Linux, AI/ML
- `STEM_&_Academics` — math, science, research, studying
- `Business_&_Finance` — money, startups, work, management
- `Creative_&_Media` — writing, art, music, stories, games
- `Admin_&_Productivity` — scheduling, emails, organization, planning
- `Lifestyle_&_Health` — fitness, food, mental health, daily life
- `Social_&_Relationships` — people, emotions, communication, family
- `World_&_Current_Events` — news, geography, politics, history
- `Meta_AI` — asking about the AI itself, how prompting works
- `Null_Noise` — gibberish, tests, empty messages
- `General_Reference_&_Trivia` — random facts, definitions, general knowledge

**Intent Labels (11 total) — what does the user want to accomplish:**

- `Factual_Retrieval` — "what is X", "tell me about Y"
- `Troubleshooting` — "this is broken, fix it", "why doesn't this work"
- `Generation` — "write me X", "create Y", "generate Z"
- `Ideation` — "brainstorm ideas for", "what are some ways to"
- `Analysis_&_Summarization` — "summarize this", "analyze that"
- `Strategic_Planning` — "help me plan", "what's the best approach for"
- `Decision_Making` — "should I do X or Y", "which is better"
- `Emotional_Processing` — venting, expressing feelings, seeking empathy
- `Utility_Formatting` — "convert this to JSON", "format this as a table"
- `Casual_Banter` — hello, thanks, jokes, small talk
- `Open_Exploration` — "I wonder about X", "let's think through Y together"

**Context Reliance Classes (exactly one per prompt — pick the best one):**

- `Zero_Shot` — the AI can answer from its training alone, no memory needed
- `Long_Term_Memory` — answering well requires searching past conversations
- `Real_Time_Search` — answering well requires real-time information (news, prices, current events)

A prompt can have multiple active topics and multiple active intents, but exactly one context reliance class.

---

### Step 2B.3 — Write the labeling script

Create the file `scripts/label_dataset.py`. This is the most important script in Phase 2. Here is exactly what it needs to do:

1. Load `data/labeled/prompts_unlabeled.jsonl` into a pandas DataFrame
2. Add new columns for the labels: `topic_labels`, `intent_labels`, `context_reliance` — start them all as None
3. Check if a `data/labeled/prompts_labeled_partial.jsonl` file already exists. If it does, load it and skip rows that already have labels. This is your checkpoint/resume system — if the labeling crashes halfway through, you don't start over.
4. For each unlabeled prompt, send it to Ollama with this exact system prompt:

```
You are a classification expert. You will be given a user prompt and must classify it.

Respond ONLY with a valid JSON object. No explanation. No preamble. No markdown. Just the JSON.

The JSON must have exactly these three keys:
- "reasoning": a one-sentence explanation of your classification
- "topic_labels": a list of active topic labels from the allowed set
- "intent_labels": a list of active intent labels from the allowed set
- "context_reliance": exactly one string from the allowed set

TOPIC LABELS (choose all that apply):
Software_&_Tech, STEM_&_Academics, Business_&_Finance, Creative_&_Media, Admin_&_Productivity, Lifestyle_&_Health, Social_&_Relationships, World_&_Current_Events, Meta_AI, Null_Noise, General_Reference_&_Trivia

INTENT LABELS (choose all that apply):
Factual_Retrieval, Troubleshooting, Generation, Ideation, Analysis_&_Summarization, Strategic_Planning, Decision_Making, Emotional_Processing, Utility_Formatting, Casual_Banter, Open_Exploration

CONTEXT RELIANCE (choose exactly one):
Zero_Shot, Long_Term_Memory, Real_Time_Search
```

And send the user's prompt as the user message. Set temperature to 0.0 (deterministic — same prompt always gets same output).

5. Parse the JSON response. If it's malformed JSON, retry up to 3 times. If still failing after 3 retries, mark the row as `label_error=True` and skip it.
6. After every 100 rows, save the current progress to `data/labeled/prompts_labeled_partial.jsonl`. This is your checkpoint.
7. After all rows are done, save the final result to `data/labeled/prompts_labeled.jsonl`.

Run it. Monitor it. It will take a long time. Come back and check on it periodically.

---

### Step 2B.4 — Validate the labels

After labeling is done, create `scripts/validate_labels.py`. This script:

1. Loads `data/labeled/prompts_labeled.jsonl`
2. Checks: how many rows have `label_error=True`? Print this count.
3. Checks: does every row have exactly one `context_reliance` value? Print any rows that have zero or more than one.
4. Checks: are all label strings valid? Any label that isn't in your known list gets flagged.
5. Prints a summary: how many prompts have each topic label, how many have each intent label, how many have each context reliance class.

Fix any systematic errors you find before moving on to training.

---

### Step 2B.5 – Convert labels to vectors

**Create `scripts/training/build_training_data.py`.**

1. **Label order (immutable)** – define these constants exactly:

```python
TOPIC_LABELS = [
    "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance",
    "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health",
    "Social_&_Relationships", "World_&_Current_Events", "Meta_AI",
    "Null_Noise", "General_Reference_&_Trivia"
]

INTENT_LABELS = [
    "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation",
    "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making",
    "Emotional_Processing", "Utility_Formatting", "Casual_Banter",
    "Open_Exploration"
]

CONTEXT_RELIANCE_LABELS = [
    "Zero_Shot", "Long_Term_Memory", "Real_Time_Search"
]
```

2. **Input** – read `data/labeled/labeled_prompts.jsonl` (the final merged file of real + validated synthetic prompts).  
   **Filter** – **skip any row where `topic_labels` is empty OR `intent_labels` is empty**. These are the orphan prompts (224 no‑topic, 151 no‑intent). Dropping them here ensures they never reach training.

3. **Create label vector** (list of 25 floats):  
   - Positions 0–10: 1.0 if that topic label is active in the row’s `label.topic_labels`, else 0.0  
   - Positions 11–21: 1.0 if that intent label is active in the row’s `label.intent_labels`, else 0.0  
   - Positions 22–24: one‑hot encoding of the row’s `label.context_reliance` (exactly one position is 1.0, the others 0.0)

4. **Output** – write `data/labeled/training_data.jsonl` with one JSON object per line:  
   `{"prompt": "...", "labels": [list of 25 floats], "source": "..."}`

5. **Schema file** – write `data/labeled/label_schema.json` containing the three lists (topic, intent, context_reliance) in the exact order above. This file is the permanent record.

---

### Step 2C.1 – Install the ML packages

Do **not** use `pip`. Use `uv`:

```bash
uv add torch torchvision sentence-transformers scikit-learn
```

If any package is already in `pyproject.toml`, `uv` will just update the lock file. No manual `requirements.txt` changes needed.

---

### Step 2C.2 – Create the dataset class

**Create `src/classifier/dataset.py`.**

- Import `torch`, `torch.utils.data.Dataset`, `json`, `sentence_transformers`.
- Class `ICEClassifierDataset(Dataset)`:
  - `__init__(self, training_data_path)`:  
    - Load `training_data.jsonl` (the file produced by Step 2B.5).  
    - Load the `all-MiniLM-L6-v2` sentence‑transformer model: `SentenceTransformer("all-MiniLM-L6-v2")`.  
    - Encode every prompt **once** into a 384‑dimensional embedding: `self.embeddings = model.encode(prompts, convert_to_tensor=True)`.  
    - Convert the list of label vectors into a float tensor of shape `(N, 25)`.
  - `__len__`: return `N`.
  - `__getitem__(idx)`: return tuple `(embedding_tensor[idx], label_tensor[idx])`.

**Important:** The encoder is frozen – you never call `.train()` on it. The embeddings are pre‑computed to avoid repeated inference.

---

### Step 2C.3 – Create the model architecture

**Create `src/classifier/model.py`.**

- Imports: `torch`, `torch.nn`
- Class `ICEClassifier(nn.Module)`:
  - `__init__(self)`:  
    ```python
    self.fc1 = nn.Linear(384, 128)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.3)
    self.fc2 = nn.Linear(128, 25)
    ```
  - `forward(self, x)`: `x → fc1 → relu → dropout → fc2`  
    Return the raw logits (25 numbers, no activation).

That’s the whole model. It will be ~ 5 MB on disk.

---

### Step 2C.4 – Write the training script

**Create `scripts/training/train_classifier.py`.**

It must accept command‑line arguments:
- `--seed` (int, required)
- `--epochs` (int, default 30)
- `--batch_size` (int, default 32)
- `--lr` (float, default 0.001)
- `--val_split` (float, default 0.1)
- `--pos-weight-cap` (float, default 10.0) – how high the computed `pos_weight` values can go.

**Logic:**

1. **Set all seeds**: `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.

2. **Load dataset** `ICEClassifierDataset("data/labeled/training_data.jsonl")`.

3. **Train/val split**: `torch.utils.data.random_split(dataset, [1-val_split, val_split])`, using `torch.Generator().manual_seed(args.seed)` to keep it reproducible.

4. **DataLoaders**: `train_loader = DataLoader(..., batch_size=args.batch_size, shuffle=True)`, `val_loader = DataLoader(..., shuffle=False)`.

5. **Device**: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Move model there.

6. **Compute pos_weight for BCE losses**:  
   - Gather all label tensors from the **training set only**.  
   - For each label index (0–21), count positives (1.0).  
   - `pos_weight = (num_negatives) / (num_positives)` if positives > 0, else 1.0.  
   - Cap the weight at `args.pos_weight_cap`.  
   - Create a tensor of shape `(22,)`.  
   - Split into `topic_pos_weight = pos_weight[:11]` and `intent_pos_weight = pos_weight[11:22]`.

7. **Losses**:  
   - `bce_topic = nn.BCEWithLogitsLoss(pos_weight=topic_pos_weight.to(device))`  
   - `bce_intent = nn.BCEWithLogitsLoss(pos_weight=intent_pos_weight.to(device))`  
   - `ce_context = nn.CrossEntropyLoss()`

8. **Optimizer**: `torch.optim.Adam(model.parameters(), lr=args.lr)`

9. **Training loop**:  
   - For each epoch:  
     - `model.train()`  
     - For each batch:  
       - `emb, labels = batch[0].to(device), batch[1].to(device)`  
       - `outputs = model(emb)`  
       - `topic_out = outputs[:, :11]`, `intent_out = outputs[:, 11:22]`, `ctx_out = outputs[:, 22:]`  
       - `topic_gt = labels[:, :11]`, `intent_gt = labels[:, 11:22]`, `ctx_gt = labels[:, 22:].argmax(dim=1)`  
       - `loss = bce_topic(topic_out, topic_gt) + bce_intent(intent_out, intent_gt) + ce_context(ctx_out, ctx_gt)`  
       - `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`  
     - After epoch, compute average validation loss (no grad). Print epoch number, train loss, val loss.

10. **Early stopping**: keep track of best validation loss. If it doesn’t improve for 5 consecutive epochs, stop.

11. **Save model**: `torch.save(model.state_dict(), "models/classifier/ice_classifier.pt")`. Create directory if needed.

12. **Log the run**: append a JSON line to `models/classifier/training_runs.jsonl`:
    ```json
    {
      "seed": 42,
      "epochs_completed": 15,
      "final_val_loss": 0.234,
      "timestamp": "2026-06-05T18:00:00",
      "model_path": "models/classifier/ice_classifier.pt",
      "args": { "batch_size": 32, "lr": 0.001, "val_split": 0.1, "pos_weight_cap": 10.0 }
    }
    ```

Run it with:
```bash
uv run scripts/training/train_classifier.py --seed 42
```

---

### Step 2C.5 – Write the inference wrapper

**Create `src/classifier/classifier.py`.**

1. **Abstract base class** `IntentClassifier`:
   ```python
   from abc import ABC, abstractmethod
   class IntentClassifier(ABC):
       @abstractmethod
       def classify(self, prompt: str) -> ClassificationResult:
           ...
   ```

2. **Dataclass** `ClassificationResult`:
   ```python
   from dataclasses import dataclass
   @dataclass
   class ClassificationResult:
       topic_tags: list[str]
       intent_tags: list[str]
       context_reliance: str
       raw_probs: list[float]        # 25 probabilities
       max_confidence: float
   ```

3. **Concrete class** `PyTorchClassifier(IntentClassifier)`:
   - `__init__(self, model_path="models/classifier/ice_classifier.pt", schema_path="data/labeled/label_schema.json")`:  
     - Load label schema.  
     - Load the trained model architecture (`ICEClassifier`) and state dict; set to `eval()` mode, move to CPU.  
     - Load the `all-MiniLM-L6-v2` sentence‑transformer (CPU).
   - `classify(self, prompt: str)`:  
     - Encode the prompt to a 384‑dim tensor.  
     - `with torch.no_grad(): outputs = self.model(embedding)`  
     - Slice outputs: `topic_logits = outputs[:11]`, `intent_logits = outputs[11:22]`, `ctx_logits = outputs[22:]`  
     - `topic_probs = torch.sigmoid(topic_logits)` → list of floats  
     - `intent_probs = torch.sigmoid(intent_logits)` → list of floats  
     - `ctx_probs = torch.softmax(ctx_logits, dim=0)` → list of 3 floats  
     - Build `topic_tags` where probability > 0.5, using the schema’s topic list.  
     - Build `intent_tags` where probability > 0.5, using the schema’s intent list.  
     - `context_reliance` = the schema’s context class with the highest softmax probability.  
     - `max_confidence = max(all 25 probabilities)`.  
     - Return `ClassificationResult`.

---

### Step 2C.6 – Test the classifier manually

Create scripts/training/test_classifier.py

This script should:

    Instantiate PyTorchClassifier() (the concrete class from src/classifier/classifier.py).

    Define a list of test prompts (provided below).

    For each prompt, call classifier.classify(prompt) and print:

        The prompt

        Predicted topics, intents, context_reliance, max confidence.

    Print a summary of how many prompts got which context_reliance (to quickly spot if the model is still biased toward Zero_Shot).

The test prompts – copy this list directly into the script
python

TEST_PROMPTS = [
    # ===== Long_Term_Memory signals =====
    "that bug we fixed last night is back, the authentication module still crashes",
    "u said the goo blade would dissolve before touching him, can we keep that detail",
    "continue from where we left off, i want the next scene to start right after the hug",
    "what was the name of my friend who helped with the dipex project? i keep forgetting",
    "remember when we decided to use postgres instead of sqlite? i need to explain why",
    "my character Kael, the one with the fire ability, what should his weakness be",
    "the plan we made for the newsletter launch, can u remind me the first step",
    "we talked about adding a reflection worker, can u write the skeleton for it",
    "this is the same issue i had yesterday with docker, the container keeps exiting",
    "make the dialogue feel like what we discussed, more natural and less formal",
    # ===== Real_Time_Search signals =====
    "what’s the current price of bitcoin? i need to decide whether to sell",
    "is python 3.13 out yet or still in rc? about to set up a new venv",
    "who won the cricket match today between india and australia?",
    "what’s the weather forecast for tomorrow in mumbai, i have a flight",
    "did the supreme court issue any rulings this week on privacy?",
    "are there any new releases of vllm today? i thought they said june 5",
    "what’s the latest on that flooding in bangkok? is the airport open",
    # ===== Ambiguous / implicit references (should be LTM) =====
    "will it work this time or do i need to change the whole approach",
    "i think this is the right way to go, what do u think",
    "can u make it shorter but keep the same meaning",
    "what do you think about the ending we wrote last month",
    "i tried that thing u suggested, didn't help at all",
    # ===== Rare classes =====
    "asdfghjkl",                     # Null_Noise
    "zzzzzzzz",                      # Null_Noise
    "test",                          # Null_Noise (but be careful, might be mislabeled)
    "do you actually remember me or do you just pretend",   # Meta_AI + Factual_Retrieval
    "what’s the best way to prompt you to get really creative answers", # Meta_AI + Factual_Retrieval
    "i wonder if a sentient AI would experience time differently than us", # Open_Exploration
    "what would happen if we could store entire lifetimes in memory", # Open_Exploration
    "haha that joke was terrible lmaoo",                # Casual_Banter
    "thanks a lot for helping with that, ur a lifesaver", # Casual_Banter
    "yo morning",                                       # Casual_Banter
    # ===== Multi-label complex prompts =====
    "fix the login bug and then write a summary of the changes for the team",
    "i need a plan to learn rust in 2 weeks, and also i'm feeling overwhelmed by my current project",
    "should i use mongodb or postgres for this social app, and can u write the schema for whichever u recommend",
    # ===== Edge cases =====
    "write a haiku about a frog",     # Zero_Shot, Creative_&_Media, Generation
    "translate 'hello' to japanese", # Zero_Shot, General_Reference_&_Trivia, Utility_Formatting
    "what is the capital of france", # Zero_Shot, General_Reference_&_Trivia, Factual_Retrieval (trivia, but casual enough)
]

Expected behavior

After training, run:
bash

uv run python scripts/training/test_classifier.py

    Prompts with clear memory signals (“that bug we fixed last night”, “u said”, “continue from”) should yield Long_Term_Memory.

    Prompts requiring live data (“current price”, “today”, “this week”) should yield Real_Time_Search.

    Gibberish should yield Null_Noise.

    Meta questions should yield Meta_AI.

    The model should not blindly assign Zero_Shot to ambiguous personal‑source prompts like “will it work this time”.

If the outputs match these expectations, the classifier is working. If it still heavily biases toward Zero_Shot, you may need to check the data or increase the pos_weight for LTM.

---

**Phase 2 is done.** You have a trained classifier that can label any prompt in ~5ms on CPU.

---

---


# PHASE 3 — The Database (Architecture‑Complete)

**What you’re building:** PostgreSQL 16 + pgvector + Redis, and the complete SQLAlchemy ORM schema for **all** ICE tables. This is the permanent unified storage layer; future phases add application code, not schema changes.

**Tooling reminder:** Use `uv` for Python packages. Docker for Postgres and Redis.

---

### Step 3.1 – Start the database and Redis with Docker

Create `docker/docker-compose.yml` with the following content:

```yaml
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: ice_postgres
    restart: unless-stopped
    environment:
        POSTGRES_USER: ${POSTGRES_USER}
        POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
        POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - ice_postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    container_name: ice_redis
    restart: unless-stopped
    ports:
      - "6379:6379"

volumes:
  ice_postgres_data:
```

Start them (from the project root):

```bash
cd docker
docker compose up -d
cd ..
```

Check they’re running:

```bash
docker compose ps
```

Add to your `.env` file:

```
DATABASE_URL=postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
REDIS_URL=redis://localhost:6379/0
POSTGRES_USER=ice
POSTGRES_PASSWORD=ice_local_dev
POSTGRES_DB=ice_db
```

### Step 3.1a – Enable the pgvector extension

The pgvector extension must be manually activated once on the database.
Run this command:

```bash
docker exec -i ice_postgres psql -U ice -d ice_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
---

### Step 3.2 – Install database packages

```bash
uv add sqlalchemy alembic "psycopg[binary]" python-dotenv pgvector
```

No `pip` or `requirements.txt` needed – `uv` updates `pyproject.toml` and the lock file.

---

### Step 3.3 – Initialize Alembic

```bash
uv run alembic init alembic
```

Edit `alembic.ini` – change the `sqlalchemy.url` line to:

```
sqlalchemy.url = postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db
```

Later you’ll wire Alembic to your models (step 3.5).

---

### Step 3.4 – Create all database models

Create the file `src/memory/models.py` with the following **complete** content.  
This defines every table from the architecture specification.

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Date, Text,
    ForeignKey, ARRAY, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    memory_scope_type = Column(Text, nullable=False, default="auto")  # none, auto, project, manual
    cluster_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    custom_filter = Column(Text, nullable=True)

    episodic_turns = relationship("EpisodicMemory", back_populates="conversation")


class EpisodicMemory(Base):
    __tablename__ = "episodic_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("context_clusters.id"), nullable=True)
    parent_message_id = Column(UUID(as_uuid=True), ForeignKey("episodic_memory.id"), nullable=True)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utcnow)
    topic_tags = Column(ARRAY(Text), default=[])
    intent_tags = Column(ARRAY(Text), default=[])
    context_reliance = Column(Text, nullable=False)
    entropy_score = Column(Float, nullable=True)
    lossless_flag = Column(Boolean, nullable=True)  # NULL = not yet evaluated
    raw_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=True)
    embedding = Column(Vector(384), nullable=True)
    decay_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    is_archived = Column(Boolean, default=False)
    is_bookmarked = Column(Boolean, default=False)
    decay_immune = Column(Boolean, default=False)
    idempotency_key = Column(Text, unique=True, nullable=False)

    conversation = relationship("Conversation", back_populates="episodic_turns")


class MemorySlot(Base):
    __tablename__ = "memory_slots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slot_name = Column(Text, nullable=False)  # one of the seven predefined names
    content = Column(Text, default="")
    token_count = Column(Integer, default=0)
    version = Column(Integer, default=1)
    last_updated = Column(DateTime(timezone=True), default=utcnow)
    updated_by = Column(Text, nullable=False)  # user | reflection_worker
    is_active = Column(Boolean, default=True)


class ContextCluster(Base):
    __tablename__ = "context_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class CodexEntity(Base):
    __tablename__ = "codex_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name = Column(Text, nullable=False, unique=True)
    aliases = Column(ARRAY(Text), default=[])
    tags = Column(ARRAY(Text), default=[])
    properties = Column(JSONB, default={})
    context_payload = Column(Text, default="")
    last_updated = Column(DateTime(timezone=True), default=utcnow)


class CodexEdge(Base):
    __tablename__ = "codex_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    target_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    relation = Column(Text, nullable=False)
    strength = Column(Float, default=1.0)
    source_batch = Column(UUID(as_uuid=True), nullable=False)
    confidence = Column(Text, default="pending")  # pending | active
    valid_from = Column(DateTime(timezone=True), default=utcnow)
    valid_until = Column(DateTime(timezone=True), nullable=True)  # NULL = currently true


class CodexEvent(Base):
    __tablename__ = "codex_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    event_type = Column(Text, nullable=False)  # edge_added, edge_expired, property_updated, etc.
    payload = Column(JSONB, default={})
    timestamp = Column(DateTime(timezone=True), default=utcnow)
    batch_source = Column(UUID(as_uuid=True), nullable=False)
    compacted = Column(Boolean, default=False)


class CodexSnapshot(Base):
    __tablename__ = "codex_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    snapshot_ts = Column(DateTime(timezone=True), default=utcnow)
    last_event_id = Column(UUID(as_uuid=True), nullable=False)
    full_state = Column(JSONB, default={})


class ProceduralMemory(Base):
    __tablename__ = "procedural_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_name = Column(Text, nullable=False)
    pattern_description = Column(Text, default="")
    topic_tags = Column(ARRAY(Text), default=[])
    trigger_conditions = Column(JSONB, default={})
    reinforcement_count = Column(Integer, default=1)
    confidence_score = Column(Float, default=0.0)
    first_observed = Column(DateTime(timezone=True), default=utcnow)
    last_observed = Column(DateTime(timezone=True), default=utcnow)
    is_active = Column(Boolean, default=True)
    source_batch_ids = Column(ARRAY(UUID(as_uuid=True)), default=[])
    embedding = Column(Vector(384), nullable=True)


class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text, nullable=False)
    file_type = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=utcnow)
    token_count = Column(Integer, default=0)


class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("rag_documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)


class SentinelRule(Base):
    __tablename__ = "sentinel_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    is_active = Column(Boolean, default=True)
    trigger_type = Column(Text, nullable=False)  # threshold, frequency, absence, contradiction, composite
    trigger_conditions = Column(JSONB, default={})
    action_type = Column(Text, nullable=False)   # notify, schedule_worker, create_review_item, log_event, propose_memory_update
    action_payload = Column(JSONB, default={})
    cooldown_seconds = Column(Integer, default=0)
    last_fired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class SentinelEvent(Base):
    __tablename__ = "sentinel_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("sentinel_rules.id"), nullable=False)
    fired_at = Column(DateTime(timezone=True), default=utcnow)
    trigger_state = Column(JSONB, default={})
    action_taken = Column(Text, nullable=False)


class SessionReplay(Base):
    __tablename__ = "session_replays"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    event_sequence = Column(ARRAY(JSONB), default=[])
    created_at = Column(DateTime(timezone=True), default=utcnow)


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    # The bulletproof version:
    session_date = Column(Date, default=utcnow)
    topics_covered = Column(ARRAY(Text), default=[])
    decisions_made = Column(Text, default="")
    unresolved_items = Column(Text, default="")
    entities_updated = Column(ARRAY(UUID(as_uuid=True)), default=[])
    patterns_observed = Column(ARRAY(UUID(as_uuid=True)), default=[])


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"

    key = Column(Text, primary_key=True)
    processed_at = Column(DateTime(timezone=True), default=utcnow)


class ColdStorage(Base):
    __tablename__ = "cold_storage"

    id = Column(UUID(as_uuid=True), primary_key=True)  # original episodic turn id
    archived_at = Column(DateTime(timezone=True), default=utcnow)
    raw_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=True)
    topic_tags = Column(ARRAY(Text), default=[])
    timestamp = Column(DateTime(timezone=True), nullable=False)


class CuratedLabel(Base):
    __tablename__ = "curated_labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    prompt = Column(Text, nullable=False)
    corrected_topic_labels = Column(ARRAY(Text), default=[])
    corrected_intent_labels = Column(ARRAY(Text), default=[])
    corrected_context_reliance = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
```

---

### Step 3.5 – Create and run the Alembic migration

Edit `alembic/env.py`:

1. Add this import at the top (adjust path if needed):
   ```python
   import pgvector.sqlalchemy
   from src.memory.models import Base
   ```
2. Set `target_metadata = Base.metadata` (it currently says `target_metadata = None`).

Now generate the migration:

```bash
uv run alembic revision --autogenerate -m "initial schema"
```

A file will appear in `alembic/versions/`. Review it quickly – it should contain `CREATE TABLE` for all 18 tables.

Apply the migration:

```bash
uv run alembic upgrade head
```

Verify the tables exist:

```bash
docker exec -it ice_postgres psql -U ice -d ice_db
```

Inside psql, run:

```sql
\dt
```

You should see 18 tables listed. Type `\q` to exit.

---

### Step 3.6 – Create pgvector indexes

Create a script `scripts/create_indexes.sql` (or run the SQL directly).  
Here is the content – it creates the three vector indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_episodic_embedding
ON episodic_memory USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_procedural_embedding
ON procedural_memory USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding
ON rag_chunks USING hnsw (embedding vector_cosine_ops);
```

Execute the SQL by connecting to the database and running:

```bash
docker exec -i ice_postgres psql -U ice -d ice_db -f /dev/stdin < scripts/create_indexes.sql
```

Or manually:

```bash
docker exec -it ice_postgres psql -U ice -d ice_db
```

And then paste the three `CREATE INDEX` lines one by one.

---

**Phase 3 is complete.** You now have a fully initialized database with all tables and indexes, ready for the FastAPI proxy and all future subsystems.

---

# PHASE 4 — The FastAPI Proxy

**What this phase is:** Building the HTTP server that sits between Open WebUI and Ollama. It intercepts every chat message, runs the classifier, and (for now) just passes the request through to Ollama while storing the turn in episodic_memory. Full retrieval and prompt assembly comes later.

**What you'll learn:** FastAPI, async Python, OpenAI-compatible APIs, Server-Sent Events (SSE).

**Where to learn:**

- FastAPI from scratch: https://fastapi.tiangolo.com/tutorial/ (do the entire tutorial)
- What Server-Sent Events are: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- How the OpenAI API format works: https://platform.openai.com/docs/api-reference/chat/create (just read the request/response shape)

---

### Step 4.1 — Install FastAPI and related packages

```
pip install fastapi uvicorn httpx sse-starlette structlog pydantic python-dotenv uuid
```

Add to `requirements.txt`:

```
fastapi
uvicorn
httpx
sse-starlette
structlog
pydantic
python-dotenv
```

- `fastapi` is the web framework
- `uvicorn` is the server that runs FastAPI
- `httpx` is for making async HTTP requests (to forward to Ollama)
- `sse-starlette` adds SSE streaming support to FastAPI
- `structlog` is for structured JSON logging

---

### Step 4.2 — Create the config module

Create `src/api/config.py`. This file reads your `.env` and exposes configuration values as a Python object.

It needs to read these values:

- `DATABASE_URL`
- `REDIS_URL`
- `OLLAMA_BASE_URL` (add `OLLAMA_BASE_URL=http://localhost:11434` to your `.env`)
- `CLASSIFIER_THRESHOLD` (add `CLASSIFIER_THRESHOLD=0.50` to your `.env`)
- `CONFIDENCE_FALLBACK_THRESHOLD` (add `CONFIDENCE_FALLBACK_THRESHOLD=0.75` to your `.env`)

Use `pydantic-settings` for this (add to requirements):

```
pip install pydantic-settings
```

**Where to learn:** https://docs.pydantic.dev/latest/concepts/pydantic_settings/

---

### Step 4.3 — Create the database session module

Create `src/api/db.py`. This module creates the SQLAlchemy database engine and gives each request its own database connection that gets cleaned up after the request finishes.

It needs:

- An `engine` created from `DATABASE_URL`
- A `SessionLocal` sessionmaker
- A `get_db()` async generator function (using FastAPI's dependency injection pattern) that yields a database session and closes it when done

**Where to learn:** https://fastapi.tiangolo.com/tutorial/sql-databases/

---

### Step 4.4 — Create the main FastAPI application

Create `src/api/main.py`. This is the entry point.

It needs:

1. A FastAPI app instance
2. Startup event: initialize the classifier, log that the system is ready
3. A health check endpoint: `GET /health` returns `{"status": "ok"}`
4. The main chat endpoint: `POST /v1/chat/completions` (this path matches the OpenAI API format, which is what Open WebUI uses)

---

### Step 4.5 — Write the chat endpoint

This is the heart of Phase 4. In `src/api/main.py`, the `POST /v1/chat/completions` endpoint needs to:

1. Accept the request body in OpenAI format: a JSON object with `messages` (list of message objects with `role` and `content`) and `model` (string), and `stream` (boolean)
    
2. Generate a `correlation_id` using `uuid.uuid4()` — attach this to all log lines for this request
    
3. Extract the last user message (the current prompt) from the messages list
    
4. Run the classifier: `result = classifier.classify(prompt)`
    
5. Log the classification result with structlog
    
6. For now, just forward the entire request to Ollama's API at `http://localhost:11434/v1/chat/completions` using httpx
    
7. Stream the response back to Open WebUI as SSE tokens
    
8. After streaming completes, store the turn in `episodic_memory` with:
    
    - Generate a new UUID for id
    - Find or create a conversation_id (for now, extract from a header or generate one)
    - topic_tags = result.topic_tags
    - intent_tags = result.intent_tags
    - context_reliance = result.context_reliance
    - raw_text = the full exchange (user message + assistant response)
    - embedding = encode the user message with sentence transformer
    - idempotency_key = a SHA256 hash of correlation_id + prompt

**The SSE streaming part is the hardest piece.** The pattern is:

```python
async def generate():
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", ollama_url, json=request_body) as response:
            async for chunk in response.aiter_text():
                yield chunk

return StreamingResponse(generate(), media_type="text/event-stream")
```

Read the SSE documentation linked above before attempting this.

---

### Step 4.6 — Run the proxy

```
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag restarts the server automatically when you change a file.

Open your browser to `http://localhost:8000/health`. You should see `{"status": "ok"}`.

---

### Step 4.7 — Point Open WebUI at the proxy

Open WebUI needs to be told to use your proxy instead of Ollama directly.

In Open WebUI settings, find the "Connections" or "API" section. Change the Ollama API URL from `http://localhost:11434` to `http://localhost:8000`.

Now send a test message in Open WebUI. Watch your terminal — you should see log output from your FastAPI server showing the request being received, classified, and forwarded.

Check your database: `docker exec -it ice_postgres psql -U ice -d ice_db -c "SELECT id, topic_tags, intent_tags, context_reliance FROM episodic_memory LIMIT 5;"`

You should see rows being written.

---

**Phase 4 is done.** ICE is now live as a proxy. Every message flows through your classifier and gets stored in your database.

---

---

# PHASE 5 — Background Workers (Celery + Redis)

**What this phase is:** Setting up the asynchronous task system. After a conversation turn is stored, background workers process it — evaluating its value, extracting knowledge, etc. These workers run separately from the FastAPI server and don't block your responses.

**What you'll learn:** What a message queue is, how Celery works, what Redis is used for here.

**Where to learn:**

- What Celery is: https://docs.celeryq.dev/en/stable/getting-started/introduction.html
- Celery with FastAPI: https://testdriven.io/blog/fastapi-and-celery/

---

### Step 5.1 — Install Celery

```
pip install celery redis
```

Add to `requirements.txt`:

```
celery
redis
```

---

### Step 5.2 — Create the Celery app

Create `src/workers/celery_app.py`:

```python
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

app = Celery(
    "ice_workers",
    broker=os.getenv("REDIS_URL"),
    backend=os.getenv("REDIS_URL"),
    include=[
        "workers.post_flight",
        "workers.codex_extractor",
        "workers.procedural_extractor",
        "workers.reflection",
        "workers.decay",
        "workers.sentinel_monitor",
    ]
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"
```

---

### Step 5.3 — Create the GPU utilization check

Create `src/workers/gpu_check.py`. This module has one function: `is_gpu_busy() -> bool`.

It runs `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`, parses the output to get a number, and returns `True` if that number is above a configurable threshold (default: 20%).

Every background worker calls this before starting work. If the GPU is busy, the worker waits and retries.

**How to call nvidia-smi in Python:**

```python
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
utilization = int(result.stdout.strip())
```

---

### Step 5.4 — Create the Post-Flight Evaluator worker

Create `src/workers/post_flight.py`. This is the first real background worker.

It needs:

1. A Celery task decorated with `@app.task`
2. Input: `batch_id` (UUID string), `prompt` (str), `response` (str), `conversation_id` (UUID string)
3. Logic:
    - Check if GPU is busy — if yes, retry with a 30-second delay (`raise self.retry(countdown=30)`)
    - Check idempotency: look up `batch_id` in `idempotency_keys`. If found, return early (already processed).
    - Evaluate lossless_flag:
        - `lossless = True` if the response contains code blocks (look for ` ``` ` in the response text), OR contains 3+ proper nouns (capitalize words that look like names), OR is longer than 500 words
        - `lossless = False` otherwise
    - Generate `summary_text` if `lossless = False`:
        - Send the exchange to the 1.5B Ollama model with the prompt: "Summarize this conversation exchange in 2-3 sentences, focusing on the key information exchanged."
    - Update the `episodic_memory` row: set `lossless_flag`, set `summary_text`
    - Write `batch_id` to `idempotency_keys`
    - Emit a `BATCH_PROCESSED` event to Redis (this will trigger the Codex Extractor in Phase 8)

---

### Step 5.5 — Modify the FastAPI endpoint to enqueue work

Back in `src/api/main.py`, after storing the episodic turn, enqueue the post-flight task:

```python
from workers.post_flight import evaluate_turn

evaluate_turn.delay(
    batch_id=str(batch_id),
    prompt=user_message,
    response=assistant_response,
    conversation_id=str(conversation_id)
)
```

The `.delay()` call sends the task to Redis immediately and returns — it does not wait for the task to complete. The FastAPI response is already back with the user before the background work even starts.

---

### Step 5.6 — Run the Celery worker

Open a new terminal, activate your `.venv`, and run:

```
celery -A src.workers.celery_app worker --loglevel=info
```

You should see the worker start up and connect to Redis. When you send a message through Open WebUI now, you'll see the post-flight task appear in the Celery worker's logs.

---

**Phase 5 is done.** You now have async background processing. Messages are stored, then evaluated by a worker running separately.

---

---

# PHASE 6 — Codex Extractor

**What this phase is:** After each conversation turn is evaluated, extract knowledge from it and build the knowledge graph (the Codex).

**Where to learn:**

- What a knowledge graph is: https://www.ibm.com/topics/knowledge-graph
- What NER is: https://en.wikipedia.org/wiki/Named-entity_recognition

---

### Step 6.1 — Create the Codex Extractor worker

Create `src/workers/codex_extractor.py`. This Celery task runs after `BATCH_PROCESSED`.

It needs to:

1. GPU check — yield if busy
2. Idempotency check
3. Load the episodic turn from the database by batch_id
4. Only process turns where `lossless_flag = True` (high-value turns only — not worth extracting entities from low-value turns)
5. Send the `raw_text` to the 1.5B Ollama model with this prompt:
    
    ```
    Extract all named entities and relationships from this text as subject-relation-object triplets.Output ONLY a JSON array of objects, each with keys: "subject", "relation", "object".Each value must be a short noun phrase or verb phrase.If no clear entities or relationships exist, output an empty array [].Example: [{"subject": "ICE", "relation": "uses", "object": "PostgreSQL"}]
    ```
    
6. Parse the JSON response
7. For each triplet:
    - Look up `subject` in `codex_entities` by canonical_name or aliases. If not found, create a new entity.
    - Look up `object` the same way.
    - Create a new `codex_edge` with `confidence = "pending"` and `valid_until = NULL`.
    - Record a `codex_event` of type "edge_added".
8. For each entity, check if there's already an active edge with the same relation to the same target (a contradicting fact). If yes, set `valid_until = now()` on the old edge.
9. All writes happen inside a single database transaction (commit all or nothing).
10. Emit `ENTITY_UPDATED` events to Redis.

---

### Step 6.2 — Create the Compaction Worker

Create `src/workers/compaction.py`. This simpler worker runs periodically.

For each entity in `codex_entities` where the number of uncompacted `codex_events` exceeds a threshold (default: 50):

1. Load all uncompacted events for the entity
2. Reconstruct the entity's current state by applying them in order
3. Save a `codex_snapshot` with `full_state` = current state
4. Mark all processed events as `compacted = True`
5. Do this inside a transaction

This prevents the codex_events table from growing forever.

---

**Phase 6 is done.** The Codex is now being built from your conversations automatically.

---

---

# PHASE 7 — Retrieval Orchestrator

**What this phase is:** Instead of just passing prompts straight to Ollama, ICE now queries memory stores and injects relevant context into the prompt before sending it.

**What you'll learn:** What BM25 is, what cosine similarity is, what Reciprocal Rank Fusion is.

**Where to learn:**

- What cosine similarity is: https://en.wikipedia.org/wiki/Cosine_similarity (read the intro)
- What BM25 is: https://en.wikipedia.org/wiki/Okapi_BM25 (read the intro)
- What RRF is: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf (just read the abstract — the formula is what matters)

---

### Step 7.1 — Create the retrieval module

Create `src/retrieval/orchestrator.py`. This is the class that handles all retrieval.

It needs a main method: `retrieve(classification_result, conversation_id, prompt_embedding, scope) -> list[ContextFragment]`.

Where `ContextFragment` is a dataclass with: `text`, `source_type` (episodic/codex/procedural/rag), `score`, `token_count`, `source_batch_id`.

The method runs retrieval in three stages:

**Stage 1 — Check context reliance**

If `classification_result.context_reliance == "Zero_Shot"`, return an empty list immediately. No retrieval needed.

If it's `"Long_Term_Memory"`, proceed to Stage 2.

If it's `"Real_Time_Search"`, return a special marker telling the API to route to a web search — this is not yet implemented in V1, but the path needs to be there.

**Stage 2 — Run retrieval legs in parallel**

Execute these four queries:

_BM25 Episodic:_ Use PostgreSQL full-text search:

```sql
SELECT id, raw_text, summary_text, lossless_flag, ts_rank(to_tsvector('english', raw_text), query) as score
FROM episodic_memory, to_tsquery('english', :search_terms) query
WHERE to_tsvector('english', raw_text) @@ query
AND topic_tags && :topic_tags
AND is_archived = false
ORDER BY score DESC
LIMIT 10
```

_Vector Episodic:_ Use pgvector cosine similarity:

```sql
SELECT id, raw_text, summary_text, lossless_flag, 
       1 - (embedding <=> :prompt_embedding) as score
FROM episodic_memory
WHERE topic_tags && :topic_tags
AND is_archived = false
ORDER BY score DESC
LIMIT 10
```

_Codex:_ Extract entity names from the prompt using a simple NER pass (look for capitalized multi-word phrases), then look them up in codex_entities and traverse 1 hop outward along codex_edges.

_RAG:_ Only if the intent includes `Factual_Retrieval` or `Analysis_&_Summarization`:

```sql
SELECT chunk_text, 1 - (embedding <=> :prompt_embedding) as score
FROM rag_chunks
ORDER BY score DESC
LIMIT 5
```

**Stage 3 — Fuse with RRF**

Merge all results into one list using this formula for each result: `fused_score = sum of 1/(rank_in_list + 60) across all lists where this result appears`

Sort by fused_score descending. Remove duplicates (same text). Apply session diversification: no single conversation_id can contribute more than 3 results. Take the top results that fit within a token budget (default: 2000 tokens total across all retrieved context).

Return the final list as `ContextFragment` objects.

---

### Step 7.2 — Create the prompt assembler

Create `src/api/prompt_assembler.py`. This takes retrieved context fragments + active memory slots and assembles the final prompt payload.

The order must be:

1. `[SYSTEM RULES]` block (a hardcoded system prompt about ICE's behavior)
2. `[PERSISTENT CONTEXT]` block (active memory slots — always present)
3. `[CODEX: ABSOLUTE FACTS]` block (codex context fragments — only if non-empty)
4. `[EPISODIC CONTEXT]` block (episodic context fragments — only if non-empty)
5. `[USER INPUT]` (the actual user message)

For each episodic fragment: inject `raw_text` if `lossless_flag = True`, inject `summary_text` if `lossless_flag = False`.

The output is a list of message objects in OpenAI format, with the assembled context injected into the system message.

---

### Step 7.3 — Wire it into the API

Back in `src/api/main.py`, after classification but before forwarding to Ollama:

1. If context_reliance is `Long_Term_Memory`, call `retrieval_orchestrator.retrieve(...)`
2. Load active memory slots from the database
3. Call `prompt_assembler.assemble(memory_slots, retrieved_fragments, user_message)`
4. Send the assembled prompt to Ollama instead of the original

---

**Phase 7 is done.** ICE now retrieves relevant past context and injects it into every Long_Term_Memory classified prompt.

---

---

# PHASE 8 — Memory Slots

**What this phase is:** The seven persistent memory slots that are always injected into every prompt, every session.

**This is actually a short phase** — the database table already exists, and the prompt assembler already injects them. You just need the API endpoints to read and write them.

---

### Step 8.1 — Create the memory slots router

Create `src/api/routers/memory_slots.py`. This FastAPI router needs these endpoints:

- `GET /memory-slots` — return all active memory slots
- `GET /memory-slots/{slot_name}` — return one specific slot
- `PUT /memory-slots/{slot_name}` — update a slot's content (user writes)
- `POST /memory-slots/initialize` — create the default 7 slots with empty content if they don't exist

Include this router in `src/api/main.py`.

---

### Step 8.2 — Initialize default slots

Create `scripts/initialize_memory_slots.py`. This script inserts the 7 default empty slots if they don't already exist:

```
persona, user_preferences, tool_guidelines, project_context, 
guidance, pending_items, session_patterns
```

Run it once after the database is created.

---

**Phase 8 is done.** Memory slots are live and injectable.

---

---

# PHASE 9 — Remaining Systems

At this point you have a fully functional V1 of ICE. The remaining systems — Procedural Memory, Memory Decay, Sentinel Monitor, Reflection Worker, Drop Zone, and the Simulation Harness — are layered on top. They follow the same patterns you've already learned:

- Each is a Celery worker in `src/workers/`
- Each has a trigger (Redis event or scheduled timer)
- Each reads from and writes to the database using your existing models
- Each calls the 1.5B Ollama model for NLP tasks

---

### Step 9.1 — Procedural Extractor

Create `src/workers/procedural_extractor.py`. Triggered by `BATCH_PROCESSED` (same as Codex Extractor).

1. GPU check
2. Idempotency
3. Load the raw_text for the batch
4. Prompt the 1.5B model: "Does this exchange reveal a recurring workflow, decision pattern, or behavioral habit that the user consistently exhibits? If yes, describe it in one sentence. If no, output 'NONE'."
5. If the response is not "NONE":
    - Encode the pattern description as an embedding
    - Search `procedural_memory` for similar patterns by embedding similarity
    - If a match is found (similarity > 0.85): increment `reinforcement_count`, update `last_observed`
    - If no match: insert a new row with `reinforcement_count = 1`, `confidence = pending`
    - Promote to `is_active = True` when `reinforcement_count >= 3`

---

### Step 9.2 — Decay Worker

Create `src/workers/decay.py`. This is a Celery beat task (scheduled, not event-triggered) running daily.

1. GPU check
2. For all episodic turns older than 7 days where `decay_immune = False` and `is_bookmarked = False`:
    - Apply decay: `new_decay_score = old_decay_score * 0.97` (3% decay per day)
    - If `decay_score < 0.1`: set `is_archived = True`
3. For recently retrieved turns (accessed in the last 24 hours): add `+0.15` to decay_score (cap at 1.0)
4. Move rows with `is_archived = True` and `decay_score < 0.05` to the `cold_storage` table

---

### Step 9.3 — Reflection Worker

Create `src/workers/reflection.py`. Triggered by session end event or daily schedule.

1. GPU check
2. Load all episodic turns from the last session
3. Prompt the 1.5B model to produce a session summary: what was discussed, what was decided, what's unresolved
4. Write to `session_summaries`
5. Check `pending_items` memory slot — if the session produced unresolved items, append them
6. Propose updates to `project_context` slot if the session strongly indicated the user's active project changed

---

### Step 9.4 — Sentinel Monitor

Create `src/workers/sentinel_monitor.py`. Celery beat task, runs every 30 minutes.

1. GPU check
2. Load all active sentinel_rules
3. For each rule: evaluate its `trigger_conditions` against the current database state
4. For any rule that fires and is past its cooldown: execute the `action_type`
    - `notify`: add an entry to a notifications table
    - `log_event`: write to `sentinel_events`
    - `create_review_item`: write to a review_queue table
5. Update `last_fired_at` for fired rules

You should start with just the `log_event` action type and expand to others once the basic monitor is working.

---

### Step 9.5 — Drop Zone

Create `src/workers/drop_zone.py`. A file watcher that monitors the `/ingest_inbox` directory.

Use the `watchdog` library:

```
pip install watchdog
```

When a file appears in `ingest_inbox/`:

1. Detect file type (`.txt`, `.pdf`, `.jsonl`)
2. For `.txt` files: apply the Amnesia Method to extract human prompts (same as Phase 2A.2)
3. For each extracted chunk: run the classifier to get topic/intent tags
4. Compute content hash (SHA256) — skip if hash already exists in episodic_memory
5. Generate embedding and insert into `episodic_memory` or `rag_chunks` depending on file type
6. Move processed file to `ingest_inbox/processed/`

---

### Step 9.6 — Simulation Harness

Create `scripts/run_simulation.py`. This lets you replay historical conversations into a fresh database to test the whole system.

Input: a JSONL file of `(prompt, response, original_timestamp)` tuples.

For each tuple in order:

1. Write the turn to `episodic_memory` with a synthetic timestamp (preserve original spacing, but scale so months become hours)
2. Run the full post-flight pipeline on it
3. Wait briefly for workers to process before the next turn (configurable delay)

Accept `--seed` as a required argument. Log everything to `data/simulation_runs.jsonl`.

---

---

# Global Reference — What to Install Where

## Globally on your machine (already done or via pacman):

- Python (via pyenv)
- Docker + Docker Compose
- Git
- CUDA drivers
- Ollama
- Open WebUI

## In your Docker Compose (in `docker/docker-compose.yml`):

- PostgreSQL 16 with pgvector
- Redis 7

## In your Python virtual environment (via pip):

```
pandas datasets tqdm
ollama
torch torchvision
sentence-transformers
scikit-learn
fastapi uvicorn httpx sse-starlette
structlog pydantic pydantic-settings
python-dotenv
sqlalchemy alembic psycopg2-binary pgvector
celery redis
watchdog
```

---

# Cheat Sheet — Commands You'll Use Every Day

```bash
# Activate your venv (always do this first)
source .venv/bin/activate

# Start the database and Redis
cd docker && docker compose up -d && cd ..

# Run the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run the Celery worker
celery -A src.workers.celery_app worker --loglevel=info

# Run a database migration after changing models
alembic revision --autogenerate -m "description of change"
alembic upgrade head

# Check what's in the database
docker exec -it ice_postgres psql -U ice -d ice_db

# View Celery tasks in real time
celery -A src.workers.celery_app events

# Train the classifier
python scripts/train_classifier.py --seed 42

# Test the classifier
python scripts/test_classifier.py
```

---

# Build Order Summary (the one-liner version)

Phase 1 → folder + env + git  
Phase 2A → collect and clean data  
Phase 2B → label data with 70B model  
Phase 2C → train 5MB classifier  
Phase 2D → test classifier  
Phase 3 → PostgreSQL + all tables + indexes  
Phase 4 → FastAPI proxy intercepts Open WebUI → Ollama  
Phase 5 → Celery + Redis + post-flight worker  
Phase 6 → Codex Extractor worker + knowledge graph  
Phase 7 → Retrieval Orchestrator + prompt assembly  
Phase 8 → Memory Slots endpoints  
Phase 9 → Procedural, Decay, Reflection, Sentinel, Drop Zone, Simulation

**Minimum working ICE: Phases 1–4. Everything after is enhancement.**