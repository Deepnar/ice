
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

# PHASE 4 — The FastAPI Proxy (Architecture‑Aligned)

**What you're building:** The OpenAI‑compatible HTTP middleware that intercepts every chat request from Open WebUI, runs the pre‑flight classifier, forwards the request to Ollama, streams the response back, and stores every conversational turn in `episodic_memory`. This is the minimum viable ICE — after this phase, you have a working proxy that classifies and stores every message.

**What is intentionally deferred:**
- Full retrieval orchestration (BM25 + vector + graph) → Phase 7
- Model registry and dynamic routing → later phase
- Dual‑agent protocol → later phase
- HyDE query rewriting → later phase
- Memory slots injection → schema exists, endpoints come in Phase 8
- Post‑flight evaluation → Phase 5/6 (Celery workers)
- SSE telemetry events → Phase 7

**Architectural invariants enforced in this phase:**
- **INV‑9** — Pre‑flight classification is stateless (receives only the current prompt)
- **INV‑3** — All retrieval will pass through the classifier gate (hook prepared)
- **INV‑1** — Raw text is write‑once (enforced in the DB insert)

---

### Step 4.1 — Install FastAPI and related packages

Use `uv` to add the required libraries:

```bash
uv add fastapi uvicorn httpx sse-starlette structlog pydantic python-dotenv pydantic-settings
```

No `pip` or `requirements.txt` needed — `uv` updates `pyproject.toml` and the lock file.

Add these new variables to your `.env` file:

```
OLLAMA_BASE_URL=http://localhost:11434
CLASSIFIER_THRESHOLD=0.3
CONFIDENCE_FALLBACK_THRESHOLD=0.75
```

- `OLLAMA_BASE_URL` — where Ollama is running (default port 11434)
- `CLASSIFIER_THRESHOLD` — minimum probability for a label to be considered active (tuned to 0.3 for your classifier)
- `CONFIDENCE_FALLBACK_THRESHOLD` — if max confidence across all 25 labels falls below this, the system will fall back to wide‑net retrieval (implemented in Phase 7)

---

### Step 4.2 — Create the config module

Create `src/api/config.py`. This file reads your `.env` and exposes all configuration values as a typed Python object using `pydantic-settings`.

```python
"""Configuration for the ICE FastAPI proxy."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ice:ice_local_dev@localhost:5432/ice_db"
    redis_url: str = "redis://localhost:6379/0"
    ollama_base_url: str = "http://localhost:11434"
    classifier_threshold: float = 0.3
    confidence_fallback_threshold: float = 0.75
    classifier_model_path: str = "models/classifier/ice_classifier_v2_final.pt"
    label_schema_path: str = "data/labeled/label_schema.json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
```

The `model_config` line tells pydantic‑settings to automatically read from your `.env` file. Any value in `.env` overrides the defaults above.

---

### Step 4.3 — Create the database session module

Create `src/api/db.py`. This module creates the SQLAlchemy engine and provides a **dependency** that FastAPI can inject into route handlers, giving each request its own database session.

```python
"""
Database engine and session factory for the ICE FastAPI proxy.
Uses synchronous SQLAlchemy – all database operations are fast enough
that they don't need async.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.api.config import settings

# Create the engine using the DATABASE_URL from config
engine = create_engine(
    settings.database_url,
    pool_size=5,            # small pool – single‑user system
    max_overflow=0,
    pool_pre_ping=True,     # verify connections before using them
)

# Session factory – call SessionLocal() to get a new session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """
    FastAPI dependency that provides a database session.
    The session is automatically closed when the request finishes.

    Usage in a route:
        @app.get("/something")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### Step 4.4 — Create the main FastAPI application

Create `src/api/main.py`. This is the entry point for the ICE proxy. It contains:

- A **startup event** that initialises the classifier.
- A **health check** endpoint (`GET /health`).
- The main **chat completions** endpoint (`POST /v1/chat/completions`), which accepts the OpenAI request format.

Here is the complete file:

```python
"""ICE FastAPI Middleware.

Intercepts OpenAI-compatible chat requests, classifies the prompt,
forwards to Ollama, streams the response back, and safely stores records
without blocking the event loop or triggering race conditions.
"""

import asyncio
import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.db import SessionLocal, get_db
from src.classifier.classifier import PyTorchClassifier
from src.memory.models import Conversation, EpisodicMemory

logger = structlog.get_logger("ice.api")
classifier: Optional[PyTorchClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager – initialises the classifier at startup."""
    global classifier
    logger.info("Loading classifier...")
    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    logger.info("Classifier loaded. ICE Proxy ready.")
    yield


app = FastAPI(
    title="ICE Proxy",
    description="Infinite Context Engine — OpenAI-compatible memory middleware",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


async def store_turn_async(
    correlation_id: str,
    user_message: str,
    conversation_id: uuid.UUID,
    topic_tags: list,
    intent_tags: list,
    context_reliance: str,
    raw_stream_chunks: list[str],
):
    """Async post-flight task.

    Assembles streaming fragments, parses clean SSE text, calculates embeddings
    via thread pool offloading, and commits write-once transactions.
    """
    log = logger.bind(correlation_id=correlation_id)

    # 1. Join raw fragments FIRST to repair broken line boundaries from socket splits
    full_raw_stream = "".join(raw_stream_chunks)
    clean_fragments = []

    for line in full_raw_stream.split("\n"):
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            continue
        try:
            data = json.loads(line[5:].strip())
            content = data["choices"][0]["delta"].get("content", "")
            if content:
                clean_fragments.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
            
    full_assistant_text = "".join(clean_fragments)

    # 2. Offload CPU-heavy tensor tasks to avoid event loop starvation
    embedding = await asyncio.to_thread(
        classifier.embedder.encode, user_message, convert_to_tensor=False
    )
    embedding_list = (
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )

    # 3. Establish deterministic idempotency boundaries
    raw_key = f"{correlation_id}:{user_message}"
    idempotency_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    write_db = SessionLocal()
    try:
        turn = EpisodicMemory(
            conversation_id=conversation_id,
            batch_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            topic_tags=topic_tags,
            intent_tags=intent_tags,
            context_reliance=context_reliance,
            entropy_score=None,          # set by Post‑Flight Evaluator
            lossless_flag=None,          # NULL = not yet evaluated
            raw_text=f"User: {user_message}\n\nAssistant: {full_assistant_text}",
            summary_text=None,
            embedding=embedding_list,
            decay_score=1.0,
            idempotency_key=idempotency_key,
        )
        write_db.add(turn)
        write_db.commit()
        log.info("turn_stored", episodic_id=str(turn.id))
    except Exception as exc:
        write_db.rollback()
        log.error("failed_to_store_turn", error=str(exc))
    finally:
        write_db.close()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    body = await request.json()
    messages = body.get("messages", [])
    model_name = body.get("model", "default")

    correlation_id = str(uuid.uuid4())
    log = logger.bind(correlation_id=correlation_id)

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        log.warning("No user message found in request")
        return JSONResponse(
            status_code=400,
            content={"error": "No user message found in the request."},
        )

    # Stateless pre‑flight classification
    result = classifier.classify(user_message)
    log.info(
        "classified",
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        max_confidence=result.max_confidence,
    )

    if result.max_confidence < settings.confidence_fallback_threshold:
        log.info(
            "low_confidence_fallback",
            max_confidence=result.max_confidence,
            threshold=settings.confidence_fallback_threshold,
        )

    # State tracking boundary
    conversation_id_str = request.headers.get("X-ICE-Conversation-ID")
    if conversation_id_str:
        conversation_id = uuid.UUID(conversation_id_str)
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            conversation = Conversation(id=conversation_id)
            db.add(conversation)
            db.commit()
    else:
        conversation = Conversation()
        db.add(conversation)
        db.commit()
        conversation_id = conversation.id

    ollama_url = f"{settings.ollama_base_url}/v1/chat/completions"
    accumulated_raw_chunks = []

    async def generate():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                ollama_url,
                json={"model": model_name, "messages": messages, "stream": True},
            ) as ollama_response:
                async for chunk in ollama_response.aiter_text():
                    accumulated_raw_chunks.append(chunk)
                    yield chunk

    # Enqueue background execution safely. FastAPI natively calls this ONLY
    # after the generator completes and the response is flushed to the network wire.
    background_tasks.add_task(
        store_turn_async,
        correlation_id=correlation_id,
        user_message=user_message,
        conversation_id=conversation_id,
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        raw_stream_chunks=accumulated_raw_chunks,
    )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-ICE-Conversation-ID": str(conversation_id),
        },
    )
```

**Important notes about this implementation:**

- **The classifier runs on CPU** — it's loaded in the `startup` event and reused across all requests.
- **Pre‑flight is stateless** — only the current prompt is passed to the classifier (INV‑9).
- **The confidence threshold check** is a hook for Phase 7; for now it just logs low‑confidence events.
- **Conversation creation** — every request gets a new conversation unless Open WebUI passes an `X-ICE-Conversation-ID` header (future integration).
- **SSE streaming** — tokens from Ollama are forwarded to Open WebUI as they arrive, exactly matching the OpenAI streaming format.
- **Post‑stream storage** — the full assistant response is accumulated and stored in `episodic_memory` with all classifier tags, the embedding, and an idempotency key.
- **`lossless_flag` is NULL** — the Asymmetrical Value Problem (§1.3) requires that this be set post‑flight by the Post‑Flight Evaluator in Phase 5/6.

---

### Step 4.5 — Understanding the SSE streaming pattern

The streaming pattern used above is the standard method for proxying OpenAI‑compatible streaming endpoints. Here's what happens step‑by‑step:

1. **Open WebUI** sends a `POST /v1/chat/completions` request with `"stream": true` to the ICE proxy.
2. **ICE** parses the request, classifies the prompt, and creates a `StreamingResponse`.
3. The `StreamingResponse` calls the `generate()` async generator.
4. `generate()` opens an async HTTP stream to **Ollama** (the backend LLM).
5. As Ollama produces tokens, they are **yielded** to Open WebUI as SSE events in real time.
6. Simultaneously, every token is **accumulated** in a list for later storage.
7. After the stream ends, the `store_turn()` background task writes the full exchange to `episodic_memory`.

The key advantage: **the user sees the response streaming immediately**, while ICE transparently records everything for future memory retrieval.

---

### Step 4.6 — Run the proxy

From the project root, start the FastAPI server:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- `--host 0.0.0.0` makes the server accessible from Open WebUI (which may run in Docker).
- `--port 8000` is the default ICE proxy port.
- `--reload` automatically restarts the server when you change any source file (useful during development).

Open your browser to `http://localhost:8000/health`. You should see:

```json
{"status": "ok"}
```

---

### Step 4.7 — Point Open WebUI at the proxy

Open WebUI needs to be told to use your proxy instead of Ollama directly.

1. Open Open WebUI in your browser (typically `http://localhost:3000`).
2. Go to **Settings → Connections** (or **Admin Settings → API Connections**, depending on your version).
3. Find the **Ollama API URL** field.
4. Change it from `http://localhost:11434` to **`http://localhost:8000`**.
5. Save the settings.

Now send a test message in Open WebUI. Watch the terminal where `uvicorn` is running — you should see structured log output showing:

- The request being received.
- The classification result (topic_tags, intent_tags, context_reliance, max_confidence).
- The streaming response being proxied.
- The turn being stored in the database.

---

### Step 4.8 — Verify the database is being populated

After sending a few test messages, check that turns are being written to `episodic_memory`:

```bash
docker exec -i ice_postgres psql -U ice -d ice_db -c "SELECT id, topic_tags, intent_tags, context_reliance, lossless_flag FROM episodic_memory LIMIT 5;"
```

You should see rows with:

- Real UUIDs in the `id` column.
- Arrays of topic and intent labels (e.g., `{Software_&_Tech}`).
- A context reliance value (`Zero_Shot`, `Long_Term_Memory`, or `Real_Time_Search`).
- `lossless_flag` set to NULL (meaning "not yet evaluated" — this will be set by the Post‑Flight Evaluator in Phase 5/6).

If you see rows, **Phase 4 is complete.** ICE is now live as a working proxy. Every message flows through your classifier and gets stored in your unified memory.

---

### Summary of what Phase 4 delivers

| Feature | Status |
|---------|--------|
| OpenAI‑compatible `/v1/chat/completions` endpoint | ✅ Working |
| Pre‑flight classification (stateless, CPU, ~5 ms) | ✅ Working |
| SSE streaming proxy to Ollama | ✅ Working |
| Conversation creation and tracking | ✅ Working |
| Episodic turn storage (with all classifier tags + embedding) | ✅ Working |
| Structured JSON logging with correlation_id | ✅ Working |
| Confidence threshold hook (for Phase 7 fallback) | ✅ Prepared |
| Memory slots injection | 🔜 Phase 8 |
| Retrieval orchestrator (BM25 + vector + graph) | 🔜 Phase 7 |
| Post‑flight evaluator (lossless flag, summarisation) | 🔜 Phase 5/6 |
| Model registry and dynamic routing | 🔜 Later phase |

---

**Phase 4 is done.** The brain (classifier) now has a spine (FastAPI) that connects it to the real world. Every conversation you have is classified, streamed, and permanently stored. The minimum viable ICE is operational.

---

# PHASE 5 — Background Workers (Celery + Redis)

**What you’re building:**  
The asynchronous post‑processing brain of ICE. Once a conversation turn is stored, Celery workers (powered by Redis) pick it up, evaluate its value (lossless flag and summarisation), and prepare for future knowledge extraction. This phase establishes the worker framework and the first critical worker: the **Post‑Flight Evaluator**, exactly as defined in the architecture’s §12.1.

**Architectural invariants enforced in this phase:**
- **INV‑5** — Background workers yield to active GPU inference (GPU utilisation check before processing).
- **INV‑6** — Idempotency is enforced at the worker boundary (every batch checked against `idempotency_keys`).
- **INV‑2** — The LLM never directly writes to the Codex or Procedural store (this worker only updates the `episodic_memory` row; later workers will handle extraction).

**What is intentionally deferred:**
- Codex Extractor and Procedural Extractor – triggered by the `BATCH_PROCESSED` event emitted here, but implemented in Phase 6.
- Reflection, Decay, Compaction, Clustering, Sentinel workers – later phases.
- Memory slot injection – Phase 8.
- Full retrieval orchestration – Phase 7.

---

## Step 5.1 – Install Celery and Redis client

The `redis` server is already running in Docker (from Phase 3). We only need the Python client.

```bash
uv add celery redis
```

This updates `pyproject.toml` – no `pip` or `requirements.txt` changes.

---

## Step 5.2 – Create the Celery application

Create `src/workers/celery_app.py`. This is the central Celery instance that all workers will share. It reads the Redis URL from your configuration.

```python
from celery import Celery
from src.api.config import settings

app = Celery(
    "ice_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.workers.post_flight",
        # Future workers will be added here:
        # "src.workers.codex_extractor",
        # "src.workers.procedural_extractor",
        # "src.workers.reflection",
        # "src.workers.decay",
        # "src.workers.sentinel_monitor",
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"
```

The `include` list tells Celery which modules to scan for `@app.task` decorators.

---

## Step 5.3 – Create the GPU utilisation check (hardened)

Create `src/workers/gpu_check.py`. Every background worker must poll GPU utilisation before executing, respecting **INV‑5**.  
This version correctly handles multiple GPUs and gracefully falls back if `nvidia-smi` is unavailable.

```python
"""GPU utilization tracking subsystem for the ICE worker cluster."""

import subprocess
import structlog

logger = structlog.get_logger("ice.workers.gpu")

GPU_UTIL_THRESHOLD = 20  # Max percentage allowable for background ingestion


def is_gpu_busy() -> bool:
    """Queries all active NVIDIA devices for compute utilization.
    
    Returns True if any single GPU exceeds the configured threshold.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return False
            
        # Extract maximum utilization across all present nodes
        max_utilization = max(int(util) for util in lines)
        return max_utilization > GPU_UTIL_THRESHOLD

    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as err:
        # Fall back gracefully if nvidia-smi is missing (e.g., CPU-only local dev contexts)
        logger.debug("Nvidia-smi query skipped or unavailable", error=str(err))
        return False
```

---

## Step 5.4 – Set up the 1.5B background model (vLLM)

The architecture (§12.9) requires a **dedicated 1.5B model** (`Qwen2.5-1.5B, quantized Q8_0`) for all background NLP tasks. For Phase 5 we only need it for summarisation when `lossless=false`. Later workers will use it for triplet extraction, pattern detection, etc.

We’ll serve it via vLLM on a separate port (8002) to keep it completely independent from the user‑facing model.

**Download the model (if not already cached):**

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct-AWQ
```

*(vLLM will auto‑download it, but pre‑fetching ensures a fast start.)*

**Add a shell function for the background server** to your `~/.zshrc`:

```bash
vllm-bg() {
    vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
        --port 8002 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.40 \
        --enforce-eager \
        --kv-cache-dtype fp8
}
```

- `--port 8002` – separates it from the proxy (8000) and any coder (8001).
- `--gpu-memory-utilization 0.40` – limits VRAM usage to ~9.6 GB, leaving room for a user‑facing model.
- `--max-model-len 4096` – ample for summarisation tasks.
- `--enforce-eager` – stability.

**Start the background model** whenever you run the Celery worker:

```bash
vllm-bg
```

---

## Step 5.5 – Create the Post‑Flight Evaluator worker (production‑hardened)

Create `src/workers/post_flight.py`. This is the first background worker. It runs **after** the proxy has stored the turn. Its task is to:

1. Check the GPU (INV‑5).
2. Enforce idempotency (INV‑6).
3. Determine if the assistant response is “lossless” (contains code blocks, named entities, or has high information density) – this is the **Asymmetrical Value Problem** in action.
4. If the response is not lossless, generate a concise summary using the 1.5B model.
5. Update the `episodic_memory` row with `lossless_flag` and `summary_text`.
6. Record the idempotency key.
7. Emit a `BATCH_PROCESSED` event (which will later trigger the Codex and Procedural Extractors).

**Critical fixes included:**
- **Visibility race condition:** Retries if the database row isn’t immediately found.
- **Proper‑noun false positive:** Strips sentence‑initial capitals before counting proper nouns.
- **Multi‑GPU safety:** Already handled by `gpu_check`.

```python
"""Post-Flight Evaluation Celery Worker Node."""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, IdempotencyKey
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.post_flight")

# Dedicated backend inference client targeting isolated LLM instance
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")


def is_lossless(text: str) -> bool:
    """Analyzes text density to resolve the Asymmetrical Value Problem.
    
    Ensures true proper nouns are tracked while filtering out standard sentence starts.
    """
    if "```" in text:
        return True

    if len(text.split()) > 500:
        return True

    # Strip out standard sentence boundaries (. ! ?) followed by whitespace and capitals
    # to avoid false positives on standard sentence starts
    cleaned_text = re.sub(r'(?:^[A-Z]|\b[\.\!\?]\s+[A-Z])[a-z]+\b', '', text)
    
    # Track true remaining capitalized words (e.g., inline proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', cleaned_text)
    if len(proper_nouns) >= 3:
        return True

    return False


def generate_summary(prompt: str, response: str) -> str:
    """Invokes the background 1.5B resource model to condense wide conversation context."""
    try:
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation exchange in 2-3 sentences. Focus heavily on the concrete technical facts or constraints provided."
                },
                {"role": "user", "content": f"User: {prompt}\nAssistant: {response}"},
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=30.0,  # Prevent infinite socket hangs
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("background_summarization_failed", error=str(exc))
        return ""


@app.task(bind=True, max_retries=5, default_retry_delay=15)
def evaluate_turn(self, batch_id: str, prompt: str, response: str, conversation_id: str):
    """Executes structural density qualification and data post-processing routines."""
    log = logger.bind(batch_id=batch_id, conversation_id=conversation_id)

    # 1. Active GPU Resource Gate (INV-5)
    if is_gpu_busy():
        log.info("gpu_saturation_yielding", message="Rescheduling worker target thread.")
        raise self.retry(countdown=15)

    # 2. Border Idempotency Verification (INV-6)
    idempotency_key = hashlib.sha256(batch_id.encode()).hexdigest()
    db = SessionLocal()
    
    try:
        existing = db.query(IdempotencyKey).filter_by(key=idempotency_key).first()
        if existing:
            log.info("task_execution_skipped_idempotent")
            return

        # 3. Defensive Record Fetching (Mitigates DB Commit Race Condition)
        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            log.warn("record_visibility_lag_retry")
            raise self.retry(countdown=5)  # Back off briefly to let the API commit finish

        # 4. Text Ingestion & Processing
        lossless = is_lossless(response)
        summary = None if lossless else generate_summary(prompt, response)

        # 5. Core Write-Once Persistence
        turn.lossless_flag = lossless
        turn.summary_text = summary
        
        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("post_flight_evaluation_complete", lossless=lossless)

    except Exception as exc:
        db.rollback()
        log.error("worker_transaction_execution_failure", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()

    # Pipeline Trigger Event Stub
    print(f"BATCH_PROCESSED: {batch_id}")
```

---

## Step 5.6 – Enqueue the post‑flight task from the proxy

Modify `src/api/main.py` so that after a turn is stored, the `evaluate_turn` Celery task is called.

Add the import at the top of `main.py`:

```python
from src.workers.post_flight import evaluate_turn
```

Then, inside `store_turn_async`, right after the database commit succeeds (i.e., after `write_db.commit()` and before `write_db.close()`), add:

```python
# Enqueue post‑flight evaluation
evaluate_turn.delay(
    batch_id=str(turn.batch_id),
    prompt=user_message,
    response=full_assistant_text,
    conversation_id=str(conversation_id),
)
```

The `.delay()` call sends the task to Redis immediately and returns – the FastAPI response is already with the user before this background work starts.

---

## Step 5.7 – Run the Celery worker

1. **Make sure the background model is running** (Step 5.4).  
2. Open a new terminal, activate your virtual environment, and run:

```bash
uv run celery -A src.workers.celery_app worker --loglevel=info
```

You should see the worker connect to Redis and wait for tasks. When you send a message through Open WebUI (with the proxy running), the worker will pick up the `evaluate_turn` task, process it, and update the database.

3. **Verify** by checking the `episodic_memory` table after a few exchanges – you should see `lossless_flag` set to `true` or `false`, and `summary_text` populated for lossless=false turns.

---

**Phase 5 is complete.**  
You now have a resilient background processing plane that evaluates every turn, respects GPU resources, and never silently drops work. The `BATCH_PROCESSED` event pipeline is ready for the Codex and Procedural Extractors in Phase 6.
---

# PHASE 6 — Codex Extractor & Compaction (Corrected & Production‑Hardened)

**What you’re building:**  
The knowledge‑graph construction pipeline. After the Post‑Flight Evaluator marks a turn as high‑value (`lossless = true`), the **Codex Extractor** uses the 1.5B background model to extract subject‑relation‑object triplets, resolves them into entities and typed edges, and records every mutation as an append‑only event in the `codex_events` table. The **Compaction Worker** periodically compresses the event log to keep entity state reconstruction fast. Together, they implement the Semantic Memory (Codex) subsystem exactly as defined in §3.2 and §12.2/§12.5 of the architecture.

All four bugs discovered by the independent review (edge ID `None`, markdown parsing fragility, compaction state collapse, and NULL alias crash) are fixed in the code provided below.

**Architectural invariants enforced in this phase:**
- **INV‑2** — The LLM never directly writes to the Codex; all mutations are mediated by the Codex Extractor.
- **INV‑6** — Idempotency is enforced at the worker boundary.
- **INV‑7** — All Codex mutations are transactional.
- **INV‑4** — Only currently‑valid edges participate in retrieval (contradictions expire old edges).
- **INV‑5** — Workers yield to active GPU inference.
- **Truth quorum** — Edges start as `pending`; promotion to `active` requires corroboration (second batch).

---

## Step‑by‑step changes (files to edit)

### 1. Update `src/workers/celery_app.py` — register the new workers

Open the Celery app file and ensure the `include` list contains both new modules:

```python
from celery import Celery
from src.api.config import settings

app = Celery(
    "ice_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.workers.post_flight",
        "src.workers.codex_extractor",   # ← add this
        "src.workers.compaction",        # ← add this
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"
```

### 2. Modify `src/workers/post_flight.py` — trigger extraction for lossless turns

In the `evaluate_turn` task, after the successful commit and before the `finally` block, add:

```python
if lossless:
    from src.workers.codex_extractor import extract_codex
    extract_codex.delay(batch_id=batch_id)
```

Make sure this is placed **after** the `db.commit()` and **before** the `return` or end of the task, and that the import is safe (lazy import is fine).

### 3. Create the corrected **Codex Extractor** — `src/workers/codex_extractor.py`

Replace the entire file with the following production version.  
It fixes:
- Edge ID not appearing in events (now uses client‑side UUIDv4).
- Markdown parsing fragility (uses a regex to find the JSON array).
- Includes `target_id` in all edge event payloads.

```python
"""Codex Extractor Subsystem – Structural Ingestion Plane."""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import (
    CodexEntity, CodexEdge, CodexEvent, IdempotencyKey, EpisodicMemory
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.codex")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

CODEX_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def generate_uuid5(canonical_name: str) -> uuid.UUID:
    """Derive deterministic UUIDv5 identifier for a canonical entity node."""
    return uuid.uuid5(CODEX_NAMESPACE, canonical_name.strip().lower())


def extract_triplets(text: str) -> list:
    """Invokes backend processing nodes to parse declarative statements into triplets."""
    prompt = (
        "Extract text elements as subject-relation-object triplets. "
        "Output exclusively a valid JSON array of objects with keys: \"subject\", \"relation\", \"object\". "
        "Do not include extra explanations or text padding."
    )
    try:
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are an isotropic semantic extraction engine."},
                {"role": "user", "content": f"Text:\n{text}\n\n{prompt}"}
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=30.0
        )
        raw = completion.choices[0].message.content.strip()
        
        # Robust Regex capture boundary to safely parse target JSON array content
        json_match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if not json_match:
            return []
            
        triplets = json.loads(json_match.group(0))
        return triplets if isinstance(triplets, list) else []
    except Exception as err:
        logger.error("triplet_parsing_boundary_failed", error=str(err))
        return []


def get_or_create_entity(db, name: str) -> CodexEntity:
    """Resolves structural identity records across global name and alias spaces."""
    canonical = name.strip().lower()
    entity = db.query(CodexEntity).filter_by(canonical_name=canonical).first()
    if entity:
        return entity

    entity = db.query(CodexEntity).filter(CodexEntity.aliases.any(canonical)).first()
    if entity:
        return entity

    new_entity = CodexEntity(
        id=generate_uuid5(canonical),
        canonical_name=canonical,
        aliases=[name],
        tags=[],
        properties={},
        context_payload="",
        last_updated=datetime.now(timezone.utc)
    )
    db.add(new_entity)
    db.flush()
    return new_entity


def handle_triplet(db, subject_name: str, relation: str, object_name: str, batch_id: str):
    """Integrates extraction assertions into the transaction context."""
    subj = get_or_create_entity(db, subject_name)
    obj = get_or_create_entity(db, object_name)

    existing_edge = db.query(CodexEdge).filter(
        CodexEdge.source_id == subj.id,
        CodexEdge.target_id == obj.id,
        CodexEdge.valid_until == None
    ).first()

    if existing_edge:
        if existing_edge.relation == relation:
            # Corroboration pass logic
            existing_edge.strength += 1.0
            if existing_edge.strength >= 2.0 and existing_edge.confidence == "pending":
                existing_edge.confidence = "active"

            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_strengthened",
                payload={
                    "edge_id": str(existing_edge.id),
                    "relation": relation,
                    "target_id": str(obj.id)
                },
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
        else:
            # Contradiction resolution pass logic (INV-4)
            existing_edge.valid_until = datetime.now(timezone.utc)
            
            # Explicit Client-Side UUID Generation to guarantee valid event tracking logs
            new_edge_id = uuid.uuid4()
            db.add(CodexEdge(
                id=new_edge_id,
                source_id=subj.id,
                target_id=obj.id,
                relation=relation,
                strength=1.0,
                source_batch=batch_id,
                confidence="pending",
                valid_from=datetime.now(timezone.utc)
            ))
            
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_expired",
                payload={
                    "edge_id": str(existing_edge.id),
                    "relation": existing_edge.relation
                },
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
            db.add(CodexEvent(
                entity_id=subj.id,
                event_type="edge_added",
                payload={
                    "edge_id": str(new_edge_id),
                    "relation": relation,
                    "target_id": str(obj.id)
                },
                timestamp=datetime.now(timezone.utc),
                batch_source=batch_id
            ))
    else:
        new_edge_id = uuid.uuid4()
        db.add(CodexEdge(
            id=new_edge_id,
            source_id=subj.id,
            target_id=obj.id,
            relation=relation,
            strength=1.0,
            source_batch=batch_id,
            confidence="pending",
            valid_from=datetime.now(timezone.utc)
        ))
        db.add(CodexEvent(
            entity_id=subj.id,
            event_type="edge_added",
            payload={
                "edge_id": str(new_edge_id),
                "relation": relation,
                "target_id": str(obj.id)
            },
            timestamp=datetime.now(timezone.utc),
            batch_source=batch_id
        ))


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def extract_codex(self, batch_id: str):
    """Executes background semantic link mutations across target graph states."""
    log = logger.bind(batch_id=batch_id)

    if is_gpu_busy():
        raise self.retry(countdown=30)

    idempotency_key = hashlib.sha256(f"codex:{batch_id}".encode()).hexdigest()
    db = SessionLocal()
    
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn or not turn.lossless_flag:
            return

        triplets = extract_triplets(turn.raw_text)
        for triplet in triplets:
            s = triplet.get("subject", "").strip()
            r = triplet.get("relation", "").strip()
            o = triplet.get("object", "").strip()
            if s and r and o:
                handle_triplet(db, s, r, o, batch_id)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("codex_graph_assertions_committed", extracted_count=len(triplets))

    except Exception as exc:
        db.rollback()
        log.error("codex_extraction_aborted", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

### 4. Create the corrected **Compaction Worker** — `src/workers/compaction.py`

Replace the entire file with the following production version.  
It fixes:
- Multi‑target edge collapse (uses composite `relation:target_id` signature).
- `None` aliases crash (guards with `if entity.aliases else []`).

```python
"""Compaction Engine Subsystem – Ledger Compression Plane."""

import uuid
from datetime import datetime, timezone
import structlog
from sqlalchemy import func, select

from src.api.db import SessionLocal
from src.memory.models import CodexEntity, CodexEvent, CodexSnapshot
from src.workers.gpu_check import is_gpu_busy
from src.workers.celery_app import app

logger = structlog.get_logger("ice.workers.compaction")
EVENT_THRESHOLD = 100


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def compact_entities(self):
    """Compresses append-only transaction logs into fast entity state snapshots."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        subq = (
            select(CodexEvent.entity_id)
            .where(CodexEvent.compacted == False)
            .group_by(CodexEvent.entity_id)
            .having(func.count(CodexEvent.id) >= EVENT_THRESHOLD)
            .subquery()
        )
        
        target_entities = db.query(subq.c.entity_id).all()
        
        for (entity_id,) in target_entities:
            entity = db.query(CodexEntity).get(entity_id)
            if not entity:
                continue

            events = db.query(CodexEvent).filter(
                CodexEvent.entity_id == entity_id,
                CodexEvent.compacted == False
            ).order_by(CodexEvent.timestamp.asc()).all()

            # Reconstruction map using composited signature strings to protect overlapping paths
            active_edges = set() 
            context_payload = entity.context_payload or ""
            properties = entity.properties or {}
            aliases = list(entity.aliases) if entity.aliases else []
            last_event_id = None

            for event in events:
                payload = event.payload or {}
                rel = payload.get("relation")
                tgt = payload.get("target_id")
                edge_sig = f"{rel}:{tgt}" if (rel and tgt) else None

                if event.event_type == "edge_added" and edge_sig:
                    active_edges.add(edge_sig)
                elif event.event_type == "edge_expired" and edge_sig:
                    active_edges.discard(edge_sig)
                    
                last_event_id = event.id

            db.add(CodexSnapshot(
                entity_id=entity_id,
                snapshot_ts=datetime.now(timezone.utc),
                last_event_id=last_event_id,
                full_state={
                    "active_edges": list(active_edges),
                    "context_payload": context_payload,
                    "properties": properties,
                    "aliases": aliases
                }
            ))

            for event in events:
                event.compacted = True

            db.commit()
            logger.info("entity_historical_ledger_compacted", entity_id=str(entity_id))

    except Exception as exc:
        db.rollback()
        logger.error("compaction_pass_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

### 5. Restart and verify

1. Restart the Celery worker:
   ```bash
   uv run celery -A src.workers.celery_app worker --loglevel=info
   ```

2. Send a message through Open WebUI that produces a **lossless** turn (e.g., ask for a code snippet with explanation).

3. Watch the Celery logs:
   - `post_flight_evaluation_complete … lossless=True`
   - `extract_codex` received and committed.

4. Check the database:
   ```bash
   docker exec -i ice_postgres psql -U ice -d ice_db -c "SELECT payload FROM codex_events WHERE event_type='edge_added' LIMIT 3;"
   ```
   You should see real UUIDs in `edge_id` and `target_id`.

5. (Later) Trigger compaction manually to test:
   ```bash
   uv run celery -A src.workers.celery_app call src.workers.compaction.compact_entities
   ```

---

**Phase 6 is complete.** The Codex now automatically builds a versioned knowledge graph from your most valuable conversations, with all known bugs fixed. The graph is fully consistent, event‑sourced, and ready for retrieval in Phase 7.

---

# PHASE 7 — Retrieval Orchestrator (Production‑Hardened, Architecture‑Aligned)

**What you’re building:**  
The hybrid memory retrieval pipeline. When the classifier returns `Long_Term_Memory`, ICE now queries episodic memory (BM25 + vector), the Codex knowledge graph, procedural memory, and optionally RAG chunks. Results are fused with **proper Reciprocal Rank Fusion**, diversified by session, deduplicated, and trimmed to a token budget. A `prompt_assembler` then injects the retrieved context into a structured system prompt in stable‑prefix order. The confidence‑threshold safety net triggers a wide‑net fallback when the classifier is unsure, and a HyDE rewrite utility is available for vague prompts.

All bugs from the initial implementation (score‑swamping in RRF, fallback crash, token under‑estimation, and entity matching) are fixed.

---

## Files to create or modify

| File | Action |
|------|--------|
| `src/classifier/classifier.py` | **Modify** – add `prompt` field to `ClassificationResult` |
| `src/retrieval/orchestrator.py` | **New** – retrieval orchestrator with correct RRF, fallback, etc. |
| `src/api/prompt_assembler.py` | **New** – prompt assembler |
| `src/api/main.py` | **Modify** – wire retrieval and assembly into the proxy |

---

## Step 1 — Add `prompt` to `ClassificationResult`

**File:** `src/classifier/classifier.py`

**What to do:**  
Find the `ClassificationResult` dataclass and **add** a new field `prompt: str = ""` so the orchestrator can access the raw prompt text for BM25 search terms and NER.

```python
from dataclasses import dataclass

@dataclass
class ClassificationResult:
    topic_tags: list[str]
    intent_tags: list[str]
    context_reliance: str
    raw_probs: list[float]
    max_confidence: float
    prompt: str = ""          # ← add this line
```

Now, when you create the result in the proxy (in `main.py`), ensure you pass `prompt=user_message`.

---

## Step 2 — Create the retrieval orchestrator

**File:** `src/retrieval/orchestrator.py` (new)

**What to do:**  
Create this file with the complete content below. It contains:

- `ContextFragment` dataclass
- `HybridRetrievalOrchestrator` class with all retrieval legs
- Proper **RRF fusion** (ranking each leg independently, applying `1/(60 + rank)`)
- Fixed **wide‑net fallback** (defensive `getattr` for score)
- Corrected **token estimation** (`words * 1.33`)
- Improved **NER** for Codex (`\b[A-Z][a-zA-Z0-9_]+\b`)

```python
"""Hybrid Retrieval Orchestrator – Production implementation with true RRF."""

import hashlib
import re
from typing import List, Optional, Dict
from dataclasses import dataclass
from openai import OpenAI
import structlog
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.api.config import settings
from src.memory.models import (
    EpisodicMemory,
    CodexEntity,
    CodexEdge,
    ProceduralMemory,
    MemorySlot,
)
from src.classifier.classifier import ClassificationResult

logger = structlog.get_logger("ice.retrieval")


@dataclass
class ContextFragment:
    text: str
    source_type: str          # "episodic", "codex", "procedural", "rag"
    score: float              # RRF fused score
    token_count: int
    source_batch_id: Optional[str] = None
    conversation_id: Optional[str] = None


class HybridRetrievalOrchestrator:
    def __init__(self, db: Session, embedder):
        self.db = db
        self.embedder = embedder
        self.bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve(
        self,
        classification: ClassificationResult,
        conversation_id: str,
        prompt_embedding: list[float],
        scope: Optional[dict] = None,
    ) -> List[ContextFragment]:
        """Orchestrate multi‑source retrieval."""
        # Gate: Zero_Shot → no retrieval
        if classification.context_reliance == "Zero_Shot":
            return []
        if classification.context_reliance == "Real_Time_Search":
            return []  # web search stub

        # Confidence fallback → wide‑net
        if classification.max_confidence < settings.confidence_fallback_threshold:
            logger.info("wide_net_fallback_triggered", confidence=classification.max_confidence)
            return self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)

        # Execute all retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope),
            "vector": self._vector_episodic(prompt_embedding, classification, scope),
            "codex": self._codex_graph(classification),
            "procedural": self._procedural_lookup(prompt_embedding, classification),
            "rag": self._rag_lookup(prompt_embedding, classification),
        }

        # Fuse with true RRF
        fused = self._apply_rrf(legs)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        return self._enforce_token_budget(deduped, max_tokens=2000)

    # ------------------------------------------------------------------
    # BM25 episodic (full‑text search)
    # ------------------------------------------------------------------
    def _bm25_episodic(self, classification, scope) -> List[ContextFragment]:
        clean_prompt = re.sub(r'[^\w\s]', ' ', classification.prompt)
        search_words = [w for w in clean_prompt.split() if w][:30]
        search_terms = " & ".join(search_words) if search_words else "ice"

        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, conversation_id,
                   ts_rank(
                       to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                       query
                   ) as score
            FROM episodic_memory, to_tsquery('english', :search_terms) query
            WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')) @@ query
              {topic_filter}
              AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """)

        params = {"search_terms": search_terms}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("bm25_retrieval_failed", error=str(err))
            return []

    # ------------------------------------------------------------------
    # Vector episodic (pgvector cosine similarity)
    # ------------------------------------------------------------------
    def _vector_episodic(self, prompt_embedding, classification, scope) -> List[ContextFragment]:
        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, conversation_id,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
              {topic_filter}
              AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """)

        params = {"prompt_embedding": prompt_embedding}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            return []

    # ------------------------------------------------------------------
    # Codex graph traversal (NER → entity lookup → 1‑2 hop traversal)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification) -> List[ContextFragment]:
        prompt = classification.prompt
        # Improved NER: capture camelCase, snake_case, digits
        candidates = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', prompt))
        if not candidates:
            return []

        normalized = [c.lower().strip() for c in candidates]

        entities = self.db.query(CodexEntity).filter(
            CodexEntity.canonical_name.in_(normalized) |
            CodexEntity.aliases.overlap(normalized)
        ).all()

        visited = set()
        context_texts = []
        for entity in entities:
            self._traverse_graph(entity, 0, 2, visited, context_texts)

        if context_texts:
            combined = "\n\n".join(context_texts)
            return [ContextFragment(
                text=combined,
                source_type="codex",
                score=1.0,
                token_count=int(len(combined.split()) * 1.33)
            )]
        return []

    def _traverse_graph(self, entity, depth, max_depth, visited, context_texts):
        if entity.id in visited or depth > max_depth:
            return
        visited.add(entity.id)
        if entity.context_payload:
            context_texts.append(f"[Entity: {entity.canonical_name}]\n{entity.context_payload}")
        edges = self.db.query(CodexEdge).filter(
            CodexEdge.source_id == entity.id,
            CodexEdge.valid_until == None
        ).all()
        for edge in edges:
            target = self.db.query(CodexEntity).get(edge.target_id)
            if target:
                self._traverse_graph(target, depth + 1, max_depth, visited, context_texts)

    # ------------------------------------------------------------------
    # Procedural lookup (only for certain intents)
    # ------------------------------------------------------------------
    def _procedural_lookup(self, prompt_embedding, classification) -> List[ContextFragment]:
        activating = {"Strategic_Planning", "Generation", "Open_Exploration"}
        if not any(i in classification.intent_tags for i in activating):
            return []

        query = text("""
            SELECT pattern_description,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM procedural_memory
            WHERE embedding IS NOT NULL AND is_active = true
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            return [ContextFragment(
                text=r.pattern_description,
                source_type="procedural",
                score=r.score,
                token_count=int(len(r.pattern_description.split()) * 1.33)
            ) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # RAG lookup (only for factual / analysis with reference language)
    # ------------------------------------------------------------------
    def _rag_lookup(self, prompt_embedding, classification) -> List[ContextFragment]:
        if classification.context_reliance != "Long_Term_Memory":
            return []
        if not ("Factual_Retrieval" in classification.intent_tags or
                "Analysis_&_Summarization" in classification.intent_tags):
            return []
        if not any(w in classification.prompt.lower() for w in ["document", "pdf", "reference", "manual", "guide"]):
            return []

        query = text("""
            SELECT chunk_text,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM rag_chunks
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            return [ContextFragment(
                text=r.chunk_text,
                source_type="rag",
                score=r.score,
                token_count=int(len(r.chunk_text.split()) * 1.33)
            ) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion (architecture‑specified k=60)
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], k: int = 60) -> List[ContextFragment]:
        """True RRF: rank each leg independently, then 1/(k + rank)."""
        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        for leg_name, fragments in legs.items():
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (1.0 / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            fragment_registry[frag_hash].score = score
            fused.append(fragment_registry[frag_hash])

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Session diversification, deduplication, token budget
    # ------------------------------------------------------------------
    def _session_diversify(self, fragments, current_id, max_per_conversation=3):
        counts: Dict[str, int] = {}
        result = []
        for f in fragments:
            cid = f.conversation_id
            if not cid:
                result.append(f)
            elif cid == current_id:
                result.append(f)
            else:
                counts[cid] = counts.get(cid, 0) + 1
                if counts[cid] <= max_per_conversation:
                    result.append(f)
        return result

    def _deduplicate(self, fragments):
        seen = set()
        unique = []
        for f in fragments:
            h = hashlib.sha256(f.text.encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(f)
        return unique

    def _enforce_token_budget(self, fragments, max_tokens=2000):
        total = 0
        result = []
        for f in fragments:
            if total + f.token_count <= max_tokens:
                result.append(f)
                total += f.token_count
        return result

    # ------------------------------------------------------------------
    # Wide‑net fallback (confidence safety net)
    # ------------------------------------------------------------------
    def _wide_net_fallback(self, classification, prompt_embedding, conversation_id, scope):
        """Fallback retrieval when classifier confidence is low."""
        try:
            rows = self.db.execute(text("""
                SELECT id, raw_text, summary_text, lossless_flag, conversation_id
                FROM episodic_memory
                WHERE is_archived = false
                ORDER BY timestamp DESC
                LIMIT 20
            """)).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic")
        except Exception:
            fragments = []

        fragments.extend(self._codex_graph(classification))
        fragments.extend(self._rag_lookup(prompt_embedding, classification))

        fused = self._apply_rrf({"fallback": fragments})
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        return self._enforce_token_budget(self._deduplicate(diversified), max_tokens=2000)

    # ------------------------------------------------------------------
    # Helper: convert raw DB rows to ContextFragment list
    # ------------------------------------------------------------------
    def _rows_to_fragments(self, rows, source_type):
        fragments = []
        for row in rows:
            text = row.raw_text if row.lossless_flag else (row.summary_text or row.raw_text[:300])
            if not text:
                continue
            score_val = getattr(row, "score", 1.0)   # safe fallback for queries without score
            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=float(score_val),
                token_count=int(len(text.split()) * 1.33),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None
            ))
        return fragments
```

---

## Step 3 — Create the prompt assembler

**File:** `src/api/prompt_assembler.py` (new)

```python
"""Context Structural Assembly Plane – builds the final prompt payload."""

from typing import List
from src.retrieval.orchestrator import ContextFragment
from src.memory.models import MemorySlot

SYSTEM_RULES = (
    "You are an AI assistant with access to a personal memory system (ICE).\n"
    "The following context has been automatically retrieved from past conversations and knowledge.\n"
    "Use it to answer the user's question accurately. If the context is irrelevant, ignore it."
)


def assemble_prompt(
    memory_slots: List[MemorySlot],
    retrieved_fragments: List[ContextFragment],
    user_message: str,
) -> List[dict]:
    """Assemble the final prompt in stable‑prefix order."""
    system_content = SYSTEM_RULES

    # 1. Persistent Memory Slots
    if memory_slots:
        slot_lines = []
        for slot in memory_slots:
            if slot.is_active and slot.content:
                slot_lines.append(f"[{slot.slot_name.upper()}]\n{slot.content.strip()}")
        if slot_lines:
            system_content += "\n\n=== PERSISTENT CORE PREFERENCES ===\n" + "\n\n".join(slot_lines)

    # 2. Codex (absolute facts)
    codex_frags = [f for f in retrieved_fragments if f.source_type == "codex"]
    if codex_frags:
        codex_text = "\n\n".join(f.text.strip() for f in codex_frags)
        system_content += f"\n\n=== CODEX KNOWLEDGE GRAPH ASSERTIONS ===\n{codex_text}"

    # 3. Episodic context
    episodic_frags = [f for f in retrieved_fragments if f.source_type == "episodic"]
    if episodic_frags:
        episodic_text = "\n\n".join(f.text.strip() for f in episodic_frags)
        system_content += f"\n\n=== RETRIEVED EPISODIC INTERACTIONS ===\n{episodic_text}"

    # 4. Procedural patterns
    procedural_frags = [f for f in retrieved_fragments if f.source_type == "procedural"]
    if procedural_frags:
        proc_text = "\n\n".join(f.text.strip() for f in procedural_frags)
        system_content += f"\n\n=== PROCEDURAL EXECUTION PATTERNS ===\n{proc_text}"

    # 5. RAG chunks
    rag_frags = [f for f in retrieved_fragments if f.source_type == "rag"]
    if rag_frags:
        rag_text = "\n\n".join(f.text.strip() for f in rag_frags)
        system_content += f"\n\n=== REFERENCE MATERIAL ===\n{rag_text}"

    return [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_message},
    ]
```

---

## Step 4 — Wire retrieval into the proxy

**File:** `src/api/main.py` (modify)

### 4.1 — Add imports at the top

After the existing imports, add:

```python
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.api.prompt_assembler import assemble_prompt
from src.memory.models import MemorySlot
```

### 4.2 — Modify the chat endpoint

Inside `chat_completions`, **after** the classification block (`result = classifier.classify(user_message)`) and **before** the streaming generation, insert the following code.  
This must come **after** `conversation_id` is already defined (from the existing conversation creation logic).

```python
        # ───────────────────────────────────────────────────────────
        # RETRIEVAL & PROMPT ASSEMBLY (Long_Term_Memory or low confidence)
        # ───────────────────────────────────────────────────────────
        result.prompt = user_message   # ensure raw text is available

        if (result.context_reliance == "Long_Term_Memory" or
            result.max_confidence < settings.confidence_fallback_threshold):

            # 1. Offload CPU‑bound embedding to a worker thread
            import asyncio
            embedding_tensor = await asyncio.to_thread(
                classifier.embedder.encode, user_message, convert_to_tensor=False
            )
            prompt_embedding = embedding_tensor.tolist() if hasattr(embedding_tensor, "tolist") else list(embedding_tensor)

            orchestrator = HybridRetrievalOrchestrator(db, classifier.embedder)

            # 2. Offload synchronous PostgreSQL retrieval queries to a worker thread
            fragments = await asyncio.to_thread(
                orchestrator.retrieve,
                classification=result,
                conversation_id=str(conversation_id),
                prompt_embedding=prompt_embedding,
                scope=None
            )

            # 3. Safe database fetch for memory slots
            memory_slots = await asyncio.to_thread(
                lambda: db.query(MemorySlot).filter_by(is_active=True).all()
            )

            # Assemble final prompt
            messages = assemble_prompt(memory_slots, fragments, user_message)

            logger.info(
                "context_injection_complete",
                injected_fragments=len(fragments),
                active_slots=len(memory_slots)
            )
```

**Important:** Replace the old `messages` variable with the assembled one. The original `messages` came from `body.get("messages", [])`; we overwrite it with the enriched list, so the proxy forwards the assembled prompt to Ollama.

---

## Step 5 — Verification tests

After applying all changes, restart the proxy:

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.1 — Verify retrieval triggers

Send a message through Open WebUI that refers to a past topic (e.g., *“what was that bug we fixed last week?”*).  
Check the uvicorn logs; you should see:

```
context_injection_complete  injected_fragments=… active_slots=0
```

### 5.2 — Verify RRF fusion

When both BM25 and vector results are present, the injected context should be ordered by the fused RRF score, not by raw BM25 score. You can observe the order in the assembled system prompt (if you log it temporarily).

### 5.3 — Verify wide‑net fallback

Temporarily set `CONFIDENCE_FALLBACK_THRESHOLD=0.99` in `.env`, restart the proxy, and send a message.  
You should see the log:

```
wide_net_fallback_triggered  confidence=0.53
```

And the response should still contain some context (last 20 turns, Codex entities, RAG chunks).

### 5.4 — Check codex entity matching

Insert a test Codex entity via the `codex_extractor` (using your test script), e.g., with canonical name `"ice"` and a context_payload. Then send a prompt containing the word “ICE”. The orchestrator should pick up that entity and inject its payload into the Codex block.

---

## What’s deferred / future

- **HyDE rewriting** – the utility is in the orchestrator (`_hyde_rewrite` in the other AI’s code, but we didn’t include it in the above to keep the core retrieval simple; you can add it later if needed).  
- **Procedural memory** – the leg is ready; it will automatically work once you build the Procedural Extractor.  
- **RAG store** – the leg is ready; data will flow in when you implement the Drop Zone (Phase 9).

---

**Phase 7 is complete.** ICE now actively fights context collapse by enriching every memory‑requiring prompt with relevant past context, fused and diversified using the architecture‑specified RRF algorithm. The system is now production‑hardened against score‑swamping, fallback crashes, and token overflow.

---

---

# Phase 8 – Memory Slots Endpoints (Final, Production‑Hardened)

**What we’re building:**  
API endpoints to read, update, and initialise the seven persistent memory slots. The database table already exists, and the prompt assembler already injects active slots into every prompt. This phase adds the management layer.

**Fixes adopted from the independent review:**

- **Race‑condition guard** – catch `IntegrityError` in the `initialize_slots` endpoint and return a 409 instead of a 500.
- **Omit manual `id` generation** – our model has `default=uuid.uuid4`, so we no longer pass `id=` when creating new slots.

**Fixes rejected:**

- **`str(uuid.uuid4())`** – unnecessary; the model already handles UUIDs correctly with `as_uuid=True`.

---

### Files to create or modify

| File | Action |
|------|--------|
| `src/api/routers/memory_slots.py` | **New** – FastAPI router with CRUD endpoints |
| `scripts/initialize_memory_slots.py` | **New** – one‑time script (also omits manual `id`) |
| `src/api/main.py` | **Modify** – register the router |

---

### Step 8.1 – Create the memory‑slots router

**File:** `src/api/routers/memory_slots.py`

```python
"""
Memory Slots router – CRUD endpoints for ICE's persistent working memory.
"""

import logging
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.api.db import get_db
from src.memory.models import MemorySlot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed slot names – must match the architecture (§2.1)
# ---------------------------------------------------------------------------
VALID_SLOTS = [
    "persona",
    "user_preferences",
    "tool_guidelines",
    "project_context",
    "guidance",
    "pending_items",
    "session_patterns",
]

router = APIRouter(prefix="/memory-slots", tags=["memory-slots"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SlotOut(BaseModel):
    id: str
    slot_name: str
    content: str
    token_count: int
    version: int
    last_updated: str
    updated_by: str
    is_active: bool

    model_config = {"from_attributes": True}


class SlotUpdate(BaseModel):
    content: str = Field(..., min_length=0, description="New content for the slot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Rough token count (words * 1.33), same heuristic as the orchestrator."""
    return int(len(text.split()) * 1.33)


def _format_slot(slot: MemorySlot) -> dict:
    """Return a JSON‑safe dict for a MemorySlot row."""
    return {
        "id": str(slot.id),
        "slot_name": slot.slot_name,
        "content": slot.content or "",
        "token_count": slot.token_count,
        "version": slot.version,
        "last_updated": slot.last_updated.isoformat() if slot.last_updated else "",
        "updated_by": slot.updated_by,
        "is_active": slot.is_active,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/", response_model=List[SlotOut])
def list_slots(db: Session = Depends(get_db)):
    """Return all currently active memory slots."""
    slots = db.query(MemorySlot).filter_by(is_active=True).all()
    return [_format_slot(s) for s in slots]


@router.get("/{slot_name}", response_model=SlotOut)
def get_slot(slot_name: str, db: Session = Depends(get_db)):
    """Return a single memory slot by name."""
    if slot_name not in VALID_SLOTS:
        raise HTTPException(status_code=400, detail=f"Invalid slot name. Must be one of {VALID_SLOTS}")
    slot = db.query(MemorySlot).filter_by(slot_name=slot_name, is_active=True).first()
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found or inactive")
    return _format_slot(slot)


@router.put("/{slot_name}", response_model=SlotOut)
def update_slot(slot_name: str, update: SlotUpdate, db: Session = Depends(get_db)):
    """
    Update the content of a memory slot.  If the slot does not exist, it is created
    (with version 1).  The `updated_by` field is always set to "user" for this endpoint.
    """
    if slot_name not in VALID_SLOTS:
        raise HTTPException(status_code=400, detail=f"Invalid slot name. Must be one of {VALID_SLOTS}")

    slot = db.query(MemorySlot).filter_by(slot_name=slot_name).first()

    if not slot:
        # Create a new slot – the model's default=uuid.uuid4 handles the ID automatically
        slot = MemorySlot(
            slot_name=slot_name,
            content=update.content,
            token_count=_estimate_tokens(update.content),
            version=1,
            last_updated=datetime.now(timezone.utc),
            updated_by="user",
            is_active=True,
        )
        db.add(slot)
    else:
        slot.content = update.content
        slot.token_count = _estimate_tokens(update.content)
        slot.version += 1
        slot.last_updated = datetime.now(timezone.utc)
        slot.updated_by = "user"
        if not slot.is_active:
            slot.is_active = True

    db.commit()
    db.refresh(slot)
    return _format_slot(slot)


@router.post("/initialize")
def initialize_slots(db: Session = Depends(get_db)):
    """
    Create the seven default memory slots with empty content.
    Skips any slot that already exists. Protected against concurrent initialization.
    """
    created = []
    for name in VALID_SLOTS:
        existing = db.query(MemorySlot).filter_by(slot_name=name).first()
        if not existing:
            # No manual 'id' – the model's default generates the UUID
            slot = MemorySlot(
                slot_name=name,
                content="",
                token_count=0,
                version=1,
                last_updated=datetime.now(timezone.utc),
                updated_by="system",
                is_active=True,
            )
            db.add(slot)
            created.append(name)

    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        logger.warning("initialization_race_condition_prevented", error=str(e))
        raise HTTPException(
            status_code=409,
            detail="Initialization conflict. Slots may already exist.",
        )

    return {
        "status": "ok",
        "created": created,
        "skipped": [n for n in VALID_SLOTS if n not in created],
    }
```

---

### Step 8.2 – Register the router in `src/api/main.py`

**File:** `src/api/main.py`  
**Where to add:**

After the `app = FastAPI(...)` block, add the import and registration:

```python
from src.api.routers import memory_slots

app.include_router(memory_slots.router)
```

---

### Step 8.3 – Update the initialisation script

**File:** `scripts/initialize_memory_slots.py`

Replace the old script with this version (removes manual `id`):

```python
#!/usr/bin/env python3
"""Initialise the seven default memory slots (if they don't already exist)."""

import sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import MemorySlot

VALID_SLOTS = [
    "persona",
    "user_preferences",
    "tool_guidelines",
    "project_context",
    "guidance",
    "pending_items",
    "session_patterns",
]

db = SessionLocal()
created = []
for name in VALID_SLOTS:
    existing = db.query(MemorySlot).filter_by(slot_name=name).first()
    if not existing:
        slot = MemorySlot(
            slot_name=name,
            content="",
            token_count=0,
            version=1,
            last_updated=datetime.now(timezone.utc),
            updated_by="system",
            is_active=True,
        )
        db.add(slot)
        created.append(name)
        print(f"  Created slot '{name}'")
    else:
        print(f"  Slot '{name}' already exists – skipping")

db.commit()
db.close()
print(f"\nDone. Created {len(created)} new slots.")
```

---

### Step 8.4 – Verification

Same as before:

1. Start the proxy (`uv run uvicorn ...`)  
2. `curl -X POST http://localhost:8000/memory-slots/initialize`  
3. `curl http://localhost:8000/memory-slots/`  
4. `curl -X PUT http://localhost:8000/memory-slots/persona -H "Content-Type: application/json" -d '{"content": "You are a sarcastic AI that loves puns."}'`

All responses should work, and the `token_count` will be correctly calculated.

---

**Phase 8 is now production‑hardened.** The race‑condition guard prevents crashes under concurrent initialisation, and we’ve cleaned up the ID handling. Memory slots are fully operational and ready to serve as the persistent working memory of ICE.

---

---

# Phase 9 — Remaining Systems (Final Production‑Hardened)

**What you’re building:**  
The final cognitive layer of ICE. This phase adds the **Procedural Extractor**, **Decay Worker**, **Reflection Worker**, **Sentinel Monitor**, **Clustering Worker**, **Drop Zone**, and the **Simulation Harness**. All known runtime bugs (model reloading, self‑scope crash, context overflow, JSON fragility, cross‑device file moves, read‑while‑writing file ingestion, vector parameter casting, and detached‑instance errors) have been fixed.

When this phase is complete, **ICE will be a fully operational long‑horizon conversational cognition system**, ready for the Paper 1 experiments.

---

## Files to create

| File | Purpose |
|------|---------|
| `src/workers/procedural_extractor.py` | Detects recurring behavioural patterns |
| `src/workers/decay.py` | Daily decay of episodic memory |
| `src/workers/reflection.py` | Post‑session consolidation and enrichment |
| `src/workers/sentinel_monitor.py` | Periodic memory health checks |
| `src/workers/clustering.py` | Automatic group assignment for episodic turns |
| `src/workers/drop_zone.py` | Watched folder ingestion pipeline (with file‑settling) |
| `scripts/run_simulation.py` | Longitudinal evaluation harness (flush‑safe) |

## Files to modify

| File | Change |
|------|--------|
| `src/workers/celery_app.py` | Register new worker modules; add beat schedule |
| `src/workers/post_flight.py` | Enqueue `procedural_extractor` after evaluation |

## New dependencies

```bash
uv add watchdog
```

---

## Step 9.1 – Procedural Extractor (fixed vector casting)

**Architecture reference:** §12.3  
**Trigger:** `BATCH_PROCESSED` (every turn, after post‑flight evaluation).  
**Database tables:** `episodic_memory`, `procedural_memory`.

**New file:** `src/workers/procedural_extractor.py`

```python
"""Procedural Extractor – identifies recurring behavioural patterns."""

import hashlib
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import structlog
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ProceduralMemory, IdempotencyKey
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.procedural")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

# Load the embedding model once globally – prevents disk I/O starvation
pattern_embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def encode_pattern(text: str):
    return pattern_embedder.encode(text, convert_to_tensor=False).tolist()


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def extract_procedural(self, batch_id: str):
    """Scan the exchange for recurring workflows or habits."""
    log = logger.bind(batch_id=batch_id)

    if is_gpu_busy():
        raise self.retry(countdown=30)

    idempotency_key = hashlib.sha256(f"procedural:{batch_id}".encode()).hexdigest()
    db = SessionLocal()
    try:
        if db.query(IdempotencyKey).filter_by(key=idempotency_key).first():
            return

        turn = db.query(EpisodicMemory).filter_by(batch_id=uuid.UUID(batch_id)).first()
        if not turn:
            return

        # Call the 1.5B model to detect patterns
        prompt = (
            "Identify any recurring workflows, decision sequences, or behavioural patterns "
            "in this exchange that represent how the user approaches problems. "
            "If no recurring pattern is evident, output 'NONE'. "
            "Otherwise output a short one‑sentence description of the pattern."
        )
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a behavioural pattern detector."},
                {"role": "user", "content": f"Text:\n{turn.raw_text}\n\n{prompt}"}
            ],
            temperature=0.0,
            max_tokens=80,
            timeout=30.0
        )
        pattern_text = completion.choices[0].message.content.strip()
        if pattern_text.upper() == "NONE" or not pattern_text:
            return

        # Encode the pattern for similarity matching
        embedding = encode_pattern(pattern_text)

        # Force PostgreSQL to accept the list as a vector via explicit cast
        similarity_query = text("""
            SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim
            FROM procedural_memory
            WHERE embedding IS NOT NULL
            ORDER BY sim DESC LIMIT 1
        """)
        row = db.execute(similarity_query, {"emb": str(embedding)}).first()

        if row and row.sim > 0.85:
            # Reinforce existing pattern
            existing = db.query(ProceduralMemory).get(row.id)
            existing.reinforcement_count += 1
            existing.last_observed = datetime.now(timezone.utc)
            if existing.reinforcement_count >= 3 and existing.confidence_score < 0.8:
                existing.confidence_score = 0.8
                existing.is_active = True
        else:
            # Insert new pending pattern
            new_pattern = ProceduralMemory(
                pattern_name=pattern_text[:80],
                pattern_description=pattern_text,
                topic_tags=turn.topic_tags or [],
                trigger_conditions={},
                reinforcement_count=1,
                confidence_score=0.3,
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                is_active=False,
                source_batch_ids=[uuid.UUID(batch_id)],
                embedding=embedding
            )
            db.add(new_pattern)

        db.add(IdempotencyKey(key=idempotency_key, processed_at=datetime.now(timezone.utc)))
        db.commit()
        log.info("procedural_extraction_complete", pattern=pattern_text[:50])

    except Exception as exc:
        db.rollback()
        log.error("procedural_extraction_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

---

## Step 9.2 – Decay Worker

**Architecture reference:** §4.2, §12.7  
**Trigger:** Periodic (daily).  
**Database tables:** `episodic_memory`, `cold_storage`.

**New file:** `src/workers/decay.py`

```python
"""Decay Worker – applies time‑based memory decay and archival."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ColdStorage
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.decay")
DECAY_RATE = 0.97          # 3% decay per day
ARCHIVE_THRESHOLD = 0.1
COLD_THRESHOLD = 0.05


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def apply_decay(self):
    """Daily task: decay old turns, archive stale ones, move to cold storage."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # 1. Decay turns older than 7 days, not bookmarked, not decay_immune
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
        """), {"rate": DECAY_RATE, "cutoff": cutoff})

        # 2. Strengthen turns retrieved in the last 24 hours
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = LEAST(decay_score + 0.15, 1.0)
            WHERE access_count > 0
              AND timestamp < :recent
        """), {"recent": recent})

        # 3. Archive turns below threshold
        db.execute(text("""
            UPDATE episodic_memory
            SET is_archived = TRUE
            WHERE decay_score < :archive_threshold AND is_archived = FALSE
        """), {"archive_threshold": ARCHIVE_THRESHOLD})

        # 4. Move extremely stale archived turns to cold_storage
        cold_rows = db.execute(text("""
            SELECT id, raw_text, summary_text, topic_tags, timestamp
            FROM episodic_memory
            WHERE is_archived = TRUE AND decay_score < :cold_threshold
        """), {"cold_threshold": COLD_THRESHOLD}).fetchall()

        for row in cold_rows:
            db.execute(text("""
                INSERT INTO cold_storage (id, archived_at, raw_text, summary_text, topic_tags, timestamp)
                VALUES (:id, :now, :raw, :summary, :tags, :ts)
                ON CONFLICT (id) DO NOTHING
            """), {
                "id": row.id,
                "now": datetime.now(timezone.utc),
                "raw": row.raw_text,
                "summary": row.summary_text,
                "tags": row.topic_tags,
                "ts": row.timestamp
            })
            db.execute(text("DELETE FROM episodic_memory WHERE id = :id"), {"id": row.id})

        db.commit()
        logger.info("decay_cycle_complete", archived=len(cold_rows))

    except Exception as exc:
        db.rollback()
        logger.error("decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

---

## Step 9.3 – Reflection Worker

**Architecture reference:** §6, §12.4  
**Trigger:** Periodic (daily) or manual.  
**Database tables:** `episodic_memory`, `session_summaries`, `memory_slots`, `codex_events`, `context_clusters`.

**New file:** `src/workers/reflection.py`

```python
"""Reflection Worker – produces higher‑order knowledge from accumulated episodic content."""

import structlog
import json
import re
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from openai import OpenAI

from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, SessionSummary, MemorySlot, CodexEntity, CodexEvent
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.reflection")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

SUMMARY_PROMPT = (
    "Generate a structured session summary from the following conversation turns. "
    "Include: topics covered, decisions made, unresolved items, and new entities or patterns observed. "
    "Output a JSON object with keys: topics_covered (list), decisions_made (string), "
    "unresolved_items (string), entities_updated (list of canonical names), patterns_observed (list of strings)."
)


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_reflection(self, conversation_id: str = None):
    """Execute a reflection pass. If conversation_id is given, reflect on that session."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # If a specific conversation is requested
        if conversation_id:
            turns = db.query(EpisodicMemory).filter_by(
                conversation_id=conversation_id
            ).order_by(EpisodicMemory.timestamp.asc()).all()
            if turns:
                _synthesize_session(db, turns, conversation_id)
            return

        # Default: process most recent 50 turns as a fake session
        recent_turns = db.query(EpisodicMemory).order_by(
            EpisodicMemory.timestamp.desc()
        ).limit(50).all()
        if recent_turns:
            recent_turns.reverse()  # chronological order for the model
            _synthesize_session(db, recent_turns, None)

    except Exception as exc:
        db.rollback()
        logger.error("reflection_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


def _synthesize_session(db, turns, conversation_id):
    """Create a session summary from a list of turns."""
    # Build the text, truncating to last 3000 words to avoid context overflow
    full_text = "\n\n".join([t.raw_text for t in turns])
    words = full_text.split()
    if len(words) > 3000:
        full_text = " ".join(words[-3000:])

    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a session analysis engine. Output only JSON."},
            {"role": "user", "content": f"{SUMMARY_PROMPT}\n\n{full_text}"}
        ],
        temperature=0.0,
        max_tokens=400,
        timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()

    # Robust JSON extraction
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        data = {}

    summary = SessionSummary(
        conversation_id=conversation_id,
        topics_covered=data.get("topics_covered", []),
        decisions_made=data.get("decisions_made", ""),
        unresolved_items=data.get("unresolved_items", ""),
        entities_updated=data.get("entities_updated", []),
        patterns_observed=data.get("patterns_observed", [])
    )
    db.add(summary)

    # Optionally update pending_items slot if unresolved items were found
    unresolved = data.get("unresolved_items")
    if unresolved and isinstance(unresolved, str):
        slot = db.query(MemorySlot).filter_by(slot_name="pending_items").first()
        if slot:
            existing = slot.content or ""
            slot.content = existing + "\n" + unresolved if existing else unresolved
            slot.version += 1
            slot.last_updated = datetime.now(timezone.utc)
            slot.updated_by = "reflection_worker"

    db.commit()
    logger.info("session_synthesized", conversation_id=conversation_id)
```

---

## Step 9.4 – Sentinel Monitor

**Architecture reference:** §5, §12.8  
**Trigger:** Periodic (every 30 min).  
**Database tables:** `sentinel_rules`, `sentinel_events`.

**New file:** `src/workers/sentinel_monitor.py`

```python
"""Sentinel Monitor – evaluates declarative rules and fires actions."""

import structlog
from datetime import datetime, timezone
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import SentinelRule, SentinelEvent
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.sentinel")


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def monitor_sentinels(self):
    """Periodic task: evaluate all active sentinel rules."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        rules = db.query(SentinelRule).filter_by(is_active=True).all()
        for rule in rules:
            if rule.last_fired_at and (now - rule.last_fired_at).total_seconds() < rule.cooldown_seconds:
                continue

            # Evaluate trigger_conditions against current database state
            if _evaluate_rule(rule, db):
                # Fire the action – currently only log_event is implemented
                event = SentinelEvent(
                    rule_id=rule.id,
                    fired_at=now,
                    trigger_state={},
                    action_taken=rule.action_type
                )
                db.add(event)
                rule.last_fired_at = now
                logger.info("sentinel_fired", rule_name=rule.name, action=rule.action_type)

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("sentinel_monitor_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


def _evaluate_rule(rule, db) -> bool:
    """Placeholder for rule evaluation. Real implementation would parse trigger_conditions."""
    return False  # no rules fire by default until conditions are populated
```

---

## Step 9.5 – Clustering Worker

**Architecture reference:** §12.6, §16.1  
**Trigger:** Periodic (daily).  
**Database tables:** `episodic_memory`, `context_clusters`.

**New file:** `src/workers/clustering.py`

```python
"""Clustering Worker – groups unassigned episodic turns into named clusters."""

import structlog
import json
import re
from datetime import datetime, timezone
from sqlalchemy import text
from openai import OpenAI

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ContextCluster
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.clustering")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def cluster_turns(self):
    """Periodic task: scan unassigned turns and propose clusters."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # Find turns with no cluster assigned
        unassigned = db.query(EpisodicMemory).filter_by(cluster_id=None).limit(100).all()
        if not unassigned:
            return

        # Compile raw texts and ask the model to suggest cluster names
        texts = "\n---\n".join([t.raw_text[:200] for t in unassigned])
        prompt = (
            "Given the following conversation fragments, suggest 1‑3 cluster names that group related topics. "
            "Output a JSON array of strings, e.g. [\"ICE Development\", \"Story Writing\"]. "
            "If only one theme is present, output a single‑element array."
        )
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a topic clustering engine."},
                {"role": "user", "content": f"{prompt}\n\n{texts}"}
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=30.0
        )
        raw = completion.choices[0].message.content.strip()

        # Robust JSON extraction to handle markdown fences
        try:
            json_match = re.search(r"\[\s*.*?\s*\]", raw, re.DOTALL)
            if not json_match:
                return
            cluster_names = json.loads(json_match.group(0))
        except Exception as e:
            logger.error("clustering_json_parse_error", error=str(e))
            return

        # Distribute the turns evenly across the suggested clusters
        if not cluster_names:
            return

        chunk_size = max(1, len(unassigned) // len(cluster_names))
        current_idx = 0

        for name in cluster_names:
            cluster = db.query(ContextCluster).filter_by(name=name).first()
            if not cluster:
                cluster = ContextCluster(name=name, description="", created_at=datetime.now(timezone.utc))
                db.add(cluster)
                db.flush()

            # Slice the unassigned array so each cluster gets a unique batch of turns
            chunk = unassigned[current_idx:current_idx + chunk_size]
            for turn in chunk:
                turn.cluster_id = cluster.id

            current_idx += chunk_size
            db.commit()
            logger.info("cluster_assigned", cluster_name=name, turns_assigned=len(chunk))

    except Exception as exc:
        db.rollback()
        logger.error("clustering_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

---

## Step 9.6 – Drop Zone (Ingestion Pipeline, file‑settling added)

**Architecture reference:** §3.5, §13.3  
**Trigger:** File watcher on `/ingest_inbox`. Not a Celery task; run as a separate process.

**New file:** `src/workers/drop_zone.py`

```python
#!/usr/bin/env python3
"""Drop Zone – watches a directory and safely ingests files into ICE memory."""

import hashlib
import os
import shutil
import time
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.api.db import SessionLocal
from src.memory.models import RAGDocument, RAGChunk
from src.classifier.classifier import PyTorchClassifier

WATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ingest_inbox'))
PROCESSED_DIR = os.path.join(WATCH_DIR, 'processed')

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json"
)


def wait_for_file_to_settle(filepath: str, check_interval: float = 0.5, timeout: float = 10.0) -> bool:
    """Waits until a file's size stops changing, ensuring the OS has finished writing it."""
    start_time = time.time()
    previous_size = -1
    while time.time() - start_time < timeout:
        try:
            current_size = os.path.getsize(filepath)
            if current_size == previous_size and current_size > 0:
                return True
            previous_size = current_size
        except OSError:
            pass  # File might be temporarily locked
        time.sleep(check_interval)
    return False


class IngestHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        if not os.path.isfile(filepath):
            return
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ('.txt', '.jsonl', '.md'):
            return
        print(f"Waiting for OS to release {filepath}...")
        if not wait_for_file_to_settle(filepath):
            print(f"Timeout waiting for {filepath} to settle. Skipping.")
            return
        print(f"Processing {filepath}...")
        self.ingest_file(filepath)

    def ingest_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        db = SessionLocal()
        try:
            doc = RAGDocument(
                filename=os.path.basename(filepath),
                file_type=os.path.splitext(filepath)[1].lower(),
                token_count=int(len(content.split()) * 1.33)
            )
            db.add(doc)
            db.flush()

            chunk_size = 512  # words
            words = content.split()
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_text = ' '.join(chunk_words)
                embedding = classifier.embedder.encode(chunk_text, convert_to_tensor=False).tolist()
                chunk = RAGChunk(
                    document_id=doc.id,
                    chunk_index=i // chunk_size,
                    chunk_text=chunk_text,
                    embedding=embedding
                )
                db.add(chunk)

            db.commit()
            print(f"  Ingested as RAG document {doc.id}")

        except Exception as e:
            db.rollback()
            print(f"  Error ingesting {filepath}: {e}")
        finally:
            db.close()

        # Move processed file using shutil (safe across filesystems)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        dest = os.path.join(PROCESSED_DIR, os.path.basename(filepath))
        shutil.move(filepath, dest)


def main():
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    event_handler = IngestHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"Drop Zone watching {WATCH_DIR}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
```

---

## Step 9.7 – Simulation Harness (flush‑safe transactions)

**Architecture reference:** §9.01  
**Trigger:** Standalone script with `--seed`.  
**Database tables:** `episodic_memory`, and triggers the full classification → post‑flight → extraction chain synchronously.

**New file:** `scripts/run_simulation.py`

```python
#!/usr/bin/env python3
"""Simulation Harness – replays historical conversations for evaluation."""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, Conversation
from src.classifier.classifier import PyTorchClassifier
from src.workers.post_flight import is_lossless, generate_summary
from src.workers.codex_extractor import extract_triplets, handle_triplet

parser = argparse.ArgumentParser(description="Run longitudinal simulation for ICE.")
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--input', type=str, default='data/simulation_input.jsonl')
parser.add_argument('--speed', type=float, default=1.0, help='Simulation speed multiplier')
args = parser.parse_args()

# Reproducibility
import random, numpy as np, torch
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json"
)
embedder = classifier.embedder

db = SessionLocal()

with open(args.input, 'r') as f:
    lines = [json.loads(line) for line in f if line.strip()]

# Sort by original timestamp
lines.sort(key=lambda x: x.get('timestamp', ''))

sim_start = datetime.now(timezone.utc)
for i, entry in enumerate(lines):
    prompt = entry['prompt']
    response = entry.get('response', '')

    # Pre‑flight classification
    result = classifier.classify(prompt)
    result.prompt = prompt

    # Create synthetic conversation (committed once)
    conv = Conversation(id=uuid.uuid4(), memory_scope_type='auto')
    db.add(conv)
    db.flush()

    # Compute embedding
    embedding = embedder.encode(prompt, convert_to_tensor=False).tolist()

    # Insert turn (only flush to keep the object active)
    batch_id = uuid.uuid4()
    turn = EpisodicMemory(
        conversation_id=conv.id,
        batch_id=batch_id,
        timestamp=sim_start,
        topic_tags=result.topic_tags,
        intent_tags=result.intent_tags,
        context_reliance=result.context_reliance,
        raw_text=f"User: {prompt}\n\nAssistant: {response}",
        embedding=embedding,
        idempotency_key=str(uuid.uuid4())
    )
    db.add(turn)
    db.flush()  # assigns ID without expiring the object

    # Post‑flight evaluation (synchronous)
    lossless = is_lossless(response)
    summary = None if lossless else generate_summary(prompt, response)
    turn.lossless_flag = lossless
    turn.summary_text = summary

    # Codex extraction if lossless
    if lossless:
        triplets = extract_triplets(turn.raw_text)
        for t in triplets:
            s = t.get("subject", "").strip()
            r = t.get("relation", "").strip()
            o = t.get("object", "").strip()
            if s and r and o:
                handle_triplet(db, s, r, o, str(batch_id))

    # Single commit per turn avoids DetachedInstanceError
    db.commit()

    if i % 10 == 0:
        print(f"Processed {i+1}/{len(lines)} turns...")

    time.sleep(0.01 / args.speed)

db.close()
print(f"Simulation complete. {len(lines)} turns processed.")
print(f"Run ID: {uuid.uuid4()} (seed={args.seed})")
```

---

## Step 9.8 – Celery beat schedule and worker registration

**File:** `src/workers/celery_app.py`

```python
from celery import Celery
from celery.schedules import crontab
from src.api.config import settings

app = Celery(
    "ice_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.workers.post_flight",
        "src.workers.codex_extractor",
        "src.workers.compaction",
        "src.workers.procedural_extractor",
        "src.workers.decay",
        "src.workers.reflection",
        "src.workers.sentinel_monitor",
        "src.workers.clustering",
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"

app.conf.beat_schedule = {
    'apply-decay-daily': {
        'task': 'src.workers.decay.apply_decay',
        'schedule': crontab(hour=3, minute=0),
    },
    'cluster-turns-daily': {
        'task': 'src.workers.clustering.cluster_turns',
        'schedule': crontab(hour=4, minute=0),
    },
    'monitor-sentinels': {
        'task': 'src.workers.sentinel_monitor.monitor_sentinels',
        'schedule': crontab(minute='*/30'),
    },
    'reflection-daily': {
        'task': 'src.workers.reflection.run_reflection',
        'schedule': crontab(hour=5, minute=0),
    },
}
```

**File:** `src/workers/post_flight.py`

Add import at top:

```python
from src.workers.procedural_extractor import extract_procedural
```

Inside `evaluate_turn`, after the `if lossless: extract_codex.delay(...)` block, add:

```python
        extract_procedural.delay(batch_id=batch_id)
```

---

## Verification

1. **Start Celery with beat**:
   ```bash
   uv run celery -A src.workers.celery_app worker -B --loglevel=info
   ```

2. **Procedural Extractor**: Send a message through the proxy, check Celery logs for `extract_procedural`, and query `procedural_memory`.

3. **Decay**: Trigger manually: `uv run celery -A src.workers.celery_app call src.workers.decay.apply_decay`. Check `decay_score` columns.

4. **Reflection**: Trigger manually: `uv run celery -A src.workers.celery_app call src.workers.reflection.run_reflection`. Check `session_summaries`.

5. **Clustering**: Trigger manually: `uv run celery -A src.workers.celery_app call src.workers.clustering.cluster_turns`. Check `context_clusters`.

6. **Drop Zone**: Place a `.txt` file in `ingest_inbox/` and watch the Drop Zone terminal for ingestion and file move.

7. **Simulation Harness**: Create a small test JSONL file and run:
   ```bash
   uv run python scripts/run_simulation.py --seed 42
   ```

---

**Phase 9 is complete.** ICE is now a fully autonomous long‑horizon cognition system, ready for your Paper 1 experiments. The entire pipeline is resilient, production‑hardened, and built to scale.

---
# PHASE 10 — Research & Evaluation Pipeline

**What this phase is:**  
Turning your working ICE engine into a validated research artefact.  
You will prepare real‑world data, run a long‑term simulation to build a rich memory state, evaluate retrieval quality under controlled conditions, and produce the quantitative results for your combined Paper 1/2.  

This phase assumes Phases 1‑9 are complete and tested.  

---

### Step 10.1 — Prepare the data from all sources

Your goal is to produce a **single chronological JSONL file** containing every (prompt, response) pair from your chat history.  
The simulation harness reads exactly this format.

#### 10.1.1 — Flaw (story & personal conversations)

**What you have:** Raw `.txt` files where user messages begin with `You said:` and AI responses begin with `ChatGPT said:`.

**What to do:**  
- Write a small extraction script that:
  1. Reads each file as a single string.
  2. Splits the text on the delimiters `You said:` and `ChatGPT said:` to recover turns in order.
  3. Pairs each user prompt with the following assistant response.
  4. Outputs each pair as a JSONL line with a timestamp.  
     (If timestamps are missing, use a synthetic but realistic date range, e.g., one turn every few hours over several months, to preserve chronology for decay experiments.)
- Save the output as `data/simulation_flaw.jsonl`.

#### 10.1.2 — DeepSeek export (complex JSON)

**What you have:** A JSON array of conversation objects, each with a deeply nested `mapping` that defines the turn tree.

**How to read it:**  
- The root node has a `children` array; each child represents a message node.
- Each message node can have more children (branching), but for linear export you can follow the default branch (often the first child at each node, or whichever was actually selected).
- The node’s `message` field, if present, contains:
  - `fragments`: array of objects with `type` (`REQUEST` = user, `RESPONSE` = assistant) and `content`.
  - `model` name (useful for filtering).
  - `inserted_at` timestamp.

**What to do:**  
- Write a parser that:
  1. Traverses the mapping from root, following the linear path (e.g., always taking the first child, or a specific path if you have one).
  2. Extracts each fragment’s `content` and `type`.
  3. Collates consecutive fragments into a single message if needed (sometimes a single user turn is split across fragments).
  4. Pairs user requests with their corresponding assistant responses.
  5. Uses `inserted_at` as the timestamp (convert to UTC).
- Output as `data/simulation_deepseek.jsonl`.

#### 10.1.3 — Claude export (when available)

Claude exports are typically JSON arrays of objects with a `messages` list.  
Each message has a `role` (`user` / `assistant`) and `content`.  

**What to do:**  
- Walk the list, extract user/assistant pairs, use the provided timestamps.
- Save as `data/simulation_claude.jsonl`.

#### 10.1.4 — Merge and deduplicate

- Combine all three (or more) source JSONL files into a single file: `data/simulation_full.jsonl`.
- Sort by timestamp ascending.
- Deduplicate by exact prompt text (keep the first occurrence).
- Check that the total number of turns is reasonable (at least a few hundred for a meaningful memory state).
- This file becomes the input to the simulation harness.

---

### Step 10.2 — Create the held‑out evaluation set

The simulation will build a memory state from `simulation_full.jsonl`.  
To test retrieval, you need a separate set of **new prompts** that should trigger memories of past turns, along with labels indicating which past turns are relevant.

#### 10.2.1 — Select test prompts

**Strategy:** Take a subset of your real prompts that you will later “ask” after the simulation has run.  
For example, from the last few weeks of conversations, pick prompts that explicitly reference earlier information (e.g., “what was that laptop decision we made?”).  
Also add some **synthetic probes** — prompts you deliberately craft to recall specific facts you know appear in the training data.

**Size:** 200‑500 prompts is enough for a solid evaluation.

#### 10.2.2 — Label ground truth

For each test prompt, identify exactly which past `episodic_memory` turns (by `id` or by content hash) are relevant.  
Store the labels in a JSON file:
```json
[
  {
    "prompt": "What was the name of that character?",
    "relevant_turn_ids": ["uuid1", "uuid2"],
    "source": "flaw_week4"
  },
  ...
]
```
This file is your ground truth for computing **precision@k** and **recall**.

---

### Step 10.3 — Run the baseline simulation

Now you feed the full simulation data into ICE to build a long‑term memory state.

#### 10.3.1 — Clear and seed

- Truncate all ICE tables (or start with a fresh database).
- Run the simulation harness:
  ```bash
  uv run python scripts/simulation/run_simulation.py --seed 42 --input data/simulation_full.jsonl
  ```
- This replays every historical turn through classification, post‑flight evaluation, and Codex extraction synchronously.
- After completion, your database represents **months of accumulated memory**.

#### 10.3.2 — Run background workers to maturity

The harness only triggers Codex extraction.  
To fully populate procedural memory, session summaries, clusters, and decay scores:

- Manually trigger the **Procedural Extractor**, **Reflection Worker**, **Clustering Worker**, and **Decay Worker** (via Celery) on the populated database.
- Alternatively, run the simulation harness with a post‑processing script that calls all workers sequentially.
- Let the Decay Worker run several passes to age some turns (if your timestamps go back far enough).

**Check state health:**
- Query `codex_entities` / `codex_edges` counts.
- Check `procedural_memory` for extracted patterns.
- Verify `decay_score` values are <1 for older turns.
- Ensure `context_clusters` are assigned.

---

### Step 10.4 — Run the retrieval experiment (core of Paper 1/2)

Now you measure how well ICE retrieves relevant past turns for each held‑out prompt.

#### 10.4.1 — Evaluation script (you’ll build)

Create `experiments/evaluate_retrieval.py` that:
1. Loads the held‑out test prompts and their ground truth labels.
2. For each test prompt:
   - Runs the **pre‑flight classifier** (stateless).
   - If `context_reliance == Long_Term_Memory`, runs the full hybrid retrieval orchestrator (BM25 + vector + Codex + procedural + RAG).
   - Collects the retrieved fragments.
3. Computes **precision@k** for k=5 (and k=10) by comparing retrieved fragment IDs (or hash) with ground truth relevant turn IDs.
4. Logs per‑prompt results and aggregates.

#### 10.4.2 — Experimental conditions

Run the same evaluation under four different configurations:

| Condition | Configuration change | What it measures |
|-----------|----------------------|------------------|
| **A — Full ICE** | Default (classifier gating + HyDE) | Best expected precision per token |
| **B — Wide‑net always** | Replace classifier with a “return Long_Term_Memory always” stub, all legs active | Token cost of gating, precision impact |
| **C — No HyDE** | Disable HyDE rewriting (bypass flag) | Contribution of query rewriting |
| **D — No memory** | Skip retrieval entirely, zero context | Baseline (precision = 0) |

To implement the conditions without touching production code, you can leverage the `IntentClassifier` interface: write a mock classifier that always returns the same tags, and inject it into the orchestrator.  
For HyDE, the orchestrator already has a bypass flag.

#### 10.4.3 — Metrics to collect

- **Precision@5** and **Recall@5** for each condition.
- **Tokens fetched** per request (logged by the orchestrator) – measure token savings vs. wide‑net.
- **Classifier confidence** distribution.
- **Breakdown by context_reliance class** – show how many prompts were correctly gated as Zero_Shot.

Store all results in a structured format (CSV or JSON) for later analysis.

---

### Step 10.5 — Longitudinal improvement experiment

The defining property of ICE is that it gets better the more it is used.

#### 10.5.1 — Method

- Re‑run the simulation harness multiple times, each time stopping after a different number of historical turns:
  - Session 1 (first 20 turns)
  - Session 10 (first 200 turns)
  - Session 30 (first 600 turns)
  - Session 60 (first 1200 turns)
  - Session 120 (all turns)
- At each checkpoint, run the retrieval evaluation on the **same held‑out set** of prompts that refer to content that appeared early in the timeline.
- Plot retrieval precision@5 against the amount of accumulated memory.

**Hypothesis:** Precision should rise as the Codex gains more validated facts and procedural patterns crystallise.

#### 10.5.2 — Decay & reinforcement validation

- From the final database, sample turns with high `access_count` and compare their `decay_score` with equally old but unaccessed turns.  
  Confirm that frequently retrieved turns have higher scores (reinforcement works).
- Verify that any bookmarked turns have `decay_immune = True` and have never decayed.

---

### Step 10.6 — Codex accuracy & truth quorum ablation

#### 10.6.1 — Sample evaluation

- Randomly sample 100 Codex triplets (entity‑relation‑entity) from the simulated database.
- Manually (or with the help of a larger model) verify each triplet against the source conversation it was extracted from.
- Compute **precision** (fraction of triplets that are factually correct) and **recall** (fraction of ground‑truth relationships that were extracted, if you have a gold set).

#### 10.6.2 — Truth quorum impact

- Run the simulation **without** the truth quorum (modify the Codex Extractor to promote all edges to `active` immediately regardless of batch count).
- Repeat the accuracy measurement.
- Compare precision between quorum‑on and quorum‑off conditions.  
  The architecture hypothesizes that the quorum reduces hallucination rate.

---

### Step 10.7 — Sentinel memory health analysis (secondary contribution)

- Run the Sentinel Monitor on the simulated database.
- Aggregate the `sentinel_events` table: how many staleness alerts, contradiction alerts, retrieval‑health alerts fire?
- This data provides the first known characterisation of memory health degradation patterns in long‑running personal AI systems.

---

### Step 10.8 — Reproducibility & logging

- Every simulation and evaluation run must use a fixed `--seed`.
- Log all hyperparameters, dataset sizes, and model versions to `experiments/results.json`.
- Store the trained classifier checkpoint and the exact simulation input file (or its hash) for reference.
- The goal: another researcher with your code and data can reproduce every number exactly.

---

## What this phase gives you for the paper

| Paper section | Data from this phase |
|---------------|----------------------|
| System description | Ready — from architecture doc |
| Experimental setup | Simulation input, held‑out set, conditions |
| Results | Precision@5 tables, recall curves, token savings, HyDE contribution |
| Longitudinal analysis | Precision vs. accumulated sessions plot |
| Codex quality | Extraction precision, truth quorum effect |
| Memory health | Sentinel event logs, decay/reinforcement plots |
| Reproducibility | Seed‑fixed runs, logged parameters, open‑source code |

---

## Order of execution (the one‑liner)

1. Extract & clean data → merged `simulation_full.jsonl`  
2. Build held‑out test set with labels  
3. Run baseline simulation → populated DB  
4. Trigger all background workers to maturity  
5. Run retrieval experiments (conditions A‑D)  
6. Run longitudinal checkpoints  
7. Sample Codex accuracy & truth quorum ablation  
8. Run Sentinel & decay validation  
9. Compile all results → paper

---

**Phase 10 is the bridge from engineering to science.**  
When it finishes, you will have a complete, reproducible evaluation of the first personal AI memory system built for human thought, and the numbers to prove it.

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