# 03 — Data Preparation

Your data is the most important part of fine-tuning. This guide explains every supported format, how to structure your data correctly, and how to handle messy or unusual datasets.

---

## Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| CSV | `.csv` | Most common. Open with Excel or Google Sheets. |
| JSONL | `.jsonl` | One JSON object per line. Common in AI datasets. |
| JSON | `.json` | Array of objects in a single file. |
| Plain text | `.txt` | One training example per line. |
| Excel | `.xlsx` | Requires openpyxl (installed by default). |
| PDF | `.pdf` | Text is extracted automatically. Good for documents. |
| ZIP | `.zip` | Any of the above formats, zipped together. |

---

## Data Structures by Training Mode

### Standard Fine-Tuning (SFT) — Teaching the model to answer questions

This is the most common mode. Your data should have a prompt/question and the ideal response.

**Option 1 — Instruction + Output (recommended)**

```csv
instruction,output
"Summarize this article in 2 sentences.","The article discusses climate change and its effects on polar ice. Scientists warn that sea levels may rise significantly by 2100."
"Write a professional email declining a meeting.","Dear [Name], Thank you for the invitation. Unfortunately I have a prior commitment at that time and will be unable to attend. Best regards, [Your name]"
```

**Option 2 — Single text column (for language modelling)**

```csv
text
"The quick brown fox jumps over the lazy dog."
"In a galaxy far far away, there lived a young hero who dreamed of adventure."
```

**Option 3 — Chat conversations (`messages`, JSON/JSONL or Hub)**

The standard chat format used by OpenAI/ShareGPT-style datasets. The model is trained on the **last assistant reply**; earlier turns (system, user, previous assistant replies) are context. The model's own chat template is applied, so use an instruct/chat model.

```jsonl
{"messages": [{"role": "system", "content": "You are concise."}, {"role": "user", "content": "What is DNA?"}, {"role": "assistant", "content": "The molecule that carries genetic instructions."}]}
```

Roles must be `system`, `user`, `assistant` or `tool`. Conversations without an answered user turn are dropped. TRL's conversational `prompt`/`completion` layout (both columns are message lists) is accepted too and joined into one conversation.

**Tool calling (function calling).** Assistant turns may carry `tool_calls` instead of (or with) text, and the tool's result comes back in a `tool` turn. List the available functions (JSON schemas) in a `tools` column; the model's chat template puts them in the system prompt.

```jsonl
{"messages": [{"role": "user", "content": "Weather in Rome?"}, {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]}, {"role": "tool", "name": "get_weather", "content": "Sunny, 24°C"}, {"role": "assistant", "content": "It's sunny and 24°C in Rome."}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
```

- In conversations with tool calls **every assistant turn is trained** — the call (name + exact arguments) and the final answer — as separate examples. Plain chats train their last answer.
- `arguments` may be an object or a JSON string (OpenAI style); strings are decoded. Each call keeps exactly its own arguments (an argument whose value is `null` is dropped).
- Use a model whose chat template supports tools (e.g. Qwen2.5/Qwen3, Llama 3.1+, Mistral).

**Reasoning traces.** Assistant turns may include their thinking in `reasoning_content` (or `thinking`); models whose template supports it (e.g. Qwen3's `<think>…</think>`) learn it before the answer. Templates without reasoning support ignore the field.

**Images (vision-language models).** Add an `images` column (or `image` for one) and put `{"type": "image"}` parts in the message content where each image belongs — or keep plain text content and the images go into the first user turn. In a local JSONL file, images are paths **relative to the file** and must be inside its folder (CLI; the web UI uploads a single file, so use a Hub dataset there). Train with a vision-language model (e.g. Qwen2.5-VL) — see [04 — Training](04_training.md#vision-language-models).

```jsonl
{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What colour is this?"}]}, {"role": "assistant", "content": "Red."}], "images": ["img/0.png"]}
```

**JSONL equivalent of Option 1:**
```jsonl
{"instruction": "What is the speed of light?", "output": "The speed of light is approximately 299,792 kilometres per second."}
{"instruction": "Write a haiku about rain.", "output": "Drops fall on the roof / A quiet rhythm plays on / The earth drinks deeply"}
```

---

### DPO Alignment — Teaching the model to prefer good answers over bad ones

DPO training requires three columns: a prompt, a "good" answer, and a "bad" answer the model should learn to avoid.

```csv
prompt,chosen,rejected
"How do I get better at cooking?","Start with simple recipes, practice regularly, taste as you go, and don't be afraid to make mistakes.","Just watch a lot of cooking shows."
"Explain gravity.","Gravity is a force that pulls objects toward each other. The more massive an object, the stronger its gravitational pull.","Gravity makes things fall down."
```

---

### Reward Model / ORPO — Preference data

Both require the same `chosen`/`rejected` structure (ORPO also needs `prompt`):

```csv
prompt,chosen,rejected
"What is 2+2?","2+2 equals 4.","2+2 equals fish."
```

---

## How Many Examples Do You Need?

| Goal | Minimum | Recommended |
|---|---|---|
| Testing / proof of concept | 5 | 20+ |
| Noticeable style change | 50 | 200+ |
| Domain specialisation | 200 | 1,000+ |
| Strong behaviour change | 500 | 5,000+ |

More data is almost always better, but **quality matters more than quantity**. 200 excellent examples will outperform 2,000 messy ones.

---

## Column Mapping (For Custom Headers)

If your CSV has different column names, the tool will show you three dropdown menus after uploading:

```
→ Prompt/Instruction   [dropdown]
→ Chosen/Output        [dropdown]
→ Rejected/Text        [dropdown]
```

**Example:** Your file has columns called `question` and `answer` instead of `instruction` and `output`. Just:

1. Set **→ Prompt/Instruction** to `question`
2. Set **→ Chosen/Output** to `answer`
3. Click **🔄 Apply Mapping & Refresh Preview**

The preview table will update to show how the tool will read your data.

---

## Data Augmentation

If your dataset is small, you can multiply it automatically using the augmentation tools.

Go to the **📂 Data** tab, scroll to **Dataset Enhancement**, and you'll find two tools:

### Augmentation

Augmentation creates new examples by slightly modifying your existing ones. This is useful when you have fewer than 500 examples.

| Type | What it does | Example |
|---|---|---|
| `synonym` | Replaces some words with synonyms | "quick" → "fast" |
| `random_word` | Inserts or swaps random words | Adds natural variation |
| `spelling` | Adds realistic typos | Mimics human writing |

Settings:
- **Augmentation Factor** — 2× doubles your dataset, 3× triples it, etc.
- **Augmentation Type** — `synonym` is safest for most tasks

> **Warning:** Don't over-augment. 3× is a reasonable maximum. Beyond that, the synthetic examples start hurting quality.

### Quality Filter

The quality filter removes examples that are too short, too long, or low quality.

- **Min Character Length** (default: 50) — removes very short examples
- **Max Character Length** (default: 2048) — removes examples that would be truncated anyway

A good workflow: filter first, then augment.

---

## Loading from the Hugging Face Hub

Instead of uploading a file, open **…or load from the Hugging Face Hub** in the Data tab:

1. Enter the dataset ID (`owner/name`, e.g. `trl-lib/Capybara`), and a config if the dataset has several.
2. Pick the split (`train` by default) and **Max rows** (default 20,000).
3. Click **⬇️ Load from Hub**, check the preview, then click **▶ Start Training**.

Rows are streamed, so only the rows you load are downloaded. Supported layouts: `messages` (with optional `tools` and `images`/`image`), conversational `prompt`+`completion`, `text`, `instruction`+`output`, and (in DPO mode) plain-text `prompt`+`chosen`+`rejected`. Other columns are ignored. Private or gated datasets need `HF_TOKEN`. From the CLI: `python main.py train --model <id> --hf-dataset owner/name --hf-max-rows 5000`.

---

## Creating Training Data from Your Documents

No question/answer data yet? Open **…or create training data from your documents** in the Data tab:

1. Upload one or more documents: PDF, Word (`.docx`), `.txt` or `.md`.
2. Pick the **writer** — the model that writes the questions and answers:
   - **OpenAI-compatible server**: any `/v1/chat/completions` endpoint — `vllm serve`, llama.cpp's
     `llama-server`, Ollama, or a hosted API (enter its URL, model name and API key).
     `python main.py serve` starts one from a model you have.
   - **Local model**: a Hugging Face model id or folder, run in the app (a GPU helps a lot).
3. Set **pairs per chunk** and the **quality threshold**, then click **✨ Create training data**.

The text is cut into chunks of about 4,000 characters (200 characters overlap, cut at paragraph or
sentence ends). For each chunk the writer is asked for question/answer pairs based **only** on that
text, as JSON. With a threshold above 0 the writer then rates each pair from 1 to 10 and pairs
below the threshold are dropped (pairs it can't rate are dropped too); duplicate questions are
dropped as well. Set the threshold to 0 to keep everything and skip the rating calls. This is the
recipe of Meta's synthetic-data-kit.

The result is loaded as your training data (`instruction` / `output`), ready for
**▶ Start Training**, and offered as a JSONL download that also records, for each pair, the
document, the passage it was written from (`context`) and its score. Read a sample before
training: the pairs are only as good as the writer. The same JSONL trains a search / RAG
embedding model on question ↔ passage (**🔎 Embeddings** tab, see
[Training → Embedding Models](04_training.md#embedding-models-search--rag)).
**⏹ Stop** in the Training tab stops a long run and keeps the pairs written so far.

From the CLI:

```bash
LFT_SYNTH_API_KEY=sk-... python main.py synthesize --input handbook.pdf --input faq.docx \
    --server http://localhost:8000 --output synthetic.jsonl
python main.py train --model Qwen/Qwen3-0.6B --data synthetic.jsonl --output ./runs/handbook
```

---

## Cleaning Your Data

The tool automatically checks for and warns you about:

- **Empty rows** — rows where a required column is blank (removed automatically)
- **Whitespace-only rows** — rows that look empty but contain spaces (removed automatically)
- **Duplicate examples** — identical rows, including ones that differ only in upper/lower case or spacing (removed automatically; the first is kept)
- **Very long examples** — over 2048 characters (flagged)

When training starts, the log also reports **tokens per example** as the model sees them (mean, 95th percentile, max) and how many examples exceed **Max Sequence Length** and are truncated. That's the number that matters — the character counts above are only a rough guide.

All warnings appear in the **Statistics** box after uploading.

---

## ZIP Files

You can upload a ZIP containing multiple files. The tool will:

1. Extract the ZIP safely (path traversal attacks are blocked)
2. Try to load each file inside
3. Combine all examples into a single dataset

This is useful if your data is split across many CSV files.

---

## Tips for High-Quality Data

- **Be consistent.** If some outputs use formal language and others use casual language, the model gets confused. Pick a style and stick to it.
- **Cover edge cases.** Include examples of tricky or unusual questions you expect users to ask.
- **Avoid contradictions.** Don't have two examples where the same question gets two different answers.
- **Write the output the way you want the model to respond.** The model will copy your style very closely.
- **Remove duplicates before training.** Duplicate examples waste training time and can cause the model to overfit (memorise rather than learn).

---

## Next Step

→ [04 — Training](04_training.md): Configure and start your training run.
