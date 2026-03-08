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

## Cleaning Your Data

The tool automatically checks for and warns you about:

- **Empty rows** — rows where a required column is blank (removed automatically)
- **Whitespace-only rows** — rows that look empty but contain spaces (removed automatically)
- **Duplicate examples** — identical rows (flagged with a warning; kept)
- **Very long examples** — examples over 2048 characters (flagged; they'll be truncated)

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
