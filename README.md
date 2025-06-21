# organization-sic-classifier-for-smart-foodsheds


This repository contains code for training and evaluating models that classify organizations into Standard Industrial Classification (SIC) codes based on different types of descriptive text. The data used for training and evaluation is hosted on Hugging Face and should be downloaded separately.

---

## Project Structure

```
organization-sic-classifier-for-smart-foodsheds/
|
|-- src/
|   |
|   |-- BERT/
|   |   |-- bert_train.py
|   |   └-- bert_test.py
|   |
|   |-- RoBERTa/
|   |   |-- roberta_train.py
|   |   └-- roberta_test.py
|   |
|   |-- Longformer/
|   |   |-- longformer_train.py
|   |   └-- longformer_test.py
|   |
|   └-- GPT-4o-mini/
|       |-- run_gpt4o_instructions.txt
|       |-- inference.py
|       └-- evaluation.py
|
└-- README.md
```

---

## Dataset

The dataset is available on Hugging Face and must be downloaded before running any training or testing script.

### Dataset Download Instructions

```bash
git lfs install
git clone https://huggingface.co/datasets/ICICLE-AI/organization-sic-code_smart-foodsheds

After downloading, extract and place the unzipped data/ folder in the root directory (next to src/).

### Dataset Variants

The dataset includes multiple variants based on the source of the organization descriptions:

- **Google_snippets**: Google search snippets  
- **GPT-generated-summaries**: GPT-4o-mini generated summaries  
- **LLaMA-generated-summaries**: LLaMA 3.1–8B Instruct generated summaries  
- **Google_snippets+GPT-generated-summaries** / **Google_snippets+LLaMA-generated-summaries**: Combined inputs of snippet + generated summary  

Each variant includes the following splits:

- `train.csv`
- `dev.csv`
- `test.csv`  

(or the corresponding summary files, e.g., `train_gpt_response.csv`, `test-llama3.18b-summary.csv`, etc.)

---




