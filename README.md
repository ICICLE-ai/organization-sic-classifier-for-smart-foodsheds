# organization-sic-classifier-for-smart-foodsheds

This repository contains code for training and evaluating models that classify organizations into Standard Industrial Classification (SIC) codes based on different types of descriptive text. The data used for training and evaluation is hosted on Hugging Face and should be downloaded separately.

---

## Project Structure

```
organization-sic-classifier/
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

