# organization-sic-classifier-for-smart-foodsheds

This repository contains code for training and evaluating models that classify organizations into Standard Industrial Classification (SIC) codes based on different types of descriptive text. The data used for training and evaluation is hosted on Hugging Face and should be downloaded separately.

---

## Project Structure

<pre> ## Project Structure ``` organization-sic-classifier/ | |-- src/ | | | |-- bert/ | | |-- bert_train.py | | └-- bert_test.py | | | |-- roberta/ | | |-- roberta_train.py | | └-- roberta_test.py | | | |-- longformer/ | | |-- longformer_train.py | | └-- longformer_test.py | | | └-- gpt-4o-mini/ | └-- instructions.txt | └-- README.md ``` </pre>


---
