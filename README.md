# organization-sic-classifier-for-smart-foodsheds


This repository contains code for training and evaluating models that classify organizations into Standard Industrial Classification (SIC) codes based on different types of descriptive text. The data used for training and evaluation is hosted on Hugging Face and should be downloaded separately.

- smart-foodsheds


---

## Acknowledgements
National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)

---

## How-To Guides

### Repository Clone

To get started, first clone the GitHub repository:

```
bash
git clone https://github.com/ICICLE-ai/organization-sic-classifier-for-smart-foodsheds.git
cd organization-sic-classifier-for-smart-foodsheds
```
### Create and Activate Virtual Environment
Create a virtual environment:

```
bash
python3 -m venv venv
```

Activate the virtual environment:

- On macOS/Linux:
```
bash
source venv/bin/activate
```
- On Windows:
  
```
bash
venv\Scripts\activate
```
Install all required Python packages:

```
bash
pip install -r requirements.txt
```
### Dataset

The dataset is available on Hugging Face and must be downloaded before running any training or testing script.

### Dataset Download Instructions

```
bash
git lfs install
git clone https://huggingface.co/datasets/ICICLE-AI/organization-sic-code_smart-foodsheds

After downloading, extract and place the unzipped data/ folder in the root directory (next to src/).
```
### Dataset Variants

The dataset includes multiple variants based on the source of the organization descriptions:

- gsnip: Google search snippets  
- gptsummary: GPT-4o-mini generated summaries  
- llamasummary: LLaMA 3.1–8B Instruct generated summaries  
- gsnip+gptsummary: Combined inputs of google snippets + GPT-4o-mini generated summaries
- gsnip+llamasummary: Combined inputs of google snippets + LLaMA 3.1–8B Instruct generated summaries  

Each variant includes the following splits:

- train.csv
- dev.csv
- test.csv  

(or the corresponding summary files, e.g., train_gpt_response.csv, test-llama3.18b-summary.csv, etc.)

---
## Model
Each model directory under `src/` contains separate training and testing scripts.

You only need to specify the dataset variant using the `--dataset` argument.  
Accepted options include: `gsnip`, `gptsummary`, `llamasummary`, `gsnip+gptsummary`, `gsnip+llamasummary`.


### Train

```bash
python src/bert/train_bert.py --dataset gsnip
python src/roberta/train_roberta.py --dataset gptsummary
python src/longformer/train_longformer.py --dataset gsnip+llamasummary
```
### Test

```bash
python src/bert/test_bert.py --dataset gsnip
python src/roberta/test_roberta.py --dataset gptsummary
python src/longformer/test_longformer.py --dataset gsnip+llamasummary
```
All scripts are designed to automatically handle variations in file naming and input formats.

---
## Explanation
### Project Structure

```
organization-sic-classifier-for-smart-foodsheds/
|
|-- data/
|
|-- src/
|   |
|   |-- bert/
|   |   |-- test_bert.py
|   |   └-- train_bert.py
|   |
|   |-- gpt-4o-mini/
|   |   |-- evaluation.py
|   |   |-- inference.py
|   |   └-- instructions.txt
|   |
|   |-- longformer/
|   |   |-- test_longformer.py
|   |   └-- train_longformer.py
|   |
|   |-- roberta/
|       |-- test_roberta.py
|       └-- train_roberta.py
|
|-- LICENSE
|-- README.md
|-- component.yml
|-- requirements.txt

```

---

## Citation

If you use this dataset or codebase, please cite our upcoming publication (currently under review).  
In the meantime, feel free to reference the https://github.com/ICICLE-ai/organization-sic-classifier-for-smart-foodsheds.

We will update this section with the full citation once the paper is accepted and published.

---

