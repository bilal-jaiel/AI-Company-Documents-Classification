<div align="center">

# Company Document Classification

Sorting invoices, purchase orders, shipping orders and stock reports from their raw text,<br>
with a hand-built class-aware vocabulary and an XGBoost classifier.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-EB5E28)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

</div>

<br>

| | |
|---|---|
| Data | 2,676 business documents (text extracted from PDFs), 4 classes |
| Approach | Per-class word statistics, frequency-based vocabulary filtering, bag-of-words features, XGBoost |
| Result | 536 / 536 test documents correctly classified (see the caveat below) |
| Highlight | End-to-end text classification built with the standard library and pandas, no NLP library |

---

## Contents

- [Data](#data)
- [Method](#method)
- [Results](#results)
- [Getting started](#getting-started)
- [Limitations](#limitations)

---

## Data

[Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset) (Kaggle), stored in [`data/company-document-text.csv`](data/company-document-text.csv) with three columns: `text`, `label` and `word_count`.

| Label | Documents |
|---|---:|
| invoice | 830 |
| purchase Order | 830 |
| ShippingOrder | 809 |
| report | 207 |

Documents are short (median 63 words) and follow a small number of templates; products, customers and addresses come from the Northwind sample database. This matters when reading the results.

## Method

The full pipeline is in [`notebooks/main.ipynb`](notebooks/main.ipynb), committed with its outputs.

| Step | Details |
|---|---|
| 1. Occurrence tables | Lower-case tokenisation; for each class and each word, total frequency and number of documents containing it |
| 2. Rare words | A word is dropped unless it appears in at least 10 % of the documents of some class (2,998 words removed) |
| 3. Common words | A word is dropped if it appears in all four classes with document counts within ± 20 % of each other, so it does not discriminate (78 words removed) |
| 4. Features | The top 800 remaining words of each class are merged into a 212-word vocabulary; each document becomes a vector of word counts plus its length |
| 5. Model | `MultiOutputClassifier(XGBClassifier)`, one binary output per class, trained on 80 % of the documents (random split, seed 42) |
| 6. Inference | A helper vectorises any new text with the same vocabulary and returns the predicted type |

## Results

On the 536 held-out documents, every document is classified correctly: precision, recall and F1 are 1.00 for all four classes.

This reflects how regular the dataset is rather than the difficulty of document classification in general. Each document type comes from one template with very distinctive words (*ship*, *invoice*, *stock*, *units in stock*). The project is best read as a clean, end-to-end pipeline built without NLP libraries.

## Getting started

```bash
git clone https://github.com/bilal-jaiel/AI-Company-Documents-Classification.git
cd AI-Company-Documents-Classification
pip install -r requirements.txt
cd notebooks
jupyter notebook main.ipynb   # run all cells
```

The notebook writes the occurrence tables, the training matrix and the trained model to `outputs/`, which git ignores. Its last cell shows how to classify a new document from raw text.

```
├── data/
│   └── company-document-text.csv   Kaggle dataset (text, label, word_count)
├── notebooks/
│   └── main.ipynb                  full pipeline, executed, with outputs
└── requirements.txt
```

## Limitations

- Template data: a perfect score on documents from the same templates says little about real, heterogeneous documents (scans, other layouts, other languages). Testing on another source is the meaningful next step.
- The vocabulary is chosen on the full dataset, so word filtering sees the labels of the test documents. It should be built on the training split only.
- Each document has exactly one type, so a single multi-class classifier (or a linear model on TF-IDF features as a baseline) would be simpler than a multi-output model.
- PDF text extraction is not included: the pipeline starts from already-extracted text.

---

<div align="center">
<sub>Bilâl Jaiel · <a href="https://github.com/bilal-jaiel">GitHub</a> · <a href="https://www.linkedin.com/in/bilal-jaiel/">LinkedIn</a></sub>
</div>
