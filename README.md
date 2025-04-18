# 🗃️ AI-Company-Documents-Classification

> Automatic classification of administrative company documents using custom word filtering and XGBoost-based supervised learning.

This project offers an AI-based solution for classifying **PDF documents** into four business-relevant categories: invoices, inventory reports, purchase orders, and shipping orders. It leverages keyword analysis, custom feature extraction, and a multi-label classification model trained on structured document text.

Data source: [Kaggle – Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset)

---

## 📌 Objectives

- Extract key textual features from raw administrative documents.
- Build a vocabulary of words that are **statistically significant per class**.
- Train a supervised learning model to **automatically predict** the document type.
- Provide a reliable and replicable pipeline for intelligent document classification.

---

## 🧠 Modeling Approach

The processing pipeline follows five main steps:

### 1. Parsing & Preprocessing
- Read raw PDF-extracted text (`company-document-text.csv`).
- Normalize text, tokenize into lowercase words.
- Count each word’s **frequency** and **document occurrence** per class.
- Save raw occurrence statistics to class-specific CSVs (e.g., `invoice_output.csv`).

### 2. Token Filtering & Vectorization
- Filter out:
  - Very **rare words** (appearing in <10% of documents).
  - **Overly common words** that occur similarly across all categories (e.g., "a").
- Produce cleaned class-wise word files (e.g., `invoice_output_nettoye.csv`).
- Compile a unified vocabulary from filtered words across all categories.

### 3. Feature Engineering
- Create document vectors based on:
  - Presence and frequency of selected vocabulary terms.
  - Document-level metadata (e.g., total word count).
- Assign binary labels for each document type.
- Save final structured training dataset as `training_data_set.csv`.

### 4. Supervised Modeling
- Use `XGBoost` wrapped in a `MultiOutputClassifier` for multi-label learning.
- Split data into training and test sets (80/20).
- Evaluate accuracy and per-class precision/recall using `classification_report`.
- Save trained model to `xgboost_modele.pkl`.

### 5. Evaluation & Export
- Vectorize new input documents with same word list.
- Predict document category using trained model.
- Output predicted label(s) such as `invoice`, `report`, `purchase order`, `shipping order`.

---

## ⚙️ Technologies Used

- **Language:** Python 3.11
- **Environment:** Jupyter Notebook / Scripts
- **Libraries:**
  - `pandas`, `csv`, `re` – data processing
  - `xgboost`, `sklearn` – supervised modeling
  - `joblib` – model persistence

---

## 🕵️ Business Context

In many companies, incoming administrative documents are processed manually and stored without classification. Automating this task allows for:

- Faster document routing and storage,
- Reduced human error,
- Better integration with automated workflows (ERP, CRM, etc.).

The model performs well on documents similar to the training dataset and can be integrated into a business document management system.

---

## 📈 Results Achieved

- End-to-end pipeline constructed: from raw text to XGBoost-based classifier.
- Accurate label predictions on held-out validation data.
- Classification performance is **very high within dataset domain**, and degrades moderately on foreign/unseen documents.

---

## 🔄 Global Pipeline

```mermaid
graph TD
    A[CSV Input (PDF Text)] --> B[Word Occurrence Analysis per Class]
    B --> C[Filter Rare/Common Words]
    C --> D[Cleaned Class Wordlists]
    D --> E[Feature Vector Construction]
    E --> F[XGBoost MultiOutput Training]
    F --> G[Model Evaluation]
    F --> H[Export Trained Model]
    H --> I[Use for Real-Time Prediction]
```

---

## 📂 Project Directory Structure

```plaintext
AI-Company-Documents-Classification/
│
├── 📄 README.md                     # Main documentation
├── 📄 requirements.txt               # Python dependencies to install
├── 📄 LICENSE                               # Project license (MIT)
├── 📄 training_data_set.csv          # Final training dataset
│
├── 📁 data/
│   ├── 📄 company-document-text.csv  # Base text data
│   └── 📁 occurences/
│       ├── 📄 invoice_output.csv
│       ├── 📄 purchase Order_output.csv
│       ├── 📄 report_output.csv
│       ├── 📄 ShippingOrder_output.csv
│       └── 📁 nettoyer/              # Cleaned word files
│           ├── 📄 invoice_output_nettoye.csv
│           └── ...
│
├── 📁 models/
│   └── xgboost_modele.pkl         # Trained model
│
├── 📁 notebooks/
│   └── main.ipynb                 # Main notebook with complete pipeline
```

---

## ▶️ Run the Project

```bash
# Clone the repository
git clone [URL_TO_REPO]

# Navigate to the project folder
cd AI-Company-Documents-Classification

# Install dependencies
pip install -r requirements.txt

# Launch training
python notebooks/main.pynb

# Run prediction
python src/predict.py
```

---

## 🔧 Future Improvements

- Apply dimensionality reduction (e.g., PCA or embeddings).
- Use deep learning models (e.g., transformer-based) to capture more context.
- Add confidence score thresholding and retraining strategies.

---

## 👨‍💻 Authors


**Bilâl Jaiel** [<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white" height="20">](https://www.linkedin.com/in/bilal-jaiel/) [<img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" height="20">](https://github.com/bilal-jaiel)

---

## 📄 License

Distributed under the MIT license. See the [LICENSE](LICENSE) file for more information.


