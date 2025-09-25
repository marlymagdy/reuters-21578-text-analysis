# reuters-21578-text-analysis
This project analyzes the **Reuters-21578 dataset**, focusing on **text similarity and dissimilarity** as well as **document classification**.  
We explore the relationships between documents using **cosine similarity, Jaccard distance, and Euclidean distance**, visualize results with heatmaps, and classify articles using a **Linear Support Vector Classifier** with SMOTE to handle imbalanced data.

## 📂 Dataset Overview
The **Reuters-21578 dataset** is a well-known benchmark collection of news articles widely used in **text classification** and **information retrieval** research.  
It was created in 1996 as a cleaned and standardized version of the earlier Reuters-22173 collection, ensuring less ambiguous formatting and consistent documentation.  

- **Total documents:** 21,578  
- **File distribution:** 22 SGML files (21 with 1000 docs each, 1 with 578 docs)  
- **Source:** Reuters and Carnegie Group, released for academic research  

Each article is enclosed within SGML tags with explicit attributes:  
- **TOPICS**: {YES, NO, BYPASS}  
- **LEWISSPLIT**: {TRAINING, TEST, NOT-USED}  
- **CGISPLIT**: {TRAINING-SET, PUBLISHED-TESTSET}  
- **OLDID**: Original Reuters-22173 ID  
- **NEWID**: New Reuters-21578 ID  


### 🏷 Document Structure
Articles are organized with SGML tags that mark important fields such as:  

- `<DATE>` → publication date  
- `<TOPICS>`, `<PLACES>`, `<PEOPLE>`, `<ORGS>`, `<EXCHANGES>`, `<COMPANIES>` → categorical attributes  
- `<TITLE>` and `<BODY>` → text content  
- `<AUTHOR>` and `<DATELINE>` → optional metadata  

> ⚠ **Note:**  
If `TOPICS = YES` but no topics are provided, articles are considered **negative examples** for all 135 topics.  
If `TOPICS = NO` and no topics exist → these are excluded from the dataset.  
Filtered dataset includes only entries that have topics or explicitly confirm the presence of topics.  

---

## 📊 Similarity & Dissimilarity Analysis

To explore document relationships, I computed **similarity and dissimilarity matrices** based on **topics** and **places** attributes.

### 🔹 Similarity Matrix
- Constructed using **`cosine_similarity`** from `sklearn.metrics.pairwise`.  
- Reveals how close documents (or attributes) are to each other.  
- **High values (≈1.0):** strong similarity → documents share many common features.  
- **Low values (≈0.0):** weak similarity → documents share few or no features.  

### 🔹 Dissimilarity Matrix
- Derived as: `1 - similarity`  
- Constructed using **`pairwise_distances`** from `sklearn.metrics.pairwise`.  
- Represents how different documents are.  
- Useful for **clustering** (grouping similar documents) and **classification**.  


### 🎨 Heatmap Visualization
A heatmap of the dissimilarity matrix was created:  

- **Darker colors:** higher dissimilarity (more different)  
- **Lighter colors:** lower dissimilarity (more similar)  
- Patterns and clusters can be observed, providing insights into **document structure and grouping**.  

---

## 🔎 Example: Jaccard Distance

The **Jaccard distance** measures dissimilarity between two sets of features:  
- **Range:** 0 → 1  
  - `0 = maximum similarity` (identical sets)  
  - `1 = maximum dissimilarity` (no overlap in sets)  

**Case: "doc_0" vs "doc_2"**  
- Features: `brazil, canada, uk, usa`  
- Jaccard Distance = **1.0** → fully dissimilar  

**Feature breakdown:**  
- `brazil`: (0,0) → same → distance = 0  
- `canada`: (0,1) → different → distance = 1  
- `uk`: (0,0) → same → distance = 0  
- `usa`: (0,1) → different → distance = 1  

➡ Since no features are shared, the overall distance = **1.0**.  
This shows **"doc_0" and "doc_2" are completely dissimilar.**  

---

## 🔹 Data Transformation
- Segregated dataset into **training** and **testing** based on `lewissplit`.  
- Applied **TF-IDF vectorization** on the document `Body` text.  
- Converted labels to binary:  
  - `1` → *earn*  
  - `0` → *others*  
- Handled **imbalanced data** using **SMOTE** to oversample minority class in training data.  

---

## 🤖 Modelling
- Built a **Linear Support Vector Classifier (LinearSVC)**, which achieved **99% accuracy, precision, recall, and F1-score** on the training set, demonstrating excellent classification performance after balancing the dataset with SMOTE.  

---

