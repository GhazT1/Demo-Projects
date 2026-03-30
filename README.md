# 🛒 NLP Product Review Ranking & Filtering Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML-RandomForest](https://img.shields.io/badge/Model-Random%20Forest-green.svg)](https://scikit-learn.org/)

## 📝 Project Overview
In e-commerce, the sheer volume of reviews often leads to "information overload." This project provides a production-ready solution to **filter out junk reviews** (gibberish, profanity, spam) and **rank helpful reviews** using a Pairwise Comparison Machine Learning model.

### 💡 Key Value Proposition
* **Business:** Improves Customer Experience and Conversion Rates (CVR) by surfacing high-quality information.
* **Technical:** Solves the $O(n^2)$ complexity of pairwise comparisons using **Vectorized Cross-Joins** in Pandas.

---

## 🚀 Key Technical Challenges & Solutions

### 1. Performance Optimization (The "N+1" Problem)
**Challenge:** Comparing every review against every other review using nested loops was extremely slow.
**Solution:** Refactored the ranking logic to use **Vectorized Cross-Joins** (`merge(how='cross')`). This leveraged NumPy's performance, reducing ranking time by **over 90%**.

### 2. High-Signal Feature Engineering
**Challenge:** Simple word counts don't equal quality. 
**Solution:** Engineered a **Noun-Density Metric ($R_n$)**. Using **Spacy POS tagging**, I prioritized reviews that mentioned specific objects (features) over purely emotional adjectives.

### 3. Data Integrity & Guard Clauses
**Challenge:** Handling "dirty" real-world data like "Hinglish" profanity or keyboard-mash gibberish.
**Solution:** Built a multi-stage filtering class in `utils.py` that checks for language, Markov-chain gibberish probability, and multi-lingual profanity before the ML model ever sees the data.

---

## 📂 Repository Structure
* `datapipeline.py`: The main execution script (Production-ready).
* `utils.py`: Modularized `ReviewProcessor` class for NLP logic.
* `notebooks/`: Step-by-step EDA, Feature Engineering, and Model Training.
* `models/`: Pre-trained `randomforest.joblib` judge model.

## 🛠️ Setup & Usage
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm