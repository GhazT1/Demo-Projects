# E-commerce Intelligent Review Ranking & Quality Control System

### Executive Summary
In the e-commerce landscape, high-quality User-Generated Content (UGC) is a primary driver of consumer trust. This project implements an automated end-to-end pipeline that filters out "noise"—including gibberish, profanity, and irrelevant language—and uses a **Random Forest Classifier** to rank the most helpful, high-signal reviews at the top of product pages. By automating this moderation, we ensure that "True Insight" is prioritized over "Junk," reducing manual review workload and improving the customer decision-making journey.

### The Business Problem: Review Noise
E-commerce platforms often suffer from a high volume of low-effort feedback, spam, or meaningless comments that bury insightful customer experiences.
* **The Impact:** Poor quality reviews lead to "information overload," lower customer trust, and increased bounce rates.
* **The Goal:** Build a system that identifies high-quality reviews based on linguistic density and sentiment, effectively ranking helpful content above low-signal noise.

### Methodology: The Data Pipeline
The project is structured into a modular pipeline (Preprocessing → Feature Engineering → Model Training → Deployment):

1.  **Strict Linguistic Filtering:** Multi-stage cleaning using **Spacy** and custom logic to detect:
    * **Gibberish & Profanity:** Specialized detection for English and Hinglish swear words.
    * **Language Mismatch:** Filtering out non-target languages (e.g., Hindi/Marathi) using `langdetect`.
    * **Noise Reduction:** Strict preprocessing reduced the dataset from 1,676 to 1,655 high-quality samples, effectively mitigating false positives in model training.
2.  **Advanced Feature Engineering:** We developed a proprietary "Review Score" by synthesizing:
    * **Sentiment Polarity:** Using **VADER** and **TextBlob** to quantify emotional tone.
    * **Linguistic Richness:** Calculating noun-to-total-word ratios via **POS tagging** to prioritize descriptive, noun-heavy reviews.
    * **Review Mechanics:** Analyzing review length, punctuation density, and spell-check thresholds.
3.  **Predictive Modeling:** A **Random Forest** architecture was trained to classify and rank reviews, chosen for its ability to handle non-linear relationships between linguistic features and perceived "helpfulness."

### Technical Skill Breakdown (Granular)
* **Python Libraries:** `Pandas`, `NumPy`, `Scikit-learn`, `Joblib` (Model Persistence).
* **NLP Frameworks:** `Spacy` (POS tagging & Lemmatization), `TextBlob`, `VADER`, `NLTK` (PorterStemming).
* **Feature Extraction:** `TfidfVectorizer` with custom ngram ranges, sentiment score synthesis.
* **Pipeline Engineering:** Developed `datapipeline.py` with `argparse` support for scalable, command-line execution.
* **Exploratory Data Analysis (EDA):** Advanced automated profiling using `pandas_profiling` (YData-Profiling).

### Actionable Business Recommendations
* **Strategic Sorting:** Deploy the `review_score` output as the default "Sort by Helpful" view to increase "Add to Cart" conversion rates.
* **Automated Moderation:** Use the gibberish and profanity filters as a "Pre-Publish" gate to prevent toxic content from reaching the live site.
* **Product Insights:** Flag reviews with high sentiment but low helpfulness for the product team to identify where product descriptions may be confusing to customers.

### Limitations & Next Steps
* **Data Volume:** Currently trained on ~1.6k reviews; moving to a larger dataset would allow for Transformer-based (BERT) implementations.
* **Cold-Start Problem:** New reviews lack historical "vote" data; adding a "freshness" weight to the ranking algorithm would help new, high-quality reviews gain traction faster.
* **Multimodal Analysis:** Future iterations could incorporate Computer Vision to verify if a review includes an actual product photo, a major indicator of authenticity.

---

### Installation & Usage
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the automated pipeline:**
    ```bash
    python datapipeline.py --file_name data/test.csv --model_path randomforest.joblib
    ```

### Repository Structure
* `1. Data Analysis and Preprocessing.ipynb`: Initial EDA and strict data cleaning logic.
* `2. Feature Engineering.ipynb`: Generation of linguistic and sentiment features.
* `3. Model Training.ipynb`: Random Forest training and evaluation.
* `datapipeline.py`: The production-ready script for ranking new reviews.
* `utils.py`: Core logic for language detection, profanity filtering, and feature scoring.
