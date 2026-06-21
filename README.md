# Change Point Detection in Semantic Text Streams

A Computational Statistics project that adapts classical Change Point Detection (CPD) techniques to textual data using transformer-based sentence embeddings and semantic similarity analysis.

---

## Overview

Identifying where a document changes topic is a common challenge in text analytics, document segmentation, and information retrieval. This project investigates whether traditional change point detection methods can be applied to semantic text representations.

The workflow consists of:

1. Converting each paragraph into a dense semantic embedding using the MiniLM Sentence-BERT model.
2. Measuring semantic distance between consecutive paragraphs using cosine distance.
3. Constructing a one-dimensional change signal from those distances.
4. Detecting topic boundaries using multiple change point detection techniques.
5. Evaluating detection performance on a controlled multi-topic corpus.

An interactive Streamlit application allows users to upload text and visualize detected topic transitions in real time.

---

## Key Insight

The strongest result of this project is that the most sophisticated method was not the most effective.

PELT (Pruned Exact Linear Time), a widely used state-of-the-art change point detection algorithm, consistently under-segmented the semantic distance signal. The reason was not implementation error but a mismatch between the signal structure and the assumptions of the method.

Topic transitions in text typically appear as isolated spikes in semantic distance that immediately return to baseline. PELT is designed to detect sustained distributional shifts across longer segments and therefore tends to ignore these short-lived changes.

> **Takeaway:** The effectiveness of a change point detector depends on the characteristics of the signal, not solely on algorithmic complexity.

---

## Methodology

```text
Raw Text
   │
   ▼
Paragraph Segmentation
   │
   ▼
MiniLM Embeddings (384 dimensions)
   │
   ▼
Cosine Distance Signal
   │
   ▼
Change Point Detection
   ├── Threshold Detection
   ├── Peak Detection
   └── PELT
   │
   ▼
Performance Evaluation
````

### Step 1: Text Segmentation

Input documents are divided into paragraphs, with each paragraph treated as an individual semantic unit.

### Step 2: Semantic Embeddings

Each paragraph is embedded into a 384-dimensional vector space using the `all-MiniLM-L6-v2` Sentence-BERT model.

### Step 3: Semantic Distance Signal

For every pair of consecutive paragraphs, cosine distance is computed:

```text
Distance(i) = 1 - CosineSimilarity(Paragraph_i, Paragraph_(i+1))
```

This produces a one-dimensional signal where larger values indicate stronger semantic shifts.

### Step 4: Change Point Detection

Three competing methods are applied:

#### 1. Threshold-Based Detection

A paragraph boundary is flagged when the semantic distance exceeds:

```text
Threshold = Mean + Standard Deviation
```

#### 2. Peak Detection

Local maxima in the distance signal are identified using SciPy's peak-finding algorithm and filtered using an empirically calibrated distance threshold.

#### 3. PELT

The PELT algorithm from the `ruptures` library is applied using an RBF kernel cost function to identify statistically significant segmentation points.

### Step 5: Evaluation

Performance is measured using:

* Precision
* Recall
* F1 Score
* Mean Detection Error (MDE)

---

## Statistical Assumption

Within a topic segment, semantic distances are assumed to be relatively stable. Topic boundaries introduce abrupt changes in the distribution of distances, resulting in detectable peaks within the signal.

---

## Results

### Controlled Multi-Topic Corpus

| Method              | Precision | Recall | F1 Score | Mean Detection Error |
| ------------------- | --------- | ------ | -------- | -------------------- |
| Peak Detection      | 1.00      | 1.00   | 1.00     | 0.00                 |
| Threshold Detection | 1.00      | 1.00   | 1.00     | 0.00                 |
| PELT (RBF Kernel)   | 0.00      | 0.00   | 0.00     | > 2.00               |

### Interpretation

Peak Detection and Threshold Detection successfully captured all topic boundaries in the benchmark corpus.

PELT failed because topic transitions produced isolated spikes rather than sustained regime changes, violating the assumptions underlying its cost function.

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/NishuSingh28/change-point-detection.git
cd change-point-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Analysis Notebook

```bash
jupyter notebook computational_statistics.ipynb
```

### Launch the Streamlit Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

in your browser.

---

## Repository Structure

```text
change-point-detection/
├── README.md
├── requirements.txt
├── app.py
├── computational_statistics.ipynb
├── sample_inputs/
│   ├── 01_three_topics.txt
│   ├── 02_two_topics.txt
│   ├── 03_five_topics.txt
│   ├── 04_subtle_drift.txt
│   ├── 05_uniform_single_topic.txt
│   ├── 06_too_short.txt
│   └── 07_duplicate_paragraphs.txt
└── LICENSE
```

---

## Technology Stack

### NLP & Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### Change Point Detection

* ruptures
* scipy.signal.find_peaks

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Visualization

* Matplotlib

### Deployment

* Streamlit

---

## Limitations

* Evaluation was conducted on a small curated benchmark corpus.
* Topic transitions in real-world documents may be gradual rather than abrupt.
* MiniLM is a general-purpose encoder and may not be optimal for domain-specific text.
* PELT hyperparameters were selected empirically rather than through formal model selection criteria.

---

## Future Work

Potential extensions include:

* Bayesian Online Change Point Detection for streaming text.
* Domain-specific embedding models.
* Adaptive threshold estimation.
* Hybrid methods combining peak detection with local statistical validation.
* Evaluation on longer documents, news streams, and conversational data.

---

## References

1. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost.*
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT Networks.*
3. Truong, C., Oudre, L., & Vayatis, N. (2020). *Selective Review of Offline Change Point Detection Methods.*
4. Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection.*

---

## Author

**Nishu Kumari Singh**

M.S. Data Science, Arizona State University

---

## License

This project is licensed under the MIT License.
