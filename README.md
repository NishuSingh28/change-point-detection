# Change Point Detection in Semantic Text Streams

> A computational statistics project that adapts classical change point detection (CPD) to **textual data** using sentence embeddings, with an interactive Streamlit demo.

---

## Overview

Where does the **topic shift** in a document? This project answers that question by:

1. Embedding each paragraph into a 384-dimensional semantic space using `all-MiniLM-L6-v2` (Sentence-BERT, Reimers & Gurevych 2019).
2. Computing a 1-D dissimilarity signal: `d_i = 1 − cos(e_i, e_{i+1})`.
3. Detecting topic boundaries on that signal using three competing estimators — a **moment-based threshold**, **peak detection**, and **PELT** (Pruned Exact Linear Time, Killick et al. 2012).
4. Benchmarking the three on a controlled multi-topic corpus using precision, recall, F1, and Mean Detection Error.
5. Wrapping the pipeline in a **Streamlit web app** for interactive use.

---

## Key Finding

PELT, the textbook "best" change point method, **under-segments** this signal — not from a bug but from a **signal–cost mismatch**: PELT's kernel cost is designed for *sustained* distributional shifts, but paragraph-level topic boundaries appear as **single-index spikes** that immediately return to baseline. Simple peak detection, calibrated against the empirical between-topic versus within-topic distance separation under MiniLM, outperforms it.

> *The right detector depends on the shape of the change in the signal, not on the sophistication of the method.*

---

## Quick Start

### Run the notebook
```bash
git clone https://github.com/<your-username>/change-point-detection.git
cd change-point-detection
pip install -r requirements.txt
jupyter notebook computational_statistics.ipynb
```

### Run the interactive Streamlit app
```bash
streamlit run app.py
```
Then open <http://localhost:8501> in your browser, paste text (or upload a `.txt` from `sample_inputs/`), and click **Run Detection**. Adjust the **sensitivity slider** in the sidebar to dial how many boundaries are surfaced.

---

## Repository Structure

```
change-point-detection/
├── README.md                         <- this file
├── requirements.txt                  <- pinned dependencies
├── app.py                            <- Streamlit demo app
├── computational_statistics.ipynb    <- full analysis notebook
├── sample_inputs/                    <- test texts that exercise edge cases
│   ├── 01_three_topics.txt
│   ├── 02_two_topics.txt
│   ├── 03_five_topics.txt
│   ├── 04_subtle_drift.txt
│   ├── 05_uniform_single_topic.txt
│   ├── 06_too_short.txt
│   └── 07_duplicate_paragraphs.txt
├── screenshots/                      <- add demo screenshots here
└── LICENSE
```

---

## Methodology

```
Raw text
   │
   ▼
[1] Segment into paragraphs
   │
   ▼
[2] Embed each segment           e_i ∈ ℝ^384   (MiniLM-L6-v2)
   │
   ▼
[3] Dissimilarity signal         d_i = 1 − cos(e_i, e_{i+1})
   │
   ▼
[4] Detect change points         {τ_1, …, τ_K}
   │   ├── Threshold (μ + σ)        — moment-based baseline
   │   ├── Peak detection           — primary method (absolute floor)
   │   └── PELT (RBF kernel)        — comparative state-of-the-art
   │
   ▼
[5] Evaluate                     Precision, Recall, F1, MDE
```

### Statistical assumption

Within a topic the distance signal is locally stationary; at a boundary both mean and variance shift:

> *d_i ~ N(μ_k, σ²_k) for i in segment k*

### PELT objective

> minimize over τ_{1:K} of Σ C(d_{τ_k:τ_{k+1}}) + β·K
>
> where C is a per-segment cost (RBF kernel here) and β is a complexity penalty.

### Peak-detection calibration

The sensitivity slider in the app maps to an **absolute** cosine-distance floor in [0.75, 0.95]. The floor is empirically calibrated against the observed separation under MiniLM:
- within-topic distances ≈ 0.4 – 0.7
- between-topic distances ≈ 0.9 – 1.0

A default floor of **0.85** sits in the gap so true between-topic peaks fire while within-topic noise is suppressed.

---

## Results — Method Comparison (controlled 9-paragraph corpus)

| Method                    | Precision | Recall |   F1  |  MDE  | Notes |
|---------------------------|----------:|-------:|------:|------:|-------|
| Peak detection (abs floor)| 1.000     | 1.000  | 1.000 | 0.00  | Best on this signal class |
| Threshold (μ + σ)         | 1.000     | 1.000  | 1.000 | 0.00  | Strong baseline |
| PELT (RBF kernel)         | 0.000     | 0.000  | 0.000 | 2.00+ | Under-segments single-spike boundaries |

See the notebook (section 14) for full discussion of why PELT loses here and where it *would* win (longer signals, sustained regime changes).

---

## Tech Stack

- **Embeddings:** [`sentence-transformers`](https://www.sbert.net/) — `all-MiniLM-L6-v2`
- **Change point detection:** [`ruptures`](https://centre-borelli.github.io/ruptures-docs/) (PELT), `scipy.signal.find_peaks`
- **Numerics & data:** `numpy`, `scipy`, `pandas`, `scikit-learn`
- **Visualization:** `matplotlib`
- **Web demo:** `streamlit`

---

## Limitations & Future Work

- The benchmark corpus is small and curated; real-world streams (news, dialogue) have soft, gradual transitions.
- MiniLM is general-purpose — domain-specific encoders would tighten the within/between-topic separation.
- The PELT penalty is set empirically; future work: BIC / modified BIC selection.
- A **hybrid detector** — peak detection to seed candidates, PELT-style windowed validation around each — would combine the strengths of both methods.

---

## References

- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *Optimal Detection of Changepoints With a Linear Computational Cost.* JASA 107(500), 1590–1598.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Truong, C., Oudre, L., & Vayatis, N. (2020). *Selective review of offline change point detection methods.* Signal Processing 167, 107299.
- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection.* arXiv:0710.3742.

---

## Author

**Himanshu Kumar** — Computational Statistics coursework project.

## License

MIT — see [LICENSE](LICENSE).
