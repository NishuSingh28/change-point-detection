
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ruptures as rpt

st.set_page_config(page_title="Change Point Detection", layout="wide")
st.title("Change Point Detection in Text")
st.markdown(
    "Detect where the **topic or meaning shifts** in any text. "
    "Each paragraph is embedded into a 384-dim semantic space, "
    "and topic boundaries are found as peaks in the cosine-distance signal."
)

text_input = st.text_area(
    "Paste text (separate paragraphs with a blank line):", height=250
)
uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])
text = uploaded.read().decode("utf-8") if uploaded else text_input

st.sidebar.header("Detection settings")
sensitivity = st.sidebar.slider(
    "Sensitivity",
    min_value=0.0, max_value=2.0, value=1.0, step=0.1,
    help=(
        "0.0 = only major topic shifts (floor 0.95)\n"
        "1.0 = balanced (floor 0.85)\n"
        "2.0 = catch sub-topic shifts (floor 0.75)"
    ),
)
show_pelt = st.sidebar.checkbox("Also run PELT (for comparison)", value=False)


def split(t, min_chars=40):
    return [p.strip() for p in t.split("\n\n") if len(p.strip()) >= min_chars]


def cosine_distances(emb):
    return 1 - cosine_similarity(emb[:-1], emb[1:]).diagonal()


def sensitivity_to_floor(s):
    """Map slider [0, 2] -> absolute cosine-distance floor [0.95, 0.75].

    Calibrated against the empirical separation under MiniLM-L6-v2:
      within-topic distances ~ 0.4-0.7   (the floor must exceed this band)
      between-topic distances ~ 0.9-1.0
    Default floor 0.85 sits in the gap so true between-topic peaks fire
    while within-topic noise (even strong sub-topic shifts around 0.8) is
    suppressed.
    """
    return 0.95 - 0.10 * s


def detect_peaks(distances, sensitivity):
    floor = sensitivity_to_floor(sensitivity)
    peaks, _ = find_peaks(distances, height=floor)
    return peaks, floor


def detect_pelt(z_signal, pen=1.0):
    if len(z_signal) < 2:
        return np.array([])
    algo = rpt.Pelt(model="l2").fit(z_signal.reshape(-1, 1))
    return np.array(algo.predict(pen=pen)[:-1])


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


encoder = load_model()

if st.button("Run Detection"):
    if not text.strip():
        st.warning("Please enter or upload some text.")
    else:
        segments = split(text)
        if len(segments) < 3:
            st.warning("Need at least 3 paragraphs of substantial length.")
        else:
            embeddings = encoder.encode(segments, normalize_embeddings=True)
            distances  = cosine_distances(embeddings)
            z_signal   = (distances - distances.mean()) / (distances.std() + 1e-6)

            cps_peak, floor = detect_peaks(distances, sensitivity)
            cps_pelt = detect_pelt(z_signal) if show_pelt else np.array([])

            st.caption(f"Cosine-distance floor at this sensitivity: **{floor:.2f}**")

            rows = []
            for cp in cps_peak:
                if cp < len(segments) - 1:
                    s = float(distances[cp])
                    kind = "Major" if s > 0.95 else "Moderate" if s > 0.85 else "Minor"
                    rows.append({
                        "Index":    int(cp),
                        "Type":     kind,
                        "Strength": round(s, 3),
                        "Before":   segments[cp][:100],
                        "After":    segments[cp + 1][:100],
                    })

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detected Change Points")
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info(
                        "No change points detected at this sensitivity. "
                        "The text appears semantically uniform - try raising the sensitivity slider."
                    )
            with col2:
                st.subheader("Semantic Distance Signal")
                fig, ax = plt.subplots()
                ax.plot(distances, marker="o", color="#1f77b4", label="Cosine distance")
                ax.axhline(floor, color="grey", linestyle=":", alpha=0.7,
                           label=f"Floor ({floor:.2f})")
                for j, cp in enumerate(cps_peak):
                    ax.axvline(cp, color="red", linestyle="--",
                               label="Peak detection" if j == 0 else None)
                if show_pelt:
                    for j, cp in enumerate(cps_pelt):
                        ax.axvline(cp - 0.5, color="purple", linestyle=":",
                                   label="PELT" if j == 0 else None)
                ax.set_xlabel("Segment boundary index")
                ax.set_ylabel("Cosine distance")
                ax.legend(fontsize=8)
                st.pyplot(fig)

            st.subheader("Text with Change Markers")
            cp_set = set(int(c) + 1 for c in cps_peak)
            for i, seg in enumerate(segments):
                if i in cp_set:
                    st.markdown("---  \n**Topic shift below**")
                st.markdown(f"**Segment {i}**")
                st.write(seg)

            st.success(f"Detected {len(cps_peak)} change point(s).")
