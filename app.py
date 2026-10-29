
import pandas as pd
import streamlit as st

from utils import DEMO_DIR, MODEL_PATH, load_info, load_model, open_image, predict

st.set_page_config(page_title="Pavement Crack Detector", page_icon="🛣️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.result-card {border-radius: 12px; padding: 0.8rem 1rem; margin-top: 0.4rem; color: #fff;}
.result-card.crack {background: linear-gradient(135deg, #d64545, #a82828);}
.result-card.clear {background: linear-gradient(135deg, #2e9e5b, #1e7443);}
.result-card .label {font-size: 1.15rem; font-weight: 700;}
.result-card .conf {font-size: 0.9rem; opacity: 0.9;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()


info = load_info()
metrics = info.get("metrics", {}).get(info["model_name"], {})

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🛣️ Crack Detector")
    st.caption("Deep learning inspection of pavement and concrete surfaces.")
    st.divider()
    st.subheader("Model")
    st.write(f"**Architecture:** {info['model_name']}")
    st.write(f"**Input size:** {info['img_size']}×{info['img_size']} RGB")
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("F1", f"{metrics['f1']:.2%}")
        c1.metric("Recall", f"{metrics['recall']:.2%}")
        c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.divider()
    threshold = st.slider("Crack threshold", 0.05, 0.95, float(info["threshold"]), 0.05,
                          help="An image is flagged as cracked when P(crack) is at or above this value. "
                               "Lower it to catch more cracks, at the cost of more false alarms.")
    st.divider()
    st.caption("ANN Lab · CNN vs Vision Transformer")

# ---------- Header ----------
st.title("Pavement Crack Inspection")
st.write("Upload photos of road or concrete surfaces, or take one with your camera. "
         "The model flags each image as **cracked** or **clear**.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH}`. Run the notebook, download `crack_model_export.zip`, "
             "and unzip `crack_model.keras` + `model_info.json` into the `model/` folder.")
    st.stop()

model = get_model()

tab_demo, tab_upload, tab_camera, tab_perf = st.tabs(
    ["🖼️ Try demo images", "📁 Upload images", "📷 Camera", "📊 Model performance"])


def show_results(named_images, key, expected=None):
    """named_images: [(name, PIL image)]; expected: optional list of true labels (True = crack)."""
    if not named_images:
        return
    names, images = zip(*named_images)
    with st.spinner(f"Analysing {len(images)} image(s)…"):
        probs = predict(model, list(images), info["img_size"])
    preds = probs >= threshold

    m = st.columns(4 if expected else 3)
    m[0].metric("Images analysed", len(probs))
    m[1].metric("Cracks detected", int(preds.sum()))
    m[2].metric("Clear surfaces", int((~preds).sum()))
    if expected:
        m[3].metric("Model correct", f"{sum(p == e for p, e in zip(preds, expected))} / {len(probs)}")
    st.divider()

    cols = st.columns(3)
    for i, (name, img, p, crack) in enumerate(zip(names, images, probs, preds)):
        conf = p if crack else 1 - p
        truth = ""
        if expected:
            ok = crack == expected[i]
            truth = f" · Actual: {'Crack' if expected[i] else 'No crack'} {'✔️' if ok else '❌'}"
        with cols[i % 3]:
            st.image(img, caption=name, width="stretch")
            st.markdown(
                f"<div class='result-card {'crack' if crack else 'clear'}'>"
                f"<div class='label'>{'⚠️ Crack detected' if crack else '✅ No crack'}</div>"
                f"<div class='conf'>Confidence {conf:.1%} · P(crack) = {p:.3f}{truth}</div></div>",
                unsafe_allow_html=True)
            st.progress(float(p), text="Crack probability")

    report = pd.DataFrame({"image": names, "crack_probability": probs.round(4),
                           "prediction": ["Crack" if c else "No crack" for c in preds]})
    if expected:
        report["actual"] = ["Crack" if e else "No crack" for e in expected]
    with st.expander("Results table"):
        st.dataframe(report, width="stretch", hide_index=True)
    st.download_button("⬇️ Download results (CSV)", report.to_csv(index=False), "crack_results.csv",
                       "text/csv", key=f"download_{key}")


with tab_demo:
    demo_files = sorted(p for p in DEMO_DIR.glob("*") if p.is_file() and p.suffix.lower() != ".md") \
        if DEMO_DIR.exists() else []
    if not demo_files:
        st.info("No demo images found in `demo_images/`.")
    else:
        st.write("Sample pavement images with known labels. Tick the ones to test, and the model's results appear below.")
        select_all = st.toggle("Select all")
        grid = st.columns(6)
        chosen = []
        for i, path in enumerate(demo_files):
            img, desc = open_image(path.read_bytes())
            with grid[i % 6]:
                st.image(img, width="stretch")
                if st.checkbox(path.name, value=select_all, key=f"demo_{path.name}_{select_all}", help=desc):
                    chosen.append((path, img))
        st.divider()
        if chosen:
            show_results([(p.name, img) for p, img in chosen], key="demo",
                         expected=[not p.name.startswith("no_crack") for p, _ in chosen])
        else:
            st.caption("👆 Select at least one image.")

with tab_upload:
    files = st.file_uploader("Choose images (any format: JPG, PNG, WEBP, HEIC, TIFF, BMP, GIF, AVIF…)",
                             accept_multiple_files=True)
    if files:
        good, bad = [], []
        for f in files:
            img, desc = open_image(f.getvalue())
            if img is None:
                bad.append(f.name)
            else:
                good.append((f"{f.name} ({desc})", img))
        if bad:
            st.warning("Skipped (not a readable image): " + ", ".join(bad))
        show_results(good, key="upload")
    else:
        st.info("Drop one or more images above to start. Every image is converted to RGB and resized "
                f"to {info['img_size']}×{info['img_size']} automatically.")

with tab_camera:
    shot = st.camera_input("Point the camera at the surface")
    if shot:
        img, _ = open_image(shot.getvalue())
        show_results([("camera.jpg", img)], key="camera")

with tab_perf:
    all_metrics = info.get("metrics")
    if not all_metrics:
        st.info("No metrics found in `model/model_info.json`.")
    else:
        st.subheader("Test-set comparison")
        table = pd.DataFrame(all_metrics).T
        if "params" in info:
            table["parameters"] = pd.Series(info["params"])
        st.dataframe(table.style.format({c: "{:.4f}" for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                                        | {"parameters": "{:,.0f}"})
                     .highlight_max(subset=["accuracy", "precision", "recall", "f1", "roc_auc"], color="#2e9e5b55"),
                     width="stretch")
        st.bar_chart(table[["accuracy", "precision", "recall", "f1"]].T, stack=False)

        if "confusion_matrix" in info:
            st.subheader(f"Confusion matrix: {info['model_name']}")
            cm = pd.DataFrame(info["confusion_matrix"],
                              index=[f"Actual {c}" for c in info["class_names"]],
                              columns=[f"Predicted {c}" for c in info["class_names"]])
            st.dataframe(cm, width="content")
        if "split_sizes" in info:
            st.caption("Data split: " + " · ".join(f"{k} {v:,}" for k, v in info["split_sizes"].items())
                       + f" · trained {info.get('trained_on', '')} with TF {info.get('tensorflow_version', '?')}")
