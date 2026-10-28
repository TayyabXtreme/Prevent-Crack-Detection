# 🛣️ Pavement Crack Detector

A Streamlit web app that classifies road and concrete surface photos as **cracked** or **clear**. It uses the best of two deep learning models (CNN vs Vision Transformer) trained in `../pavement_crack_detection.ipynb`.

**Features:**
- **Demo gallery:** 15 sample images with known labels. Tick any of them to see the prediction next to the correct answer.
- **Upload any image format:** JPG, PNG, WEBP, HEIC (iPhone), TIFF, BMP, GIF, AVIF, JPEG 2000, as well as grayscale, CMYK, 16-bit and transparent images. Each one is automatically converted to 8-bit RGB and resized to the model's input size. Files that aren't images are skipped with a warning.
- Camera input, adjustable crack threshold, confidence cards, CSV download, and a CNN vs ViT performance tab.

```
crack_detection_app/
├── app.py                 # Streamlit UI
├── utils.py               # model loading, any-format → RGB conversion, preprocessing (same as training)
├── demo_images/           # sample images for the demo tab (see CREDITS.md)
├── model/
│   ├── crack_model.keras  # ← from the notebook export
│   └── model_info.json    # ← from the notebook export
├── requirements.txt
├── Dockerfile             # optional: Hugging Face Spaces / Render / any Docker host
└── .streamlit/config.toml # theme
```

## 1. Train and export the model
1. Open **Google Colab** → *File → Upload notebook* → `pavement_crack_detection.ipynb`, then choose *Runtime → Change runtime type → **T4 GPU***. The notebook downloads the dataset itself with `kagglehub` (`oluwaseunad/concrete-and-pavement-crack-images`).
2. Click *Runtime → **Run all***. The last cell downloads `crack_model_export.zip` to your computer. Allow the download if your browser asks.
3. Unzip it and copy `crack_model.keras` and `model_info.json` into `crack_detection_app/model/`.

## 2. Run locally with a venv (Windows)
```powershell
cd crack_detection_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. On macOS or Linux, activate with `source venv/bin/activate` instead.

> **Version match:** `model_info.json` records the TensorFlow version the model was trained with. If loading fails, set the `tensorflow` / `tensorflow-cpu` versions in `requirements.txt` to that version.

## 3. Deploy to Streamlit Community Cloud (free)
1. Push the `crack_detection_app` folder to a **GitHub** repo, including `model/crack_model.keras` (a few MB).
   ```powershell
   git init
   git add .
   git commit -m "Pavement crack detector"
   git branch -M main
   git remote add origin https://github.com/<your-user>/crack-detector.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io → **Create app** → choose the repo, branch `main`, and main file `app.py`.
3. Under **Advanced settings**, choose Python **3.12** → **Deploy**. The first build takes a few minutes while TensorFlow installs.

## Alternative: Docker (Hugging Face Spaces, Render, etc.)
```bash
docker build -t crack-detector .
docker run -p 8501:8501 crack-detector
```
On **Hugging Face Spaces**, create a Space with the **Docker** SDK and push this folder. The Dockerfile already serves on port 8501, so add `app_port: 8501` to the Space's README header.
