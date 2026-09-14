# DRISHTI — Explainable AI for Diabetic Retinopathy Screening

**Smart India Hackathon 2026**

DRISHTI is an AI-powered screening tool that analyzes retinal fundus images to detect and grade Diabetic Retinopathy (DR), designed for use by non-specialist health workers (e.g. ASHA workers) in low-resource, rural settings where ophthalmologists are hard to reach.

Upload a fundus image → get an instant 5-level DR grade, a confidence score, and a Grad-CAM heatmap explaining which regions of the retina influenced the prediction.

🔗 **Live Demo:** https://drishti-dr-screening.streamlit.app/
💻 **Repository:** https://github.com/gauravkvbrss6558-spec/drishti-dr-screening

---

## How It Works

1. Upload a retinal fundus photo
2. Image is quality-checked and preprocessed (resize, border-crop, Ben Graham contrast enhancement)
3. EfficientNet-B0 model classifies the image into one of 5 DR severity stages
4. Grad-CAM generates a heatmap showing which regions influenced the prediction
5. App returns a risk level, confidence score, and referral recommendation

**Severity Scale:** Grade 0 – No DR · Grade 1 – Mild · Grade 2 – Moderate · Grade 3 – Severe · Grade 4 – Proliferative DR

---

## Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | 86.6% |
| Quadratic Weighted Kappa (QWK) | 0.934 |

**Architecture:** EfficientNet-B0 (two-stage transfer learning: frozen backbone → full fine-tune, with class-weighted loss, label smoothing, and test-time augmentation)

**Datasets:**
- [APTOS-2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019) — training
- [EyePACS + APTOS + Messidor (combined)](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy) — training
- [IDRiD](https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset) — used only for Grad-CAM explainability validation, not for training

---

## How to Run

**Requirements:** Python 3.9+, pip

```bash
# Clone the repository
git clone https://github.com/gauravkvbrss6558-spec/drishti-dr-screening.git
cd drishti-dr-screening

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Tech Stack

Python · PyTorch · OpenCV · EfficientNet-B0 · Grad-CAM · Streamlit · Kaggle (GPU training) · Multilingual UI

---

## References

This project builds on the following research:

1. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.
   https://proceedings.mlr.press/v97/tan19a.html

2. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV 2017.
   https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html

3. Gulshan, V., Peng, L., Coram, M., et al. (2016). *Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs*. JAMA, 316(22), 2402–2410.
   https://jamanetwork.com/journals/jama/fullarticle/2588763

**Related work** (architectural variants explored in literature; not the exact implementation used here):

4. Ghadai, K. K., Nayak, S. K., & Senapati, B. R. (2026). *A clinically interpretable deep learning pipeline for diabetic retinopathy classification using EfficientNet, advanced data augmentation and GradCAM*. BMC Ophthalmology, 26(1), 406.
   https://doi.org/10.1186/s12886-026-04897-4
   *(Uses EfficientNet-B1 + CLAHE + focal loss; DRISHTI uses EfficientNet-B0 + Ben Graham preprocessing)*

---

## Disclaimer

DRISHTI is an AI-assisted screening tool intended to **support, not replace, clinical judgment**. Highlighted regions in the heatmap indicate areas the model weighted most heavily (e.g. possible microaneurysms, hemorrhages, or exudates). Always have a qualified ophthalmologist review before treatment decisions.
