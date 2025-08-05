# ─────────────────────────────────────────────────────────────
#  CarePath AI Foundation – Parkinson's Voice Classifier API
#  Supports: WebM/Opus or WAV uploads →
#            extracts 16 voice features → Random-Forest at
#            threshold 0.63 → JSON result
# ─────────────────────────────────────────────────────────────

import os, uuid, tempfile, warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import parselmouth

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── 1  CONFIG ────────────────────────────────────────────────
MODEL_PATH  = "rf_model.pkl"
THRESHOLD   = 0.63

FEATURE_COLS = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
    'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
    'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR'
]

# ── 2  APP SETUP ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, max_age=600)
rf_model = joblib.load(MODEL_PATH)

# ── 3  FEATURE EXTRACTION ───────────────────────────────────
def extract_features(path_wav: str) -> pd.DataFrame:
    snd = parselmouth.Sound(path_wav)
    feats = {}

    # Pitch metrics
    pitch = snd.to_pitch()
    feats['MDVP:Fo(Hz)']  = np.nanmean(pitch.selected_array['frequency'])
    feats['MDVP:Fhi(Hz)'] = np.nanmax(pitch.selected_array['frequency'])
    feats['MDVP:Flo(Hz)'] = np.nanmin(pitch.selected_array['frequency'])

    # Jitter/Shimmer (Praat CC)
    pp = snd.to_point_process_cc()
    try:
        feats['MDVP:Jitter(%)']   = snd.to_jitter_local()*100
        feats['MDVP:Jitter(Abs)'] = snd.to_jitter_local_absolute()
        feats['MDVP:RAP']         = snd.to_jitter_rap()
        feats['MDVP:PPQ']         = snd.to_jitter_ppq5()
        feats['Jitter:DDP']       = snd.to_jitter_ddp()
        feats['MDVP:Shimmer']     = snd.to_shimmer_local()
        feats['MDVP:Shimmer(dB)'] = snd.to_shimmer_local_dB()
        feats['Shimmer:APQ3']     = snd.to_shimmer_apq3()
        feats['Shimmer:APQ5']     = snd.to_shimmer_apq5()
        feats['MDVP:APQ']         = snd.to_shimmer_apq5()   # proxy
        feats['Shimmer:DDA']      = snd.to_shimmer_dda()
    except Exception:
        # Fill NaNs if any metric fails
        for m in [
            'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP',
            'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5',
            'MDVP:APQ','Shimmer:DDA'
        ]: feats[m] = np.nan

    # Noise ratios
    feats['NHR'] = snd.to_noise_harmonics_ratio()  if hasattr(snd, "to_noise_harmonics_ratio")  else np.nan
    feats['HNR'] = snd.to_harmonics_noise_ratio() if hasattr(snd, "to_harmonics_noise_ratio") else np.nan

    # Ensure all columns exist
    for col in FEATURE_COLS:
        feats.setdefault(col, np.nan)

    return pd.DataFrame([feats])[FEATURE_COLS]

# ── 4  ROUTES ────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return "Parkinson's Voice Classifier Backend Running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    # ── 4.1  Validate upload
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify(error="No file uploaded"), 400
    upload = request.files["file"]

    # ── 4.2  Save upload
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        upload.save(tmp_in.name)
        orig_path = tmp_in.name

    # 4.4  Feature extraction & inference
    try:
        feats = extract_features(wav_path)
        prob  = float(rf_model.predict_proba(feats)[0, 1])
        pred  = int(prob > THRESHOLD)
        result_txt = "Likely Parkinson's Disease" if pred else "Likely Healthy"
    except Exception as e:
        _cleanup([orig_path, path_wav])
        return jsonify(error=f"Feature extraction failed: {e}"), 500

    # 4.5  Clean temp files & respond
    _cleanup([orig_path, path_wav])
    return jsonify(result=result_txt,
                   probability=round(prob, 3))

# ── 6  LOCAL DEV ENTRYPOINT ─────────────────────────────────
if __name__ == "__main__":
    # For local testing: python app.py
    app.run(host="0.0.0.0", port=10000, debug=True)
