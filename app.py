# ─────────────────────────────────────────────────────────────
#  CarePath AI Foundation – Parkinson's Voice Classifier API
#  Supports: WebM/Opus or WAV uploads →
#            extracts 16 voice features → Random-Forest at
#            threshold 0.63 → JSON result
# ─────────────────────────────────────────────────────────────

import os, uuid, tempfile, warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import joblib
import numpy as np
import pandas as pd
import parselmouth

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_PATH  = "rf_model.pkl"
THRESHOLD   = 0.63

FEATURE_COLS = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ',
    'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
    'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR'
]

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
rf_model = joblib.load(MODEL_PATH)

def _cleanup(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def extract_features(path_wav: str) -> pd.DataFrame:
    snd = parselmouth.Sound(path_wav)
    feats = {}

    pitch = snd.to_pitch()
    feats['MDVP:Fo(Hz)']  = np.nanmean(pitch.selected_array['frequency'])
    feats['MDVP:Fhi(Hz)'] = np.nanmax(pitch.selected_array['frequency'])
    feats['MDVP:Flo(Hz)'] = np.nanmin(pitch.selected_array['frequency'])

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
        feats['MDVP:APQ']         = snd.to_shimmer_apq5()
        feats['Shimmer:DDA']      = snd.to_shimmer_dda()
    except Exception:
        for m in [
            'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP',
            'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5',
            'MDVP:APQ','Shimmer:DDA'
        ]: feats[m] = np.nan

    try:    feats['NHR'] = snd.to_noise_harmonics_ratio()
    except: feats['NHR'] = np.nan
    try:    feats['HNR'] = snd.to_harmonics_noise_ratio()
    except: feats['HNR'] = np.nan

    for col in FEATURE_COLS:
        feats.setdefault(col, np.nan)

    return pd.DataFrame([feats])[FEATURE_COLS]

@app.route("/", methods=["GET"])
def health():
    return "Parkinson's Voice Classifier Backend Running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    orig_path = None
    wav_path = None
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify(error="No file uploaded"), 400
        upload = request.files["file"]

        # Save upload as temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            upload.save(tmp_in.name)
            orig_path = tmp_in.name

        # Always convert to PCM WAV for Parselmouth
        wav_path = f"/tmp/{uuid.uuid4()}.wav"
        AudioSegment.from_file(orig_path).set_frame_rate(16000).set_channels(1).export(
            wav_path, format="wav"
        )

    try:
        feats = extract_features(wav_path)
        print("Extracted features:", feats)
        prob  = float(rf_model.predict_proba(feats)[0, 1])
        print("Probability:", prob)
        pred  = int(prob > THRESHOLD)
        result_txt = "Likely Parkinson's Disease" if pred else "Likely Healthy"
        print("Result:", result_txt)
        return jsonify(result=result_txt,
                       probability=round(prob, 3),
                       threshold=THRESHOLD)
    except Exception as e:
        print("ERROR in prediction:", e)
        return jsonify(error=f"Prediction failed: {e}"), 500
    finally:
        _cleanup([orig_path, wav_path])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
