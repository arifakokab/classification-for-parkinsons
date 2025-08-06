# CarePath AI Foundation – Parkinson's Disease Voice Classifier

## Overview

This project was created by **Arifa Kokab** as the **capstone project for the M.Sc.Eng in Applied Artificial Intelligence** at the **University of San Diego, Class of 2025**.

It is a free, public, AI-powered screening tool for early detection of Parkinson’s Disease using a short voice sample (“aaah”).  
Developed under the CarePath AI Foundation, this application is a research and education prototype—not a diagnostic device.

www.parkinsonsaiscreening-carepathai-foundation.care
---

## Setup & Deployment

### **Backend (Flask API)**

#### **Local Setup**
1. Clone this repo and navigate to `/backend`
2. Create a virtualenv and activate it:
    ```
    python3.12 -m venv venv
    source venv/bin/activate
    ```
3. Install requirements:
    ```
    pip install -r requirements.txt
    ```
4. Ensure `rf_model.pkl` is present.
5. Run:
    ```
    python app.py
    ```
6. API will be at `http://localhost:10000/predict`

#### **Cloud Deployment**
- Uses [Render.com](https://render.com/)
- **.python-version** pins Python to 3.12 to support `pydub`
- Build command:  
  (Render auto-detects Python and runs `pip install -r requirements.txt`)
- **Gunicorn** runs as the production WSGI server.

---

### **Frontend (React/Vite)**

#### **Local Development**
1. Navigate to `/frontend`
2. Install dependencies:
    ```
    npm install
    ```
3. Start dev server:
    ```
    npm run dev
    ```
4. Open [http://localhost:5173](http://localhost:5173)

#### **Production Deployment**
- Deploy as a static site (e.g., Render, Vercel, Netlify)
- **Build command:**
    ```
    npm run build
    ```
- **Output:** Serve `/dist` as static

#### **API Endpoint**
- Update `API_URL` in `App.jsx` to point to your backend:
    ```js
    const API_URL = "https://your-backend-url/predict";
    ```

---

## Usage

1. Visit the web app.
2. Click **Start Recording**, say “aaah” for at least 5 seconds, then click **Stop**.
3. Click **Run Analysis**.
4. View your probability and result.

> For best results, record in a quiet room, and speak clearly.

---

## Model Deployment Notes

- **Audio** is always converted to mono 16kHz WAV on the backend.
- **Features** are extracted using Parselmouth (Praat); missing values are handled as zeros.
- **Random Forest** classifier predicts “Likely Parkinson’s Disease” or “Likely Healthy” with a fixed threshold (0.63).
- **Limitations:**  
  - Feature extraction may not work well with silence or low-quality browser audio.
  - If most features are NaN, result may not be valid—encourage users to record again.

---

## Privacy & Disclaimer

- No voice data or personal information is stored or shared.
- This tool is **not a medical device** and is for education/screening only.
- Consult a medical professional for clinical concerns.

---

## Citation / Attribution

- Developed by [Arifa Kokab] for the CarePath AI Foundation (2025).

---

## License

This project is released under the [MIT License](LICENSE).

---

**For questions, contributions, or collaborations, open an Issue or contact Arifa Kokab (akokab@sandiego.edu)**
