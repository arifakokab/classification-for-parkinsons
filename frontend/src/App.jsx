import React, { useRef, useState } from "react";
import "./index.css";

const API_URL =
  "https://parkinsonsaiscreening-carepathai-foundation.care/predict"; // Render backend

export default function App() {
  // ── State ────────────────────────────────────────────────
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState("");
  const [audioBlob, setAudioBlob] = useState(null);
  const [result, setResult] = useState("");
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  // ── Recorder helpers ─────────────────────────────────────
  const startRecording = async () => {
    setResult("");
    setAudioURL("");
    setAudioBlob(null);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorderRef.current = new MediaRecorder(stream, { mimeType: "audio/webm" });
    chunksRef.current = [];

    recorderRef.current.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
    recorderRef.current.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setAudioBlob(blob);
      setAudioURL(URL.createObjectURL(blob));
    };

    recorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    setRecording(false);
  };

  // ── Send to backend ──────────────────────────────────────
  const runAnalysis = async () => {
    if (!audioBlob) return setResult("⚠️  Please record first.");

    setResult("⏳ Analyzing…");
    const fd = new FormData();
    fd.append("file", audioBlob, "recording.webm");

    try {
      const r = await fetch(API_URL, { method: "POST", body: fd });
      const j = await r.json();
      if (j.error) setResult("❌ " + j.error);
      else
        setResult(
          `${j.result}\nProbability: ${j.probability}  (Threshold 0.63)`
        );
    } catch (err) {
      setResult("❌ Network error: " + err.message);
    }
  };

  // ── UI ───────────────────────────────────────────────────
  return (
    <div className="wrapper">
      <header>
        <h1>🎤 CarePath&nbsp;AI&nbsp;Foundation</h1>
        <h2>Parkinson&rsquo;s&nbsp;Voice&nbsp;Screening&nbsp;Tool</h2>
        <p className="tagline">
          Free, non-invasive, early detection powered by&nbsp;AI.<br />
          Simply record&nbsp;“aaah” for&nbsp;5-8&nbsp;seconds.
        </p>
      </header>

      <section className="recorder">
        <button onClick={startRecording} disabled={recording}>
          ▶️ Start&nbsp;Recording
        </button>
        <button onClick={stopRecording} disabled={!recording}>
          ⏹ Stop
        </button>

        {audioURL && (
          <>
            <audio controls src={audioURL} className="player" />
            <button onClick={runAnalysis} className="analyze-btn">
              🔍 Run&nbsp;Analysis
            </button>
          </>
        )}
      </section>

      {result && (
        <section className="result">
          <pre>{result}</pre>
        </section>
      )}

      <footer>
        <p>
          <b>Disclaimer:</b> This prototype is for educational screening only
          and is <u>not</u> a medical diagnosis. Consult a healthcare
          professional for concerns about Parkinson&rsquo;s&nbsp;Disease.
        </p>
        <p>© 2025 CarePath AI Foundation</p>
      </footer>
    </div>
  );
}
