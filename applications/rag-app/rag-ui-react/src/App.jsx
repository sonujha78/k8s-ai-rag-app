import { useState } from "react";
import axios from "axios";
import "./App.css";

const INGESTOR_URL = import.meta.env.VITE_INGESTOR_URL || "/api/ingestor";
const QUERY_URL = import.meta.env.VITE_QUERY_URL || "/api/query";

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus("Uploading and indexing...");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post(`${INGESTOR_URL}/ingest`, formData);
      setUploadStatus(`Indexed successfully: ${res.data.chunks} chunks created`);
    } catch (err) {
      setUploadStatus("Upload failed: " + err.message);
    }
  };

  const handleAsk = async () => {
    if (!question) return;
    setLoading(true);
    setAnswer("");
    try {
      const res = await axios.post(`${QUERY_URL}/query`, { question });
      setAnswer(res.data.answer);
    } catch (err) {
      setAnswer("Error: " + err.message);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>RAG App — Ask Your PDF</h1>

      <section className="card">
        <h2>1. Upload PDF</h2>
        <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload}>Upload PDF</button>
        <p>{uploadStatus}</p>
      </section>

      <section className="card">
        <h2>2. Ask a Question</h2>
        <input
          type="text"
          placeholder="Ask something about your document..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Ask Question"}
        </button>
        {answer && (
          <div className="answer">
            <strong>Answer:</strong>
            <p>{answer}</p>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
