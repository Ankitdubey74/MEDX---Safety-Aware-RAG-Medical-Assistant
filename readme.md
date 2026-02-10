MEDX – Safety-Aware RAG Medical Assistant

MEDX is a safety-first medical question answering system built using a fine-tuned PHI-2 language model and a hybrid Retrieval-Augmented Generation (RAG) pipeline.
The system is designed to provide context-aware, reliable, and explainable medical responses, while actively detecting high-risk scenarios and minimizing unsafe outputs.




🚀 Key Highlights
Fine-tuned PHI-2 LLM specifically for medical question answering
Hybrid retrieval system (Dense Embeddings + BM25) for high recall
Tool-calling safety layer for emergency detection and medical risk analysis
RAGAS-based evaluation & monitoring with failure logging
CPU-friendly inference pipeline (no GPU dependency)



🧠 System Architecture 
User Query
Weak Query Rewriting (clarifies vague medical questions)
Hybrid Retrieval
Dense semantic search (embeddings)
Sparse keyword search (BM25)
Context Injection
Fine-Tuned PHI-2 Inference
Safety Tool Layer
Emergency detection
Risk scoring
BMI computation
Final Response + Confidence Score
Evaluation & Failure Logging (RAGAS)


🏥 Medical Safety Design
MEDX is not a generic chatbot. It includes a dedicated safety-aware reasoning layer:



🚨 Emergency Detection
Identifies high-risk symptoms (e.g., chest pain, breathing difficulty)
Triggers medical disclaimers and urgent guidance
📊 Symptom Risk Scoring
Assigns severity scores based on symptoms and context
⚖️ BMI & Health Metrics
Computes BMI when height/weight data is provided
✏️ Weak Query Rewriting
Improves incomplete or poorly phrased medical queries before retrieval



🔍 Retrieval Strategy
To maximize recall and reduce hallucinations:
Dense Retrieval
Captures semantic meaning of medical queries
BM25 (Sparse Retrieval)
Handles disease names, symptoms, and medical terms effectively
Hybrid Fusion
Combines both results for robust medical context grounding




🤖 Model Fine-Tuning Details
Base Model: microsoft/phi-2
Fine-Tuning Objective: Medical QA & symptom-based reasoning
Why Fine-Tuning?
Base LLMs lack medical grounding and safety sensitivity
Fine-tuning improves domain accuracy and reduces hallucinations
⚠️ Model weights are not included in this repository due to size constraints.
This repo contains configuration, training metadata, and integration logic.



📈 Evaluation & Monitoring
MEDX uses RAGAS for systematic evaluation:
Answer relevance
Context precision & recall
Faithfulness to retrieved documents
Confidence scoring for responses



Additionally:
❌ Failure cases are logged
📊 Enables continuous improvement & explainability




🧪 Project Structure
MEDX/
│
├── main.py                # RAG pipeline orchestration
├── tools.py               # Safety tools (risk, BMI, emergency detection)
├── query_rewriter.py      # Weak query rewriting logic
├── evaluation.py          # RAGAS evaluation & scoring
├── failure_logger.py      # Failure logging & monitoring
│
├── phi2-medx-merged/      # Fine-tuning configs (no heavy weights)
├── vectorstore/           # Embedding index (optional)
│
├── requirements.txt
├── README.md
└── .gitignore



🖥️ How to Run (CPU Friendly)
pip install -r requirements.txt
python main.py


Designed to run efficiently on CPU-only systems



⚠️ Disclaimer
MEDX is an AI research project and not a replacement for professional medical advice.
It is intended for educational and experimental purposes only.



🔮 Future Improvements
Multilingual medical support
Advanced medical knowledge graph integration
Clinical-grade evaluation datasets
Real-time monitoring dashboard



👤 Author

Ankit Dubey
Data Science / AI Engineer
Focused on LLMs, RAG systems, medical AI, and safety-aware ML systems



