from fastapi import FastAPI, UploadFile, File
import json

from db import conn, cursor
from embeddings import semantic_chunking, embedding_model
from retrieval import retrieve_top_chunks
from llm import build_prompt, call_llm

app = FastAPI(
    title="RAG QA System",
    version="1.0"
)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")

    chunks = semantic_chunking(text)
    embeddings = embedding_model.encode(chunks)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        cursor.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            (file.filename, i, chunk, json.dumps(emb.tolist()))
        )

    conn.commit()
    return {
        "message": "Document ingested",
        "chunks": len(chunks)
    }


@app.post("/ask")
def ask(question: str):
    top_chunks = retrieve_top_chunks(question)

    if not top_chunks:
        return {
            "question": question,
            "answer": "I don't know based on the provided context",
            "confidence": 0.0,
            "evidence": []
        }

    best_score = max(c[0] for c in top_chunks)
    CONFIDENCE_THRESHOLD = 0.6

    if best_score < CONFIDENCE_THRESHOLD:
        return {
            "question": question,
            "answer": "I don't know based on the provided context",
            "confidence": round(float(best_score), 2),
            "evidence": []
        }

    prompt = build_prompt(question, top_chunks)
    answer = call_llm(prompt)

    # deterministic fallback
    if "I don't know based on the provided context" in answer:
        answer = top_chunks[0][3]

    evidence = [
        {
            "document": doc,
            "chunk_id": cid,
            "text": text
        }
        for _, doc, cid, text in top_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "confidence": round(float(best_score), 2),
        "evidence": evidence
    }
