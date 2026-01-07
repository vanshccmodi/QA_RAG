import json
import numpy as np

from db import cursor
from embeddings import embedding_model, cosine_similarity


def retrieve_top_chunks(question, k=3):
    q_emb = embedding_model.encode(question)

    cursor.execute(
        "SELECT document, chunk_id, text, embedding FROM chunks"
    )
    rows = cursor.fetchall()

    scored = []
    for doc, cid, text, emb_json in rows:
        emb = np.array(json.loads(emb_json))
        score = cosine_similarity(q_emb, emb)
        scored.append((score, doc, cid, text))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]
