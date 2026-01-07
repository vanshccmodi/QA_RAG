import re
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_chunking(text, similarity_threshold=0.75):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= 1:
        return sentences

    embeddings = embedding_model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])

        if sim >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    chunks.append(" ".join(current_chunk))
    return chunks
