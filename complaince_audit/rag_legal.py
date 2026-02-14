import ollama
from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    id: str
    text: str
    source: str = "Service_Agreement.txt"

def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 20]
    
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(id=f"clause_{i}", text=chunk))
    return docs

class SimpleTfidfRAG:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def retrieve(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self.docs[i], float(sims[i])) for i in idx if sims[i] > 0]

def ask_phi3(context, question):
    prompt = f"""
You are a Legal Compliance Assistant.
Extract the exact clause relevant to the user's question from the provided contract text.
Do not summarize; quote the relevant text where possible or explain the legal implication based ONLY on the text.

Context:
{context}

Question:
{question}
"""
    response = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def main():
    print("=== Legal and Compliance Document Audit ===")
    try:
        docs = load_text_file("Service_Agreement.txt")
    except FileNotFoundError:
        print("Error: 'Service_Agreement.txt' not found.")
        return

    rag = SimpleTfidfRAG(docs)

    while True:
        query = input("\nAudit Question (type exit to quit): ")
        if query.lower() == "exit":
            break

        results = rag.retrieve(query)
        if not results:
            print("No relevant clauses found in the document.")
            continue

        context = "\n\n".join([doc.text for doc, _ in results])
        print(f"\n--- Retrieved {len(results)} Clauses ---")
        print(context)
        
        answer = ask_phi3(context, query)
        print("\n--- Compliance Result ---")
        print(answer)

if __name__ == "__main__":
    main()
