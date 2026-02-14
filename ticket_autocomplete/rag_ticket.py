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
    source: str = "Support_Tickets.txt"

def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 20]
    
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(id=f"ticket_{i}", text=chunk))
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
You are a Customer Support Agent Helper.
Your goal is to draft a polite response to a new customer issue based on how similar past tickets were resolved.

Past Ticket Context:
{context}

New Customer Issue:
{question}

Draft Response:
"""
    response = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def main():
    print("=== Customer Support Ticket Autocomplete ===")
    try:
        docs = load_text_file("Support_Tickets.txt")
    except FileNotFoundError:
        print("Error: 'Support_Tickets.txt' not found.")
        return

    rag = SimpleTfidfRAG(docs)

    while True:
        query = input("\nEnter New Ticket Issue (type exit to quit): ")
        if query.lower() == "exit":
            break

        results = rag.retrieve(query)
        if not results:
            print("No similar past tickets found. Suggest manual investigation.")
            continue

        context = "\n\n".join([doc.text for doc, _ in results])
        print(f"\n--- Found {len(results)} Similar Past Tickets ---")
        print(context)
        
        answer = ask_phi3(context, query)
        print("\n--- Suggested Response ---")
        print(answer)

if __name__ == "__main__":
    main()
