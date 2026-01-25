from main import vectorstore

# Domain truth for retrieval evaluation

domain_truth = {
    "fatigue": [
        "symptom: fatigue"
    ],
    "headache": [
        "symptom: headache"
    ]
}

# Create retrieval function
def retrieve_documents(query: str, k: int = 5):
    return vectorstore.similarity_search(query, k=k)

# Implement Recall
def recall_at_k(retrieved_docs, relevant_texts, k):
    """
    retrieved_docs: list[Document]
    relevant_texts: list[str]
    """
    retrieved_k = retrieved_docs[:k]

    hits = 0
    for doc in retrieved_k:
        for rel in relevant_texts:
            if rel in doc.page_content.lower():
                hits += 1
                break

    return hits / len(relevant_texts) if relevant_texts else 0

# Implement Mean Reciprocal Rank
def reciprocal_rank(retrieved_docs, relevant_texts):
    for rank, doc in enumerate(retrieved_docs, start=1):
        for rel in relevant_texts:
            if rel in doc.page_content.lower():
                return 1 / rank
    return 0

# Perform evaluation
def evaluate_retrieval(domain_truth: dict, k: int = 5):
    recalls = []
    rrs = []

    for query, relevant_texts in domain_truth.items():
        retrieved_docs = retrieve_documents(query, k)

        recalls.append(
            recall_at_k(retrieved_docs, relevant_texts, k)
        )
        rrs.append(
            reciprocal_rank(retrieved_docs, relevant_texts)
        )

    return {
        f"Recall@{k}": sum(recalls) / len(recalls),
        "MRR": sum(rrs) / len(rrs)
    }
# Run evaluation
metrics = evaluate_retrieval(domain_truth, k=3)
pprint(metrics)
