from sentence_transformers import CrossEncoder

# Load reranker model
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_nodes(query, nodes, top_k=3):
    """
    Rerank retrieved nodes using a cross-encoder model
    """

    pairs = [(query, node.text) for node in nodes]

    scores = reranker_model.predict(pairs)

    # Combine nodes with scores
    ranked = sorted(
        zip(nodes, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top nodes
    reranked_nodes = [node for node, _ in ranked[:top_k]]

    return reranked_nodes