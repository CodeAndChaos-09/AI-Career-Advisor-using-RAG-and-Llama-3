from sentence_transformers import CrossEncoder

# Load reranker model once
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_nodes(query, nodes, top_k=3):
    """
    Rerank retrieved nodes using cross-encoder model
    """

    if not nodes:
        return []

    # create query-document pairs
    pairs = [(query, node.text) for node in nodes]

    scores = reranker_model.predict(pairs)

    # combine nodes with scores
    ranked_nodes = sorted(
        zip(nodes, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # return top nodes
    reranked_nodes = [node for node, _ in ranked_nodes[:top_k]]

    return reranked_nodes