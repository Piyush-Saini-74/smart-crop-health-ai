try:
    from torch_geometric.explain import Explainer, GNNExplainer
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx
except ImportError:
    print("Warning: torch_geometric is not installed. GNNExplainer will not work.")

def explain_graph_prediction(model, x, edge_index, target_node_idx, target_class):
    """
    Uses GNNExplainer to find the most important nodes and edges for the prediction.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    
    explanation = explainer(x, edge_index, index=target_node_idx)
    
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask
    
    return node_mask, edge_mask

def visualize_explanation(explanation, feature_names=None):
    """
    Visualizes the graph explanation.
    To be implemented when specific graph topology is finalized.
    """
    # GNNExplainer can visualize feature importance natively via PyG
    pass
