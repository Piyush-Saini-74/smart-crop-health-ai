import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

def process_environmental_data(csv_path="Crop_recommendation.csv", out_dir="data"):
    print("Loading Crop Recommendation dataset...")
    df = pd.read_csv(csv_path)
    
    # Features: N, P, K, temperature, humidity, ph, rainfall
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    
    print("Normalizing environmental features...")
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save the scaler for inference time
    os.makedirs(out_dir, exist_ok=True)
    scaler_path = os.path.join(out_dir, "env_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save normalized data (optional, useful for debugging)
    norm_csv_path = os.path.join(out_dir, "Crop_recommendation_normalized.csv")
    df.to_csv(norm_csv_path, index=False)
    print(f"Normalized data saved to {norm_csv_path}")
    
    return df, features, target

def build_graph_schema(df, features, target, out_dir="data"):
    """
    Builds an initial adjacency matrix and node definitions.
    For the Hybrid model:
    We'll have 1 Image Node + 7 Environment Nodes.
    In a real PyG application, edge_index represents connections.
    Let's fully connect the Image Node (idx 0) to all Environment Nodes (idx 1-7).
    """
    print("Designing Graph Schema...")
    
    # Node Mapping
    node_mapping = {0: "Image_CNN_Embedding"}
    for idx, feat in enumerate(features, start=1):
        node_mapping[idx] = feat
        
    # Fully connect Node 0 to Nodes 1..7 (bidirectional)
    source_nodes = []
    target_nodes = []
    for i in range(1, 8):
        # Image -> Env Factor
        source_nodes.append(0)
        target_nodes.append(i)
        # Env Factor -> Image
        source_nodes.append(i)
        target_nodes.append(0)
    
    # Convert to standard PyG edge_index format (2, num_edges)
    edge_index = np.array([source_nodes, target_nodes])
    
    graph_schema = {
        "node_mapping": node_mapping,
        "edge_index": edge_index,
        "num_nodes": 8,
        "env_features": features
    }
    
    schema_path = os.path.join(out_dir, "graph_schema.pkl")
    with open(schema_path, "wb") as f:
        pickle.dump(graph_schema, f)
        
    print(f"Graph schema saved to {schema_path}")
    print(f"Nodes: {node_mapping}")
    
if __name__ == "__main__":
    base_dir = "c:/Users/SUNNY/Desktop/Major Project/Adaptive Explainable AI System for Smart Crop Health Monitoring"
    csv_file = os.path.join(base_dir, "Crop_recommendation.csv")
    data_dir = os.path.join(base_dir, "data")
    
    if os.path.exists(csv_file):
        df, features, target = process_environmental_data(csv_file, data_dir)
        build_graph_schema(df, features, target, data_dir)
    else:
        print(f"Error: Could not find {csv_file}. Please ensure the dataset is present.")
