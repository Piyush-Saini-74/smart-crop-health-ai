import os
import torch
import pickle
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

class CropGraphDataset(InMemoryDataset):
    def __init__(self, root, img_embeddings_path, env_data_path, schema_path, transform=None, pre_transform=None):
        self.img_embeddings_path = img_embeddings_path
        self.env_data_path = env_data_path
        self.schema_path = schema_path
        super().__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['crop_graph_dataset.pt']

    def process(self):
        print("Loading image embeddings...")
        img_data = torch.load(self.img_embeddings_path)
        embeddings = img_data['embeddings']  # [N_samples, 2048]
        labels = img_data['labels']          # [N_samples]
        
        print("Loading environmental data...")
        env_df = pd.read_csv(self.env_data_path)
        
        print("Loading graph schema...")
        with open(self.schema_path, "rb") as f:
            schema = pickle.load(f)
            
        edge_index = torch.tensor(schema['edge_index'], dtype=torch.long)
        features_cols = schema['env_features']
        num_samples = len(embeddings)
        
        # We might not have aligned perfectly matching multimodal data (1 image = 1 env row)
        # So we simulate fusion: Assign a random valid environment row to each image.
        # In a real deployed app, the user uploads an image AND types env values.
        print("Fusing Image and Environment data into graphs...")
        data_list = []
        
        for i in range(num_samples):
            # 1. Image features
            img_feat = embeddings[i] # [2048]
            
            # 2. Pick a random env row for demonstration of training the graph
            # If disease classes match the env data label, we'd map it deterministically.
            env_row = env_df.sample(1).iloc[0]
            env_feat_vals = [env_row[col] for col in features_cols]
            
            # 3. Create node features matrix (8 nodes: 1 image + 7 env)
            # Since PyG expects nodes to have the same feature dimension, we can project 
            # them or pad them. Let's create an embedding for env nodes or pad to 2048.
            x = torch.zeros((8, 2048), dtype=torch.float)
            x[0, :] = img_feat
            
            for j in range(7):
                # Put the scalar env value into the first dim of that node's features
                # and leave the rest 0. 
                x[j+1, 0] = env_feat_vals[j]
                
            y = torch.tensor([labels[i]], dtype=torch.long)
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            
            if i % 1000 == 0 and i > 0:
                print(f"Constructed {i} / {num_samples} graphs...")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Graph Dataset processing complete!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    img_embeddings_path = os.path.join(data_dir, "image_embeddings.pt")
    env_data_path = os.path.join(data_dir, "Crop_recommendation_normalized.csv")
    schema_path = os.path.join(data_dir, "graph_schema.pkl")
    
    # Dataset will be saved to data/processed/crop_graph_dataset.pt
    dataset = CropGraphDataset(data_dir, img_embeddings_path, env_data_path, schema_path)
    print(f"Successfully loaded dataset with {len(dataset)} graphs.")
    print(f"Sample Graph: {dataset[0]}")
