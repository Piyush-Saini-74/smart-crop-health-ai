import unittest
import torch
import os
import sys
import numpy as np
import pickle
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_extractor import CNNFeatureExtractor
from models.gnn_fusion import GNNFusion
from app.streamlit_app import process_image

class TestE2ESystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a dummy image for testing
        cls.test_img_path = "tests/dummy_leaf_e2e.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(cls.test_img_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.remove(cls.test_img_path)

    def test_full_pipeline_execution(self):
        """End-to-End Test: Load models from disk, process an image, format env, and predict."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cnn_path = os.path.join(base_dir, "models", "cnn_feature_extractor.pth")
        gnn_path = os.path.join(base_dir, "models", "hybrid_gnn_fusion.pth")
        schema_path = os.path.join(base_dir, "data", "graph_schema.pkl")
        
        # Ensure weights exist before testing full e2e (this confirms training finished)
        self.assertTrue(os.path.exists(cnn_path), "CNN Model weights not found!")
        self.assertTrue(os.path.exists(gnn_path), "GNN Model weights not found!")
        self.assertTrue(os.path.exists(schema_path), "Schema not found!")
        
        device = torch.device("cpu")
        
        # 1. Load CNN
        cnn = CNNFeatureExtractor(backbone='resnet50', pretrained=False)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn.eval().to(device)
        
        # 2. Load GNN
        gnn = GNNFusion(in_channels=2048, hidden_channels=256, out_channels=38)
        gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        gnn.eval().to(device)
        
        with open(schema_path, "rb") as f:
            schema = pickle.load(f)
            
        edge_index = torch.tensor(schema['edge_index'], dtype=torch.long).to(device)
            
        # 3. Process Image
        img_tensor, raw_img = process_image(self.test_img_path)
        with torch.no_grad():
            img_embedding = cnn(img_tensor)
            
        self.assertEqual(img_embedding.shape, (1, 2048))
        
        # 4. Construct Graph
        x = torch.zeros((8, 2048), dtype=torch.float).to(device)
        x[0, :] = img_embedding[0]
        # Simulate env
        env_norm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for j in range(7):
            x[j+1, 0] = env_norm[j]
            
        # 5. GNN Inference
        with torch.no_grad():
            disease_logits, stress_scores = gnn(x, edge_index)
            
        probs = torch.softmax(disease_logits, dim=1)
        _, pred_class_idx = torch.max(probs, dim=1)
        
        self.assertEqual(disease_logits.shape, (1, 38))
        self.assertTrue(0 <= pred_class_idx.item() < 38)
        print(f"E2E Successful! Generated pseudo-prediction class idx: {pred_class_idx.item()} with Stress: {stress_scores.item():.2f}")

if __name__ == '__main__':
    unittest.main()
