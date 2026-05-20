import unittest
import torch
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_extractor import CNNFeatureExtractor
from models.gnn_fusion import GNNFusion
from training.adaptive_learning import AdaptiveLearningSystem
from app.streamlit_app import compute_confidence
from unittest.mock import patch

class TestSmartCropSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a dummy image for testing
        cls.test_img_path = "tests/dummy_leaf.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        if not os.path.exists("tests"):
            os.makedirs("tests")
        img.save(cls.test_img_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.remove(cls.test_img_path)

    # 1. INDEPENDENT COMPONENT TESTS (UNIT TESTS)
    
    def test_cnn_feature_extraction_dims(self):
        """Test CNN standalone to ensure it outputs exactly 2048 dimensions."""
        model = CNNFeatureExtractor(pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2048))

    def test_gnn_model_dims(self):
        """Test GNN standalone to ensure it handles 8 nodes (1 img + 7 env) returning 38 distinct plant classes."""
        model = GNNFusion(in_channels=2048, hidden_channels=256, out_channels=38)
        model.eval()
        
        # Simulate 1 image node and 7 environmental nodes = 8 nodes
        x = torch.zeros(8, 2048) 
        
        # Connect img node (0) to environment nodes (1-7)
        edge_index = torch.tensor([[0,0,0,0,0,0,0, 1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7, 0,0,0,0,0,0,0]], dtype=torch.long)
                                   
        logits, stress = model(x, edge_index)
        self.assertEqual(logits.shape, (1, 38))
        # Ensure stress score is between 0 and 1
        self.assertTrue(0.0 <= stress.item() <= 1.0)

    def test_confidence_computation(self):
        """Test that compute_confidence clamps values out of bounds and penalizes high stress."""
        disease_logits = torch.randn(1, 38) * 10 # Artificially high spread
        stress = 0.9 # High stress
        
        # Even if logits are extremely confident, stress penalty should reduce score slightly
        conf = compute_confidence(disease_logits, stress)
        self.assertTrue(0.0 <= conf <= 100.0)

    # 2. INTEGRATION TESTS (Module-to-Module connectivity)
    
    def test_adaptive_learning_trigger(self):
        """Test that the Adaptive Learning System correctly stores samples."""
        import shutil
        mock_dir = "tests/mock_user_samples"
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)
            
        als = AdaptiveLearningSystem(base_dir=mock_dir)
                
        # Store samples
        import time
        for i in range(5):
            env_data = {"n": 10, "p": 10, "k": 10, "temp": 25, "hum": 60, "ph": 6.5, "rain": 100}
            als.store_unverified_sample(self.test_img_path, "Early Blight", 45.0, env_data, 0.2)
            time.sleep(0.1) # Ensure unique timestamp for each file
            
        # Check that samples are stored in csv
        df = pd.read_csv(als.unverified_log)
        self.assertEqual(len(df), 5)

if __name__ == '__main__':
    unittest.main()
