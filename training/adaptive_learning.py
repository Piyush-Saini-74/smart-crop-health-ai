import os
import shutil
from datetime import datetime
import pandas as pd
import json

class AdaptiveLearningSystem:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.unverified_dir = os.path.join(self.base_dir, "data", "unverified")
        self.auto_verified_dir = os.path.join(self.base_dir, "data", "auto_verified")
        
        self.unverified_log = os.path.join(self.unverified_dir, "unverified_samples.csv")
        self.auto_verified_log = os.path.join(self.auto_verified_dir, "auto_verified.csv")
        
        for d in [self.unverified_dir, self.auto_verified_dir]:
            os.makedirs(d, exist_ok=True)
            
        self._init_csv(self.unverified_log)
        self._init_csv(self.auto_verified_log)

    def _init_csv(self, path):
        if not os.path.exists(path):
            df = pd.DataFrame(columns=["timestamp", "filename", "predicted_label", "confidence", "mc_variance", "env_data"])
            df.to_csv(path, index=False)

    def _store_sample(self, image_path, predicted_label, confidence, env_dict, mc_variance, target_dir, log_csv):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{os.path.basename(image_path)}"
        dest_path = os.path.join(target_dir, filename)
        
        try:
            shutil.copy(image_path, dest_path)
            
            # Log to CSV
            new_entry = {
                "timestamp": timestamp,
                "filename": filename,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "mc_variance": mc_variance,
                "env_data": json.dumps(env_dict)
            }
            
            df = pd.read_csv(log_csv)
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(log_csv, index=False)
            
            return True
        except Exception as e:
            print(f"Error storing sample: {e}")
            return False

    def store_unverified_sample(self, image_path, predicted_label, confidence, env_dict, mc_variance):
        """
        Stores low-confidence inputs (<60%) for potential clustering, self-supervised learning,
        or later offline analysis (fully autonomous, no manual review interface).
        """
        return self._store_sample(image_path, predicted_label, confidence, env_dict, mc_variance, 
                                  self.unverified_dir, self.unverified_log)

    def store_pseudo_label(self, image_path, predicted_label, confidence, env_dict, mc_variance):
        """
        Automatically establishes a pseudo-label for high-confidence predictions (>95%).
        These samples are trusted and will be included in the next periodic autonomous retraining cycle.
        """
        return self._store_sample(image_path, predicted_label, confidence, env_dict, mc_variance, 
                                  self.auto_verified_dir, self.auto_verified_log)

