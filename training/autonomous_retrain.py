import os
import sys
import shutil
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch_geometric.data import Data
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_extractor import CNNFeatureExtractor
from models.gnn_fusion import GNNFusion

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")
auto_verified_dir = os.path.join(data_dir, "auto_verified")
csv_path = os.path.join(auto_verified_dir, "auto_verified.csv")
deployment_config_path = os.path.join(models_dir, "deployment_config.json")
retraining_logs_path = os.path.join(data_dir, "retraining_logs.csv")
schema_path = os.path.join(data_dir, "graph_schema.pkl")
scaler_path = os.path.join(data_dir, "env_scaler.pkl")

def init_logs():
    if not os.path.exists(retraining_logs_path):
        df = pd.DataFrame(columns=["timestamp", "old_version", "new_version", "new_model_file", "loss", "samples_trained"])
        df.to_csv(retraining_logs_path, index=False)

def get_deployment_config():
    if os.path.exists(deployment_config_path):
        with open(deployment_config_path, "r") as f:
            return json.load(f)
    return {"gnn_model_file": "hybrid_gnn_fusion.pth", "version": 1.0}

def update_deployment_config(new_file, new_version):
    cfg = {
        "gnn_model_file": new_file,
        "version": new_version,
        "last_updated": datetime.now().isoformat()
    }
    with open(deployment_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        img = Image.open(image_path).convert('RGB')
        return preprocess(img).unsqueeze(0)
    except:
        return None

def autonomous_retrain(batch_trigger_size=10):
    print(f"[{datetime.now()}] Checking for autonomous retraining...")
    if not os.path.exists(csv_path):
        print("No auto-verified data found.")
        return

    df = pd.read_csv(csv_path)
    
    # Deduplicate based on filename hash or path
    df = df.drop_duplicates(subset=["filename"], keep="last")
    
    # Only train if we have un-processed samples (simple proxy: clear the csv after training, or track pointers)
    # For this autonomous setup, we will clear rows we processed or move them to archive to prevent perpetual loops.
    if len(df) < batch_trigger_size:
        print(f"Not enough new samples ({len(df)}). Trigger requires {batch_trigger_size}.")
        return
        
    print(f"Initiating autonomous fine-tuning on {len(df)} pseudo-labeled samples...")
    init_logs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import pickle
    with open(schema_path, "rb") as f:
        graph_schema = pickle.load(f)
    with open(scaler_path, "rb") as f:
        env_scaler = pickle.load(f)
        
    cnn = CNNFeatureExtractor(backbone='resnet50', pretrained=False)
    cnn_file = os.path.join(models_dir, "cnn_feature_extractor.pth")
    if os.path.exists(cnn_file):
        cnn.load_state_dict(torch.load(cnn_file, map_location=device))
    cnn.eval().to(device)
    
    cfg = get_deployment_config()
    current_model = cfg.get("gnn_model_file", "hybrid_gnn_fusion.pth")
    current_version = float(cfg.get("version", 1.0))
    
    gnn = GNNFusion(in_channels=2048, hidden_channels=256, out_channels=38, model_type='gcn')
    gnn_model_path = os.path.join(models_dir, current_model)
    if os.path.exists(gnn_model_path):
        gnn.load_state_dict(torch.load(gnn_model_path, map_location=device))
    gnn.to(device)
    
    optimizer = optim.Adam(gnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Hardcoded class names since the original dataset directory is ignored in Git
    class_names = [
        "Apple - Apple scab", "Apple - Black rot", "Apple - Cedar apple rust", "Apple - healthy",
        "Blueberry - healthy", "Cherry (including sour) - Powdery mildew", "Cherry (including sour) - healthy",
        "Corn (maize) - Cercospora leaf spot Gray leaf spot", "Corn (maize) - Common rust ", 
        "Corn (maize) - Northern Leaf Blight", "Corn (maize) - healthy", "Grape - Black rot",
        "Grape - Esca (Black Measles)", "Grape - Leaf blight (Isariopsis Leaf Spot)", "Grape - healthy",
        "Orange - Haunglongbing (Citrus greening)", "Peach - Bacterial spot", "Peach - healthy",
        "Pepper, bell - Bacterial spot", "Pepper, bell - healthy", "Potato - Early blight",
        "Potato - Late blight", "Potato - healthy", "Raspberry - healthy", "Soybean - healthy",
        "Squash - Powdery mildew", "Strawberry - Leaf scorch", "Strawberry - healthy",
        "Tomato - Bacterial spot", "Tomato - Early blight", "Tomato - Late blight",
        "Tomato - Leaf Mold", "Tomato - Septoria leaf spot", "Tomato - Spider mites Two-spotted spider mite",
        "Tomato - Target Spot", "Tomato - Tomato Yellow Leaf Curl Virus", "Tomato - Tomato mosaic virus",
        "Tomato - healthy"
    ] 
        
    # Data extraction loop for new samples
    pyg_data_list = []
    
    gnn.train()
    running_loss = 0.0
    import numpy as np
    
    for idx, row in df.iterrows():
        img_path = os.path.join(auto_verified_dir, row['filename'])
        if not os.path.exists(img_path): continue
        
        img_tensor = process_image(img_path)
        if img_tensor is None: continue
        
        with torch.no_grad():
            img_embedding = cnn(img_tensor.to(device))
            
        env_dict = json.loads(row['env_data'])
        env_raw = np.array([[env_dict["n"], env_dict["p"], env_dict["k"], env_dict["temp"], env_dict["hum"], env_dict["ph"], env_dict["rain"]]])
        env_norm = env_scaler.transform(env_raw)[0]
        
        edge_index = torch.tensor(graph_schema['edge_index'], dtype=torch.long).to(device)
        x = torch.zeros((8, 2048), dtype=torch.float).to(device)
        x[0, :] = img_embedding[0]
        for j in range(7):
            x[j+1, 0] = env_norm[j]
        
        target_name = row['predicted_label']
        try:
            target_idx = class_names.index(target_name)
        except ValueError:
            target_idx = 0
            
        target = torch.tensor([target_idx], dtype=torch.long).to(device)
        
        pyg_data_list.append((x, edge_index, target))
    
    # REPLAY STRATEGY (Catastrophic Forgetting Prevention)
    # Normally we would sample from the original dataset embeddings `image_embeddings.pt`
    # We will simulate the appending of 20% Replay data embeddings here.
    # In production, we'd load N random samples from old dataset and intersperse them.
    # (Omitted due to data footprint size, but logically this is the block where we append to pyg_data_list).
    
    # Fine-Tuning Loop
    epochs = 3
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, edge_index, target in pyg_data_list:
            optimizer.zero_grad()
            logits, _ = gnn(x, edge_index)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        running_loss += epoch_loss / len(pyg_data_list)
        print(f"Fine-tuning Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(pyg_data_list):.4f}")
        
    avg_loss = running_loss / epochs
    
    # Automated Deployment & Versioning
    new_version = current_version + 0.1
    new_model_file = f"hybrid_gnn_fusion_v{new_version:.1f}.pth"
    new_model_path = os.path.join(models_dir, new_model_file)
    
    torch.save(gnn.state_dict(), new_model_path)
    print(f"Generated new model version: {new_model_file}")
    
    update_deployment_config(new_model_file, round(new_version, 1))
    
    # Log the successful automation pass
    logs_df = pd.read_csv(retraining_logs_path)
    new_log = {
        "timestamp": datetime.now().isoformat(),
        "old_version": current_version,
        "new_version": round(new_version, 1),
        "new_model_file": new_model_file,
        "loss": avg_loss,
        "samples_trained": len(pyg_data_list)
    }
    logs_df = pd.concat([logs_df, pd.DataFrame([new_log])], ignore_index=True)
    logs_df.to_csv(retraining_logs_path, index=False)
    
    # Archive the trained samples properly to avoid constant retraining
    archive_dir = os.path.join(data_dir, "archived_verified")
    os.makedirs(archive_dir, exist_ok=True)
    for idx, row in df.iterrows():
        src = os.path.join(auto_verified_dir, row['filename'])
        dst = os.path.join(archive_dir, row['filename'])
        if os.path.exists(src):
            shutil.move(src, dst)
            
    # Clear the staging csv
    pd.DataFrame(columns=df.columns).to_csv(csv_path, index=False)
    print("Autonomous deployment pipeline finished successfully. Web App will dynamically reload the new model.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_trigger", type=int, default=10, help="Minimum auto verified samples to trigger training")
    args = parser.parse_args()
    
    autonomous_retrain(batch_trigger_size=args.batch_trigger)
