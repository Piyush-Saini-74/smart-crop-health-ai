import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

# Ensure models can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_extractor import CNNFeatureExtractor

def compute_confidence(probs, graph_uncertainty=0.0):
    """
    Computes a combined confidence score based on Softmax probability, Entropy, and Graph Uncertainty.
    """
    max_prob = torch.max(probs, dim=-1)[0].item()
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()
    confidence = (max_prob * 0.7) + ((1.0 - min(entropy, 1.0)) * 0.2) + ((1.0 - graph_uncertainty) * 0.1)
    return confidence * 100.0

def train_cnn(data_dir, batch_size=32, epochs=5, lr=0.001, save_path="models/cnn_feature_extractor.pth"):
    print(f"--- Phase 2: Training/Fine-tuning CNN ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard CNN Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(data_dir, "train")
    if not os.path.exists(train_path):
        print(f"Error: Training directory {train_path} not found.")
        print("Note: If the dataset is structured differently (e.g., 'New Plant Diseases Dataset(Augmented)/train'), please update the data_dir.")
        return None

    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Found {len(dataset)} images classified in {num_classes} classes.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model (we append a temporary classification head to fine-tune the extractor)
    model = CNNFeatureExtractor(backbone='resnet50', pretrained=True)
    
    # Freeze lower layers for faster training (optional, focusing on top layers)
    for param in list(model.extractor.children())[:-2]:
        for p in param.parameters():
            p.requires_grad = False

    class TempClassifier(nn.Module):
        def __init__(self, extractor, num_classes):
            super().__init__()
            self.extractor = extractor
            self.fc = nn.Linear(extractor.output_dim, num_classes)
            
        def forward(self, x):
            features = self.extractor(x)
            return self.fc(features)

    full_model = TempClassifier(model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, full_model.parameters()), lr=lr)

    # Simplified training loop
    for epoch in range(epochs):
        full_model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = full_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0
                
            # Cap at 50 batches per epoch to provide a fast model update.
            if i >= 50:
                print("Limiter: Reached 50 random batches for this epoch. Breaking to save time.")
                break

    # Save ONLY the feature extractor part for Phase 3
    print(f"Saving CNN feature extractor to {save_path}...")
    torch.save(full_model.extractor.state_dict(), save_path)
    print("CNN Training Complete.")
    return full_model.extractor

def export_embeddings(extractor_model, data_dir, save_path="data/image_embeddings.pt"):
    """
    Exports the 2048-d embeddings for the GNN to use as node features.
    """
    print(f"--- Phase 2: Exporting Image Embeddings ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor_model.eval().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_path = os.path.join(data_dir, "train")
    if not os.path.exists(train_path):
        return
        
    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    # Use smaller batch size, shuffle=True to get a uniform random distribution of all classes
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            embeddings = extractor_model(inputs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            
            if i % 20 == 0:
                print(f"Extracted {i * 32} / {len(dataset)} embeddings...")
                
            # Limit to 150 batches (~4800 images) distributed across all classes
            if i >= 150:
                print("Limiter: Exported 4800 random embeddings. Breaking.")
                break
                
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    torch.save({"embeddings": all_embeddings, "labels": all_labels}, save_path)
    print(f"Saved {len(all_embeddings)} embeddings to {save_path}")

def train_gnn(dataset_path, save_path="models/hybrid_gnn_fusion.pth", epochs=50, lr=0.01):
    print(f"--- Phase 3: Training Graph Neural Network (GCN) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Run construct_pyg_dataset.py first.")
        return
        
    # PyTorch 2.6+ unpickling bypass for PyG Data objects
    data, slices = torch.load(dataset_path, weights_only=False)
    from data.construct_pyg_dataset import CropGraphDataset
    dataset = CropGraphDataset(root=os.path.dirname(os.path.dirname(dataset_path)), 
                               img_embeddings_path="", env_data_path="", schema_path="")
    print(f"Loaded Graph Dataset: {len(dataset)} graphs.")
    
    from torch_geometric.loader import DataLoader
    from models.gnn_fusion import GNNFusion
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Init GNN (Input dims: 2048 from CNN/Env projection, Output: 38 arbitrary plant disease classes)
    model = GNNFusion(in_channels=2048, hidden_channels=256, out_channels=38, model_type='gcn').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out_logits, stress_scores = model(data.x, data.edge_index, batch=data.batch)
            
            # Compute loss
            loss = criterion(out_logits, data.y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.num_graphs
            
            pred = out_logits.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
            
        train_acc = correct / total
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.4f}")
            
    print(f"Saving Hybrid GNN Model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("GNN Training Complete.")

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_data_dir = os.path.join(base_dir, "plant_disease_dataset", "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)")
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    
    extractor_path = os.path.join(models_dir, "cnn_feature_extractor.pth")
    embeddings_path = os.path.join(data_dir, "image_embeddings.pt")
    gnn_dataset_path = os.path.join(data_dir, "processed", "crop_graph_dataset.pt")
    gnn_model_path = os.path.join(models_dir, "hybrid_gnn_fusion.pth")
    
    # 1. Train CNN
    trained_extractor = train_cnn(img_data_dir, epochs=1, save_path=extractor_path)
    
    # 2. Export Embeddings
    if trained_extractor:
        export_embeddings(trained_extractor, img_data_dir, save_path=embeddings_path)
        
        # 2.5 Rebuild the Graph Dataset forcefully
        if os.path.exists(gnn_dataset_path):
            os.remove(gnn_dataset_path)
        print("Rebuilding PyG Graph Dataset with new embeddings...")
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.construct_pyg_dataset import CropGraphDataset
        
        env_data_path = os.path.join(data_dir, "Crop_recommendation_normalized.csv")
        schema_path = os.path.join(data_dir, "graph_schema.pkl")
        CropGraphDataset(data_dir, embeddings_path, env_data_path, schema_path)
        
    # 3. Train GNN
    train_gnn(gnn_dataset_path, save_path=gnn_model_path)
