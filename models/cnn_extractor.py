import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        if backbone == 'resnet50':
            # Load pretrained ResNet50
            resnet = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer to output a 2048-d feature vector
            self.extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.output_dim = 2048
        elif backbone == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.extractor = mobilenet.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = 1280
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'mobilenet_v2'.")

    def forward(self, x):
        features = self.extractor(x)
        if hasattr(self, 'pool'):
            features = self.pool(features)
        # Flatten the features: (Batch_Size, Output_Dim, 1, 1) -> (Batch_Size, Output_Dim)
        features = features.view(features.size(0), -1)
        return features

def process_image(image_path):
    """
    Standard image preprocessing for ResNet50/MobileNetV2.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return preprocess(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Quick test
if __name__ == "__main__":
    model = CNNFeatureExtractor('resnet50')
    print(f"Model initialized with output dimension: {model.output_dim}")
    # dummy_input = torch.randn(1, 3, 224, 224)
    # output = model(dummy_input)
    # print(f"Output shape: {output.shape}")
