import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import cv2

# Ensure proper paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_extractor import CNNFeatureExtractor
from xai.grad_cam import GradCAM, display_gradcam

def test_gradcam(image_path, model_path, output_path="xai/sample_gradcam.jpg"):
    print(f"Testing Grad-CAM on {image_path}...")
    
    # 1. Load Model
    model = CNNFeatureExtractor(backbone='resnet50', pretrained=False)
    # The models/cnn_extractor.pth has the `extractor` layers saved
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded fine-tuned CNN weights.")
    else:
        print("Warning: Fine-tuned weights not found. Using untraiend CNN parameters for test.")
        
    model.eval()

    # 2. Setup GradCAM on the last convolutional layer of ResNet
    # In ResNet50, layer4 is the last residual block
    target_layer = model.extractor[-2] # extractor is Sequential(*list(resnet.children())[:-1]), layer4 is index -2, avgpool is -1
    cam = GradCAM(model, target_layer)

    # 3. Load & Process Image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)

    # 4. Generate Heatmap
    # For extraction, we don't have a classification head, so we just take the max activation 
    # as the target "class" to simulate the visualization of salient features.
    
    # We pass the class_idx=None to gradcam so it handles the argmax itself
    heatmap = cam(input_tensor, class_idx=None)  
    
    # 5. Overlay and Save
    result_img = display_gradcam(input_tensor, heatmap)
    
    # Convert RGB to BGR for OpenCV saving
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result_img_bgr)
    print(f"Grad-CAM visualization saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Needs an image. Extract one from the training set
    img_dir = os.path.join(base_dir, "plant_disease_dataset", "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "train")
    
    # Find any first valid image
    test_img = None
    for root_dir, dirs, files in os.walk(img_dir):
        for f in files:
            if f.endswith(".JPG") or f.endswith(".jpg"):
                test_img = os.path.join(root_dir, f)
                break
        if test_img:
            break
        
    if not test_img:
        print("Error: Could not find a test image in the dataset.")
    else:
        model_path = os.path.join(base_dir, "models", "cnn_feature_extractor.pth")
        out_path = os.path.join(base_dir, "xai", "sample_gradcam.jpg")
        test_gradcam(test_img, model_path, out_path)
