# Adaptive Explainable AI System for Smart Crop Health Monitoring 🌱

A Graph-based Multi-Modal approach to predicting crop diseases. This system combines Convolutional Neural Networks (CNN) for image feature extraction and Graph Neural Networks (GNN) for environmental data fusion to provide robust, explainable predictions about crop health.

## 🚀 Features
- **Hybrid AI Model**: Combines ResNet50 (CNN) and Graph Convolutional Networks (GCN).
- **Environmental Context**: Automatically fetches real-time climate (temperature, humidity, rainfall) via Open-Meteo and soil conditions (pH, Nitrogen) via ISRIC SoilGrids.
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to visually show which part of the leaf is infected.
- **Bilingual Farmer Advisory**: Provides actionable advice and reasons in both English and Hindi.
- **Autonomous Retraining**: The model intelligently logs and retrains itself on new data over time.

## 💻 Installation and Deployment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit web application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🤝 Acknowledgments & Credits
This project was developed by **[Your Name]**.

Special thanks and credits to my friend **[Friend's Name]** for their invaluable help and contributions to making this project a reality! Their support in [mention what they helped with, e.g., testing, dataset collection, UI design] was greatly appreciated.
