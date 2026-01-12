import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2

# Page config
st.set_page_config(
    page_title="SeedScout - Plant Seedling Classifier",
    page_icon="🌱",
    layout="wide"
)

# ---------------- MODEL LOADING (CACHED) ----------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 12
    
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load("best_resnet18_plant.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, device = load_model()

# ---------------- CONSTANTS ----------------
class_names = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
    "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
    "Sugar beet"
]

# Performance data for each species
species_accuracy = {
    "Black-grass": 67, "Charlock": 99, "Cleavers": 99,
    "Common Chickweed": 98, "Common wheat": 98, "Fat Hen": 97,
    "Loose Silky-bent": 88, "Maize": 99, "Scentless Mayweed": 99,
    "Shepherds Purse": 98, "Small-flowered Cranesbill": 100,
    "Sugar beet": 99
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- GRAD-CAM FUNCTION ----------------
def generate_gradcam(model, input_tensor, target_class, device):
    """Generate Grad-CAM heatmap"""
    try:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Hook to capture feature maps
        features = []
        gradients = []
        
        def forward_hook(module, input, output):
            features.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks on layer4 (last conv layer)
        forward_handle = model.layer4.register_forward_hook(forward_hook)
        backward_handle = model.layer4.register_full_backward_hook(backward_hook)
        
        # Forward pass
        output = model(input_tensor)
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Get feature maps and gradients
        feature_map = features[0].squeeze(0)
        gradient = gradients[0].squeeze(0)
        
        # Calculate weights
        weights = torch.mean(gradient, dim=(1, 2))
        
        # Generate heatmap
        cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * feature_map[i]
        
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM generation failed: {e}")
        return None

# ---------------- HEADER ----------------
st.title("🌱 SeedScout - Plant Seedling Classifier")
st.caption("AI-Powered Weed Identification | Powered by ResNet18")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📤 Batch Upload", "📚 Supported Species", "ℹ️ About"])

with tab1:
    st.markdown("### Upload Multiple Images for Batch Classification")
    
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        low_conf_threshold = st.slider(
            "Low Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Predictions below this will be flagged as low confidence"
        )
    
    with col_settings2:
        show_gradcam = st.checkbox(
            "Show Grad-CAM heatmaps",
            value=False,
            help="Visual explanation of model attention (slower processing)"
        )
    
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more seedling images"
    )
    
    if uploaded_files:
        st.info(f"📊 Processing {len(uploaded_files)} image(s)...")
        
        # Summary metrics
        total_images = len(uploaded_files)
        high_conf_count = 0
        low_conf_count = 0
        all_confidences = []
        csv_data = []
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Process each image
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Load and preprocess image
                image = Image.open(uploaded_file).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    top3_prob, top3_idx = torch.topk(probs, 3)
                
                # Extract values
                top1_conf = top3_prob[0][0].item()
                top1_class = class_names[top3_idx[0][0].item()]
                
                # Update statistics
                all_confidences.append(top1_conf)
                if top1_conf >= low_conf_threshold:
                    high_conf_count += 1
                else:
                    low_conf_count += 1
                
                # CSV data
                csv_data.append({
                    "filename": uploaded_file.name,
                    "predicted_class": top1_class,
                    "confidence": f"{top1_conf:.2%}",
                    "model_accuracy": f"{species_accuracy[top1_class]}%",
                    "top2_class": class_names[top3_idx[0][1].item()],
                    "top2_confidence": f"{top3_prob[0][1].item():.2%}",
                    "top3_class": class_names[top3_idx[0][2].item()],
                    "top3_confidence": f"{top3_prob[0][2].item():.2%}",
                })
                
                # Display result
                st.divider()
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, use_column_width=True)
                    st.caption(f"📁 {uploaded_file.name}")
                    
                    # Grad-CAM
                    if show_gradcam:
                        with st.spinner("Generating Grad-CAM..."):
                            cam = generate_gradcam(
                                model,
                                input_tensor,
                                top3_idx[0][0].item(),
                                device
                            )
                            
                            if cam is not None:
                                # Resize and overlay
                                cam_resized = cv2.resize(cam, (image.size[0], image.size[1]))
                                heatmap = cv2.applyColorMap(
                                    np.uint8(255 * cam_resized),
                                    cv2.COLORMAP_JET
                                )
                                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                
                                # Overlay on original
                                overlay = np.array(image) * 0.5 + heatmap * 0.5
                                st.image(
                                    overlay.astype(np.uint8),
                                    caption="🔍 Model Attention (Grad-CAM)",
                                    use_column_width=True
                                )
                
                with col2:
                    # Main prediction
                    st.markdown(f"### 🎯 Prediction: **{top1_class}**")
                    
                    # Confidence with color coding
                    if top1_conf >= 0.9:
                        st.success(f"**Confidence:** {top1_conf:.1%} 🟢 High")
                    elif top1_conf >= low_conf_threshold:
                        st.info(f"**Confidence:** {top1_conf:.1%} 🟡 Good")
                    else:
                        st.error(f"**Confidence:** {top1_conf:.1%} 🔴 Low")
                    
                    # Model accuracy for this species
                    acc = species_accuracy[top1_class]
                    st.metric("Model Accuracy (Test Set)", f"{acc}%")
                    
                    # Warnings
                    if top1_class == "Black-grass":
                        st.warning("⚠️ **Black-grass** has lower accuracy (67%). Verify independently.")
                    elif top1_class == "Loose Silky-bent":
                        st.info("ℹ️ **Loose Silky-bent** has good accuracy (88%) but may confuse with other grasses.")
                    
                    if top1_conf < low_conf_threshold:
                        st.error("""
                        🚨 **Low Confidence Warning**
                        
                        This prediction is uncertain. Possible reasons:
                        - Plant may not be one of 12 supported species
                        - Image quality issues
                        - Plant not at seedling stage
                        """)
                    
                    # Top 3
                    st.markdown("**Top 3 Predictions:**")
                    for i in range(3):
                        cls = class_names[top3_idx[0][i].item()]
                        prob = top3_prob[0][i].item()
                        st.progress(prob, text=f"{i+1}. {cls}: {prob:.1%}")
            
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            # Update progress
            progress_bar.progress((idx + 1) / total_images)
        
        progress_bar.empty()
        
        # ---------------- SUMMARY ----------------
        st.divider()
        st.markdown("## 📊 Batch Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", total_images)
        with col2:
            st.metric("High Confidence", high_conf_count, delta=f"{high_conf_count/total_images:.0%}")
        with col3:
            st.metric("Low Confidence", low_conf_count, delta=f"{low_conf_count/total_images:.0%}")
        with col4:
            avg_conf = np.mean(all_confidences)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        # ---------------- CSV DOWNLOAD ----------------
        st.divider()
        df = pd.DataFrame(csv_data)
        
        st.markdown("### 📥 Download Results")
        col_preview, col_download = st.columns([2, 1])
        
        with col_preview:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with col_download:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name='seedscout_predictions.csv',
                mime='text/csv',
                )
            
            st.caption(f"📄 {len(df)} predictions ready")

with tab2:
    st.header("📚 Supported Species (12 Total)")
    
    st.markdown("""
    This model can identify the following plant seedlings at early growth stages:
    """)
    
    # Create dataframe for species
    species_df = pd.DataFrame({
        'Species': class_names,
        'Model Accuracy': [f"{species_accuracy[name]}%" for name in class_names],
        'Status': [
            '⚠️ Lower Accuracy' if species_accuracy[name] < 70
            else '✅ Good' if species_accuracy[name] < 95
            else '✅ Excellent'
            for name in class_names
        ]
    })
    
    st.dataframe(species_df, use_container_width=True, hide_index=True)
    
    st.warning("""
    **⚠️ Important Limitations:**
    - Only works for these 12 species
    - Only seedling stage (1-4 weeks old)
    - Requires clear, single-plant images
    - Best with natural lighting
    """)

with tab3:
    st.header("ℹ️ About SeedScout")
    
    st.markdown("""
    ### 🌱 Purpose
    SeedScout helps identify common weed and crop seedlings using deep learning.
    
    ### 🤖 Technology
    - **Model:** ResNet18 (18-layer deep residual network)
    - **Framework:** PyTorch
    - **Dataset:** Aarhus University Plant Seedlings Dataset (~5,500 images)
    - **Overall Accuracy:** ~94%
    
    ### 📸 Best Practices
    For optimal results:
    1. Capture seedlings at early growth stage (1-4 weeks)
    2. Use good lighting (natural daylight preferred)
    3. Center single plant in frame
    4. Avoid shadows and reflections
    5. Ensure plant is clearly visible
    
    ### 🔍 Grad-CAM Feature
    When enabled, Grad-CAM (Gradient-weighted Class Activation Mapping) shows which parts
    of the image the model focused on when making its prediction. This helps verify the
    model is looking at relevant features.
    
    ### ⚖️ Disclaimer
    This tool is for educational and informational purposes. Always verify results,
    especially for critical agricultural decisions. Black-grass predictions should
    always be independently verified due to lower model accuracy.
    """)

# Footer
st.divider()
st.caption("🌱 SeedScout v2.0 | ResNet18 | Aarhus Dataset | Built with PyTorch & Streamlit")
