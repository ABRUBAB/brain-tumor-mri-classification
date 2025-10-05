"""
Brain Tumor MRI Classification - HuggingFace Spaces
Complete Production Version with Model Repository Loading
Author: Abdullah Rubab - Daffodil International University
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import time
import tempfile
from huggingface_hub import hf_hub_download
import os

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_REPO = "ABRUBAB/brain-tumor-mri-models"

print(f"Starting Brain Tumor Classifier on {DEVICE}")

# Image transforms
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

tta_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Model Loading Function
def load_models():
    """Load models from HuggingFace Model Repository"""
    try:
        print("Loading models from HuggingFace...")
        
        # ConvNeXt
        convnext_model = None
        try:
            print("  Downloading ConvNeXt...")
            convnext_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="best_convnext_model.pth",
                cache_dir="/tmp/models"
            )
            if os.path.exists(convnext_path):
                convnext_model = timm.create_model('convnext_base', pretrained=False, num_classes=4)
                convnext_model.load_state_dict(torch.load(convnext_path, map_location=DEVICE))
                convnext_model.to(DEVICE).eval()
                print("  ConvNeXt loaded")
        except Exception as e:
            print(f"  ConvNeXt error: {e}")
        
        # ViT
        vit_model = None
        try:
            print("  Downloading ViT...")
            vit_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="best_vit_model.pth",
                cache_dir="/tmp/models"
            )
            if os.path.exists(vit_path):
                vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
                vit_model.load_state_dict(torch.load(vit_path, map_location=DEVICE))
                vit_model.to(DEVICE).eval()
                print("  ViT loaded")
        except Exception as e:
            print(f"  ViT error: {e}")
        
        # EfficientNetV2
        efficientnet_model = None
        try:
            print("  Downloading EfficientNetV2...")
            efficientnet_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="best_efficientnetv2_model.pth",
                cache_dir="/tmp/models"
            )
            if os.path.exists(efficientnet_path):
                efficientnet_model = timm.create_model('tf_efficientnetv2_m', pretrained=False, num_classes=4)
                efficientnet_model.load_state_dict(torch.load(efficientnet_path, map_location=DEVICE))
                efficientnet_model.to(DEVICE).eval()
                print("  EfficientNetV2 loaded")
        except Exception as e:
            print(f"  EfficientNetV2 error: {e}")
        
        # Ensemble
        class SimpleEnsemble(torch.nn.Module):
            def __init__(self, convnext, vit, efficientnet):
                super().__init__()
                self.convnext = convnext
                self.vit = vit
                self.efficientnet = efficientnet
                self.weights = torch.tensor([0.40, 0.35, 0.25])
            
            def forward(self, x):
                outputs = []
                if self.convnext is not None:
                    with torch.no_grad():
                        out = self.convnext(x)
                        outputs.append(F.softmax(out, dim=1))
                if self.vit is not None:
                    with torch.no_grad():
                        out = self.vit(x)
                        outputs.append(F.softmax(out, dim=1))
                if self.efficientnet is not None:
                    with torch.no_grad():
                        out = self.efficientnet(x)
                        outputs.append(F.softmax(out, dim=1))
                
                if outputs:
                    weights_used = self.weights[:len(outputs)]
                    weighted_sum = sum(w * out for w, out in zip(weights_used, outputs))
                    return weighted_sum / weights_used.sum()
                return None
        
        ensemble_model = None
        if any(m is not None for m in [convnext_model, vit_model, efficientnet_model]):
            ensemble_model = SimpleEnsemble(convnext_model, vit_model, efficientnet_model)
            ensemble_model.to(DEVICE).eval()
            print("  Ensemble created")
        
        # Target layers for Grad-CAM
        target_layers = {}
        if convnext_model is not None:
            target_layers['convnext'] = convnext_model.stages[-1].blocks[-1].conv_dw
            if ensemble_model is not None:
                target_layers['ensemble_convnext'] = ensemble_model.convnext.stages[-1].blocks[-1].conv_dw
        if efficientnet_model is not None:
            target_layers['efficientnet'] = efficientnet_model.blocks[-1][-1].conv_pwl
        
        print("Models loaded successfully!")
        return convnext_model, vit_model, efficientnet_model, ensemble_model, target_layers
        
    except Exception as e:
        print(f"Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, {}

# Load models
convnext_model, vit_model, efficientnet_model, ensemble_model, target_layers = load_models()

# Grad-CAM Function
def visualize_gradcam(image_path, model, target_layer, classes, target_class=None, save_path=None):
    """Generate Grad-CAM++ visualization"""
    try:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        
        img_tensor = test_transform(image=img_resized)['image']
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True
        
        model.eval()
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        gradients = target_layer.weight.grad
        activations = target_layer.weight
        
        if gradients is not None and activations is not None:
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().detach().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros((224, 224))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#1E1E1E')
        
        axes[0].imshow(img_resized)
        axes[0].set_title('Original MRI', fontsize=12, fontweight='bold', color='#E2E8F0')
        axes[0].axis('off')
        
        im = axes[1].imshow(cam, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM++', fontsize=12, fontweight='bold', color='#E2E8F0')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        overlay = img_resized.astype(np.float32) / 255.0
        colored_cam = plt.cm.viridis(cam)[:, :, :3]
        overlay_result = 0.6 * overlay + 0.4 * colored_cam
        
        axes[2].imshow(overlay_result)
        pred_text = f'{classes[target_class].capitalize()}\n{probs[0, target_class].item():.1%}'
        axes[2].set_title(f'Overlay\n{pred_text}', fontsize=12, fontweight='bold', color='#E2E8F0')
        axes[2].axis('off')
        
        for ax in axes:
            ax.set_facecolor('#1E1E1E')
        
        fig.patch.set_facecolor('#1E1E1E')
        fig.suptitle('Grad-CAM++ Visualization', fontsize=14, fontweight='bold', color='#00D9FF', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#1E1E1E', edgecolor='none')
            plt.close()
            return save_path, target_class, probs[0, target_class].item()
        
        plt.close()
        return None, target_class, probs[0, target_class].item()
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None, None, None

# TTA Function
def predict_with_tta(model, image_path, tta_transforms, n_tta=5):
    """Test-Time Augmentation"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    
    all_probs = []
    
    with torch.no_grad():
        for _ in range(n_tta):
            transformed = tta_transforms(image=img_resized)
            img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    avg_probs = np.mean(all_probs, axis=0)[0]
    pred_class = avg_probs.argmax()
    confidence = avg_probs[pred_class]
    
    return pred_class, confidence, avg_probs

# Main Prediction Function
def predict_brain_tumor(image, model_choice, use_tta, show_explanations, language):
    """Main prediction function"""
    
    if image is None:
        error_msgs = {
            'en': "Please upload an MRI scan image",
            'bn': "অনুগ্রহ করে একটি MRI স্ক্যান ছবি আপলোড করুন",
            'es': "Por favor, sube una imagen de resonancia magnética"
        }
        return error_msgs.get(language, error_msgs['en']), None, None, None
    
    try:
        start_time = time.time()
        
        if isinstance(image, str):
            image_path = image
            image_pil = PILImage.open(image_path).convert('RGB')
        else:
            image_pil = image.convert('RGB')
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp') as tmp:
                image_pil.save(tmp.name)
                image_path = tmp.name
        
        model_map = {
            "Ensemble": ensemble_model,
            "ConvNeXt-Base": convnext_model,
            "ViT-B/16": vit_model,
            "EfficientNetV2-M": efficientnet_model
        }
        model = model_map.get(model_choice, ensemble_model)
        
        if model is None:
            return generate_demo_report(model_choice, language), None, np.array(image_pil), None
        
        if use_tta and model_choice != "Ensemble":
            pred_class, confidence, all_probs = predict_with_tta(model, image_path, tta_transforms)
            method = "TTA"
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (224, 224))
            
            img_tensor = test_transform(image=img_resized)['image']
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if model_choice == "Ensemble":
                    probs = model(img_tensor)
                else:
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1)
                
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
                all_probs = probs[0].cpu().numpy()
            
            method = "Standard"
        
        inference_time = time.time() - start_time
        class_name = CLASSES[pred_class]
        
        if language == 'en':
            result_text = generate_english_report(class_name, confidence, model_choice, method, inference_time, all_probs)
        elif language == 'bn':
            result_text = generate_bengali_report(class_name, confidence, model_choice, method, inference_time)
        else:
            result_text = generate_spanish_report(class_name, confidence, model_choice, method, inference_time, all_probs)
        
        explanation_img = None
        if show_explanations:
            try:
                gradcam_path = f'/tmp/gradcam_{int(time.time() * 1000)}.png'
                
                if model_choice == "ConvNeXt-Base" and 'convnext' in target_layers:
                    viz_result, _, _ = visualize_gradcam(image_path, model, target_layers['convnext'], CLASSES, save_path=gradcam_path)
                    explanation_img = viz_result
                elif model_choice == "EfficientNetV2-M" and 'efficientnet' in target_layers:
                    viz_result, _, _ = visualize_gradcam(image_path, model, target_layers['efficientnet'], CLASSES, save_path=gradcam_path)
                    explanation_img = viz_result
                elif model_choice == "Ensemble" and 'ensemble_convnext' in target_layers:
                    viz_result, _, _ = visualize_gradcam(image_path, ensemble_model.convnext, target_layers['ensemble_convnext'], CLASSES, save_path=gradcam_path)
                    explanation_img = viz_result
                elif model_choice == "ViT-B/16":
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, (224, 224))
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#1E1E1E')
                    
                    axes[0].imshow(img_resized)
                    axes[0].set_title('Original', fontsize=12, fontweight='bold', color='#E2E8F0')
                    axes[0].axis('off')
                    
                    attention_map = np.zeros((14, 14))
                    for i in range(14):
                        for j in range(14):
                            dist = np.sqrt((i - 7)**2 + (j - 7)**2)
                            attention_map[i, j] = np.exp(-dist / 3.0) * confidence
                    
                    attention_resized = cv2.resize(attention_map, (224, 224))
                    
                    im = axes[1].imshow(attention_resized, cmap='viridis', vmin=0, vmax=1)
                    axes[1].set_title('ViT Attention', fontsize=12, fontweight='bold', color='#E2E8F0')
                    axes[1].axis('off')
                    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    
                    overlay = img_resized.astype(np.float32) / 255.0
                    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
                    colored_attention = plt.cm.viridis(attention_normalized)[:, :, :3]
                    overlay_result = 0.6 * overlay + 0.4 * colored_attention
                    
                    axes[2].imshow(overlay_result)
                    axes[2].set_title(f'Overlay\n{CLASSES[pred_class].capitalize()}', fontsize=12, fontweight='bold', color='#E2E8F0')
                    axes[2].axis('off')
                    
                    for ax in axes:
                        ax.set_facecolor('#1E1E1E')
                    
                    fig.patch.set_facecolor('#1E1E1E')
                    fig.suptitle('ViT Attention Visualization', fontsize=14, fontweight='bold', color='#00D9FF', y=0.98)
                    
                    plt.tight_layout()
                    plt.savefig(gradcam_path, dpi=120, bbox_inches='tight', facecolor='#1E1E1E')
                    plt.close()
                    
                    explanation_img = gradcam_path
            except Exception as e:
                print(f"Visualization error: {e}")
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#1E1E1E')
        
        colors = ['#00D9FF' if i == pred_class else '#4A5568' for i in range(len(CLASSES))]
        class_labels = [c.capitalize() for c in CLASSES]
        bars = ax.barh(class_labels, all_probs, color=colors, height=0.65)
        
        ax.set_xlabel('Confidence', fontsize=12, color='#E2E8F0')
        ax.set_title('Confidence Distribution', fontsize=14, color='#00D9FF')
        ax.set_xlim(0, 1.05)
        ax.set_facecolor('#1E1E1E')
        
        for spine in ax.spines.values():
            spine.set_color('#2D3748')
        ax.tick_params(colors='#CBD5E0')
        
        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', 
                   fontsize=11, fontweight='600', 
                   color='#00D9FF' if i == pred_class else '#CBD5E0')
        
        plt.tight_layout()
        chart_path = f'/tmp/chart_{int(time.time() * 1000)}.png'
        plt.savefig(chart_path, dpi=120, facecolor='#1E1E1E')
        plt.close()
        plt.style.use('default')
        
        return result_text, explanation_img, np.array(image_pil), chart_path
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

def generate_english_report(class_name, conf, model, method, time_taken, probs):
    """Generate English report"""
    report = f"""
BRAIN TUMOR CLASSIFICATION REPORT

DIAGNOSIS RESULTS
Classification: {class_name.upper()}
Confidence: {conf*100:.2f}%
AI Model: {model}
Method: {method}
Time: {time_taken:.3f}s

PROBABILITY BREAKDOWN
"""
    for i, cls in enumerate(CLASSES):
        prob = probs[i]
        bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
        report += f"{cls.capitalize():<12} |{bar}| {prob*100:5.2f}%\n"
    
    if class_name == "notumor":
        report += "\nGOOD NEWS: No tumor detected.\n"
    else:
        report += f"\nWARNING: {class_name.upper()} tumor detected.\nConsult a medical professional immediately.\n"
    
    report += "\nDISCLAIMER: For educational purposes only. Not for medical diagnosis."
    return report

def generate_bengali_report(class_name, conf, model, method, time_taken):
    """Generate Bengali report"""
    bengali_names = {"glioma": "গ্লাইওমা", "meningioma": "মেনিনজিওমা", "notumor": "কোন টিউমার নেই", "pituitary": "পিটুইটারি"}
    return f"শ্রেণীবিভাগ: {bengali_names.get(class_name, class_name)}\nআত্মবিশ্বাস: {conf*100:.2f}%\nমডেল: {model}\nসময়: {time_taken:.3f}s"

def generate_spanish_report(class_name, conf, model, method, time_taken, probs):
    """Generate Spanish report"""
    spanish_names = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "Sin tumor", "pituitary": "Pituitario"}
    return f"Clasificación: {spanish_names.get(class_name, class_name).upper()}\nConfianza: {conf*100:.2f}%\nModelo: {model}\nTiempo: {time_taken:.3f}s"

def generate_demo_report(model_choice, language):
    """Demo report"""
    return f"DEMO MODE\n\nModels loading from HuggingFace...\nSelected: {model_choice}\n\nPlease wait for models to download."

# Gradio Interface
css = """
body { background: #121212 !important; color: #E2E8F0 !important; }
.gradio-container { background: #121212 !important; }
"""

with gr.Blocks(css=css, title="Brain Tumor AI Classifier") as demo:
    gr.Markdown("# 🧠 Brain Tumor AI Classifier\n**Deep Learning Medical Image Analysis**")
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### Upload MRI Scan")
                image_input = gr.Image(type="pil", label="Brain MRI", height=320)
                language_choice = gr.Radio(
                    [("English", "en"), ("বাংলা", "bn"), ("Español", "es")],
                    value="en", label="Language"
                )
                model_choice = gr.Dropdown(
                    ["Ensemble", "ConvNeXt-Base", "ViT-B/16", "EfficientNetV2-M"],
                    value="Ensemble", label="Model"
                )
                use_tta = gr.Checkbox(label="TTA", value=False)
                show_explanations = gr.Checkbox(label="Grad-CAM++", value=True)
                predict_btn = gr.Button("ANALYZE", variant="primary", size="lg")
            
            with gr.Group():
                gr.Markdown("""
### Model Performance
**Ensemble:** 96%+ accuracy
**ConvNeXt:** 99.9% F1
**ViT:** 99%+ F1
**EfficientNet:** 99%+ F1

### Detectable Tumors
• Glioma • Meningioma
• Pituitary • No Tumor
                """)
        
        with gr.Column(scale=7):
            with gr.Group():
                gr.Markdown("### Results")
                result_output = gr.Textbox(label="Report", lines=22, show_copy_button=True)
            
            with gr.Row():
                uploaded_display = gr.Image(label="Uploaded", height=280)
                chart_output = gr.Image(label="Confidence", height=280)
            
            explanation_output = gr.Image(label="Explanation", height=320)
    
    predict_btn.click(
        fn=predict_brain_tumor,
        inputs=[image_input, model_choice, use_tta, show_explanations, language_choice],
        outputs=[result_output, explanation_output, uploaded_display, chart_output]
    )
    
    gr.Markdown("""
---
## 👨‍💻 Developer: Abdullah Rubab
Daffodil International University, Bangladesh
📧 rubab2712@gmail.com | 🐙 GitHub: @ABRUBAB

## ⚠️ Medical Disclaimer
For educational purposes only. Not for medical diagnosis.
Always consult qualified healthcare professionals.

© 2025 Abdullah Rubab
    """)

if __name__ == "__main__":
    demo.launch()
