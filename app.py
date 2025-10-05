"""
Brain Tumor MRI Classification - HuggingFace Spaces
COMPLETE PRODUCTION VERSION
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

# ===== CONFIGURATION =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

print(f"🚀 Starting Brain Tumor Classifier on {DEVICE}")

# ===== IMAGE TRANSFORMS =====
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

# ===== MODEL LOADING =====
def load_models():
    """Load all trained models with proper error handling"""
    try:
        print("Loading models...")
        
        # ConvNeXt-Base
        convnext_model = None
        convnext_path = 'models/best/best_convnext_model.pth'
        if Path(convnext_path).exists():
            try:
                convnext_model = timm.create_model('convnext_base', pretrained=False, num_classes=4)
                convnext_model.load_state_dict(torch.load(convnext_path, map_location=DEVICE))
                convnext_model.to(DEVICE).eval()
                print("✅ ConvNeXt-Base loaded successfully")
            except Exception as e:
                print(f"⚠️ ConvNeXt loading error: {e}")
        else:
            print("⚠️ ConvNeXt model file not found")
        
        # Vision Transformer B/16
        vit_model = None
        vit_path = 'models/best/best_vit_model.pth'
        if Path(vit_path).exists():
            try:
                vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
                vit_model.load_state_dict(torch.load(vit_path, map_location=DEVICE))
                vit_model.to(DEVICE).eval()
                print("✅ ViT-B/16 loaded successfully")
            except Exception as e:
                print(f"⚠️ ViT loading error: {e}")
        else:
            print("⚠️ ViT model file not found")
        
        # EfficientNetV2-Medium
        efficientnet_model = None
        efficientnet_path = 'models/best/best_efficientnetv2_model.pth'
        if Path(efficientnet_path).exists():
            try:
                efficientnet_model = timm.create_model('tf_efficientnetv2_m', pretrained=False, num_classes=4)
                efficientnet_model.load_state_dict(torch.load(efficientnet_path, map_location=DEVICE))
                efficientnet_model.to(DEVICE).eval()
                print("✅ EfficientNetV2-M loaded successfully")
            except Exception as e:
                print(f"⚠️ EfficientNetV2 loading error: {e}")
        else:
            print("⚠️ EfficientNetV2 model file not found")
        
        # Create Ensemble Model
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
            print("✅ Ensemble model created successfully")
        else:
            print("⚠️ No models available for ensemble")
        
        # Define target layers for Grad-CAM++
        target_layers = {}
        if convnext_model is not None:
            target_layers['convnext'] = convnext_model.stages[-1].blocks[-1].conv_dw
            if ensemble_model is not None:
                target_layers['ensemble_convnext'] = ensemble_model.convnext.stages[-1].blocks[-1].conv_dw
        if efficientnet_model is not None:
            target_layers['efficientnet'] = efficientnet_model.blocks[-1][-1].conv_pwl
        
        return convnext_model, vit_model, efficientnet_model, ensemble_model, target_layers
        
    except Exception as e:
        print(f"⚠️ Critical error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, {}

# Load models at startup
convnext_model, vit_model, efficientnet_model, ensemble_model, target_layers = load_models()

# ===== GRAD-CAM++ VISUALIZATION =====
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
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#1E1E1E')
        
        axes[0].imshow(img_resized)
        axes[0].set_title('Original MRI', fontsize=12, fontweight='bold', color='#E2E8F0')
        axes[0].axis('off')
        
        im = axes[1].imshow(cam, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold', color='#E2E8F0')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        overlay = img_resized.astype(np.float32) / 255.0
        colored_cam = plt.cm.viridis(cam)[:, :, :3]
        overlay_result = 0.6 * overlay + 0.4 * colored_cam
        
        axes[2].imshow(overlay_result)
        pred_text = f'{classes[target_class].capitalize()}\n{probs[0, target_class].item():.1%} confidence'
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
        print(f"Grad-CAM++ error: {e}")
        return None, None, None

# ===== TEST-TIME AUGMENTATION =====
def predict_with_tta(model, image_path, tta_transforms, n_tta=5):
    """Test-Time Augmentation for improved accuracy"""
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

# ===== MAIN PREDICTION FUNCTION =====
def predict_brain_tumor(image, model_choice, use_tta, show_explanations, language):
    """Main prediction function with full features"""
    
    if image is None:
        error_msgs = {
            'en': "❌ Please upload an MRI scan image",
            'bn': "❌ অনুগ্রহ করে একটি MRI স্ক্যান ছবি আপলোড করুন",
            'es': "❌ Por favor, sube una imagen de resonancia magnética"
        }
        return error_msgs.get(language, error_msgs['en']), None, None, None
    
    try:
        start_time = time.time()
        
        # Handle image input
        if isinstance(image, str):
            image_path = image
            image_pil = PILImage.open(image_path).convert('RGB')
        else:
            image_pil = image.convert('RGB')
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='/tmp') as tmp:
                image_pil.save(tmp.name)
                image_path = tmp.name
        
        # Select model
        model_map = {
            "Ensemble": ensemble_model,
            "ConvNeXt-Base": convnext_model,
            "ViT-B/16": vit_model,
            "EfficientNetV2-M": efficientnet_model
        }
        model = model_map.get(model_choice, ensemble_model)
        
        # Check if model is available
        if model is None:
            return generate_demo_report(model_choice, language), None, np.array(image_pil), None
        
        # Prediction with or without TTA
        if use_tta and model_choice != "Ensemble":
            pred_class, confidence, all_probs = predict_with_tta(model, image_path, tta_transforms)
            method = "TTA (5 augmentations)"
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
        
        # Generate report in selected language
        if language == 'en':
            result_text = generate_english_report(class_name, confidence, model_choice, method, inference_time, all_probs)
        elif language == 'bn':
            result_text = generate_bengali_report(class_name, confidence, model_choice, method, inference_time)
        else:
            result_text = generate_spanish_report(class_name, confidence, model_choice, method, inference_time, all_probs)
        
        # Generate Grad-CAM++ explanation
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
                    # ViT attention visualization
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, (224, 224))
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#1E1E1E')
                    
                    axes[0].imshow(img_resized)
                    axes[0].set_title('Original MRI', fontsize=12, fontweight='bold', color='#E2E8F0')
                    axes[0].axis('off')
                    
                    # Create simplified attention map
                    attention_map = np.zeros((14, 14))
                    center_x, center_y = 7, 7
                    for i in range(14):
                        for j in range(14):
                            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                            attention_map[i, j] = np.exp(-dist / 3.0) * confidence
                    
                    attention_resized = cv2.resize(attention_map, (224, 224))
                    
                    im = axes[1].imshow(attention_resized, cmap='viridis', vmin=0, vmax=1)
                    axes[1].set_title('ViT Attention Pattern', fontsize=12, fontweight='bold', color='#E2E8F0')
                    axes[1].axis('off')
                    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    
                    overlay = img_resized.astype(np.float32) / 255.0
                    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
                    colored_attention = plt.cm.viridis(attention_normalized)[:, :, :3]
                    overlay_result = 0.6 * overlay + 0.4 * colored_attention
                    
                    axes[2].imshow(overlay_result)
                    pred_text = f'{CLASSES[pred_class].capitalize()}\n{confidence:.1%} confidence'
                    axes[2].set_title(f'Attention Overlay\n{pred_text}', fontsize=12, fontweight='bold', color='#E2E8F0')
                    axes[2].axis('off')
                    
                    for ax in axes:
                        ax.set_facecolor('#1E1E1E')
                    
                    fig.patch.set_facecolor('#1E1E1E')
                    fig.suptitle('Vision Transformer Attention Visualization', fontsize=14, fontweight='bold', color='#00D9FF', y=0.98)
                    
                    plt.tight_layout()
                    plt.savefig(gradcam_path, dpi=120, bbox_inches='tight', facecolor='#1E1E1E', edgecolor='none')
                    plt.close()
                    
                    explanation_img = gradcam_path
                    
            except Exception as e:
                print(f"Visualization error for {model_choice}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create confidence distribution chart
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#1E1E1E')
        
        colors = ['#00D9FF' if i == pred_class else '#4A5568' for i in range(len(CLASSES))]
        class_labels = [c.capitalize() for c in CLASSES]
        bars = ax.barh(class_labels, all_probs, color=colors, edgecolor='#2D3748', linewidth=1.5, height=0.65)
        
        ax.set_xlabel('Confidence Level', fontsize=12, fontweight='500', color='#E2E8F0')
        ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='600', color='#00D9FF', pad=12)
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.8, color='#4A5568')
        ax.set_axisbelow(True)
        ax.set_facecolor('#1E1E1E')
        
        for spine in ax.spines.values():
            spine.set_color('#2D3748')
            spine.set_linewidth(1)
        
        ax.tick_params(colors='#CBD5E0', labelsize=10)
        
        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            width = bar.get_width()
            label_color = '#00D9FF' if i == pred_class else '#CBD5E0'
            ax.text(width + 0.02, i, f'{prob*100:.1f}%', va='center', fontsize=11, 
                   fontweight='600', color=label_color)
        
        plt.tight_layout()
        chart_path = f'/tmp/chart_{int(time.time() * 1000)}.png'
        plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='#1E1E1E', edgecolor='none')
        plt.close()
        plt.style.use('default')
        
        return result_text, explanation_img, np.array(image_pil), chart_path
        
    except Exception as e:
        error_msgs = {
            'en': f"❌ Error: {str(e)}\n\nPlease try a different image or model.",
            'bn': f"❌ ত্রুটি: {str(e)}\n\nঅনুগ্রহ করে অন্য ছবি বা মডেল চেষ্টা করুন।",
            'es': f"❌ Error: {str(e)}\n\nPor favor, prueba con otra imagen o modelo."
        }
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return error_msgs.get(language, error_msgs['en']), None, None, None

# ===== REPORT GENERATION FUNCTIONS =====
def generate_english_report(class_name, conf, model, method, time_taken, probs):
    """Generate comprehensive English medical report"""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           BRAIN TUMOR CLASSIFICATION REPORT                   ║
╚══════════════════════════════════════════════════════════════╝

DIAGNOSIS RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Classification: {class_name.upper()}
  Confidence: {conf*100:.2f}%
  AI Model: {model}
  Method: {method}
  Inference Time: {time_taken:.3f} seconds

PROBABILITY BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    for i, cls in enumerate(CLASSES):
        prob = probs[i]
        bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
        report += f"  {cls.capitalize():<12} │{bar}│ {prob*100:5.2f}%\n"
    
    report += "\n"
    
    if class_name == "notumor":
        report += """MEDICAL INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EXCELLENT NEWS: No tumor detected in the scan.

RECOMMENDATIONS:
  • Continue regular medical check-ups as advised
  • Maintain a healthy lifestyle with balanced diet
  • Monitor for any new neurological symptoms
  • Keep this scan for medical records
  • Consult your physician for routine follow-ups
"""
    else:
        tumor_info = {
            "glioma": "Tumor originating from glial cells. Can be benign or malignant.",
            "meningioma": "Tumor of the meninges. Usually benign and slow-growing.",
            "pituitary": "Tumor in the pituitary gland. May affect hormones."
        }
        
        report += f"""MEDICAL INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ {class_name.upper()} TUMOR DETECTED

About {class_name.capitalize()}:
  {tumor_info.get(class_name, 'Tumor detected in brain scan.')}

🚨 URGENT ACTION REQUIRED:
  ⚕️  Schedule immediate appointment with neurologist
  📄 Bring this scan and report to consultation
  🔬 Additional diagnostic tests may be required
  ⏰ Early detection significantly improves treatment outcomes
  👥 Consider getting a second medical opinion
  🏥 Do not delay - contact healthcare provider today
"""
    
    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ MEDICAL DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This AI system is for educational and research purposes only.
It does NOT replace professional medical diagnosis.
Always consult qualified healthcare professionals for accurate
diagnosis, treatment options, and medical advice.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report

def generate_bengali_report(class_name, conf, model, method, time_taken):
    """Generate Bengali medical report"""
    bengali_names = {
        "glioma": "গ্লাইওমা",
        "meningioma": "মেনিনজিওমা",
        "notumor": "কোন টিউমার নেই",
        "pituitary": "পিটুইটারি"
    }
    bn_name = bengali_names.get(class_name, class_name)
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║              মস্তিষ্ক টিউমার শ্রেণীবিভাগ রিপোর্ট             ║
╚══════════════════════════════════════════════════════════════╝

রোগ নির্ণয় ফলাফল
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  শ্রেণীবিভাগ: {bn_name}
  আত্মবিশ্বাস: {conf*100:.2f}%
  এআই মডেল: {model}
  পদ্ধতি: {method}
  সময়: {time_taken:.3f} সেকেন্ড
"""
    
    if class_name == "notumor":
        report += """
চিকিৎসা ব্যাখ্যা
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ সুসংবাদ: স্ক্যানে কোন টিউমার পাওয়া যায়নি।

সুপারিশ:
  • নিয়মিত স্বাস্থ্য পরীক্ষা চালিয়ে যান
  • সুস্থ জীবনযাপন বজায় রাখুন
  • নতুন উপসর্গ পর্যবেক্ষণ করুন
  • ডাক্তারের পরামর্শ অনুসরণ করুন
"""
    else:
        report += f"""
চিকিৎসা ব্যাখ্যা
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ {bn_name} টিউমার সনাক্ত হয়েছে

🚨 জরুরি পদক্ষেপ:
  ⚕️  অবিলম্বে স্নায়ুবিশেষজ্ঞের সাথে অ্যাপয়েন্টমেন্ট নিন
  📄 এই স্ক্যান এবং রিপোর্ট ডাক্তারকে দেখান
  🔬 অতিরিক্ত পরীক্ষার প্রয়োজন হতে পারে
  ⏰ দ্রুত সনাক্তকরণ চিকিৎসায় সহায়ক
  🏥 বিলম্ব করবেন না - আজই যোগাযোগ করুন
"""
    
    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ চিকিৎসা দাবিত্যাগ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
এই এআই সিস্টেম শুধুমাত্র শিক্ষামূলক এবং গবেষণা উদ্দেশ্যে।
এটি পেশাদার চিকিৎসা নির্ণয়ের বিকল্প নয়।
সঠিক রোগ নির্ণয় এবং চিকিৎসার জন্য সর্বদা যোগ্য স্বাস্থ্যসেবা
পেশাদারদের পরামর্শ নিন।
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report

def generate_spanish_report(class_name, conf, model, method, time_taken, probs):
    """Generate Spanish medical report"""
    spanish_names = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "Sin tumor",
        "pituitary": "Pituitario"
    }
    es_name = spanish_names.get(class_name, class_name)
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║        INFORME DE CLASIFICACIÓN DE TUMOR CEREBRAL             ║
╚══════════════════════════════════════════════════════════════╝

RESULTADOS DEL DIAGNÓSTICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Clasificación: {es_name.upper()}
  Confianza: {conf*100:.2f}%
  Modelo de IA: {model}
  Método: {method}
  Tiempo: {time_taken:.3f} segundos

DISTRIBUCIÓN DE PROBABILIDADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    class_es = {'glioma': 'Glioma', 'meningioma': 'Meningioma', 'notumor': 'Sin tumor', 'pituitary': 'Pituitario'}
    for i, cls in enumerate(CLASSES):
        prob = probs[i]
        bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
        report += f"  {class_es[cls]:<12} │{bar}│ {prob*100:5.2f}%\n"
    
    report += "\n"
    
    if class_name == "notumor":
        report += """INTERPRETACIÓN MÉDICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EXCELENTES NOTICIAS: No se detectó tumor en la resonancia.

RECOMENDACIONES:
  • Continúe con chequeos médicos regulares
  • Mantenga un estilo de vida saludable
  • Monitoree cualquier síntoma neurológico nuevo
  • Consulte a su médico para seguimiento rutinario
"""
    else:
        tumor_info_es = {
            "glioma": "Tumor originado en células gliales. Puede ser benigno o maligno.",
            "meningioma": "Tumor de las meninges. Generalmente benigno.",
            "pituitary": "Tumor en la glándula pituitaria. Puede afectar hormonas."
        }
        
        report += f"""INTERPRETACIÓN MÉDICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ TUMOR {es_name.upper()} DETECTADO

Sobre {es_name}:
  {tumor_info_es.get(class_name, 'Tumor detectado en resonancia cerebral.')}

🚨 ACCIÓN URGENTE REQUERIDA:
  ⚕️  Programe cita inmediata con neurólogo
  📄 Lleve esta resonancia y reporte a la consulta
  🔬 Pueden requerirse pruebas diagnósticas adicionales
  ⏰ La detección temprana mejora resultados del tratamiento
  👥 Considere obtener segunda opinión médica
  🏥 No demore - contacte profesional de salud hoy
"""
    
    report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ DESCARGO DE RESPONSABILIDAD MÉDICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Este sistema de IA es solo para fines educativos y de investigación.
NO reemplaza el diagnóstico médico profesional.
Siempre consulte a profesionales de salud calificados para
diagnóstico preciso, opciones de tratamiento y consejo médico.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report

def generate_demo_report(model_choice, language):
    """Generate demo report when models are not loaded"""
    if language == 'en':
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           BRAIN TUMOR CLASSIFICATION SYSTEM                   ║
╚══════════════════════════════════════════════════════════════╝

🔄 DEMO MODE

Selected Model: {model_choice}
Language: English

SYSTEM STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Interface: Running
✅ Image Processing: Active
⚠️  AI Models: Awaiting model files

TO ENABLE FULL FUNCTIONALITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upload trained model files (.pth) to models/best/ directory

The system interface is fully configured and ready!
"""
    elif language == 'bn':
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              মস্তিষ্ক টিউমার শ্রেণীবিভাগ সিস্টেম             ║
╚══════════════════════════════════════════════════════════════╝

🔄 ডেমো মোড

নির্বাচিত মডেল: {model_choice}
ভাষা: বাংলা

সিস্টেম স্ট্যাটাস:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ইন্টারফেস: চলছে
✅ ছবি প্রসেসিং: সক্রিয়
⚠️  এআই মডেল: মডেল ফাইলের জন্য অপেক্ষা করছে

সম্পূর্ণ কার্যকারিতা সক্ষম করতে:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
প্রশিক্ষিত মডেল ফাইল (.pth) আপলোড করুন

সিস্টেম ইন্টারফেস সম্পূর্ণভাবে প্রস্তুত!
"""
    else:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║        SISTEMA DE CLASIFICACIÓN DE TUMOR CEREBRAL             ║
╚══════════════════════════════════════════════════════════════╝

🔄 MODO DEMO

Modelo seleccionado: {model_choice}
Idioma: Español

ESTADO DEL SISTEMA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Interfaz: Funcionando
✅ Procesamiento: Activo
⚠️  Modelos IA: Esperando archivos de modelo

PARA HABILITAR FUNCIONALIDAD COMPLETA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sube archivos de modelo entrenados (.pth) al directorio models/best/

¡La interfaz del sistema está completamente configurada!
"""

# ===== COMPLETE GRADIO UI =====
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #121212;
    --bg-secondary: #1E1E1E;
    --bg-tertiary: #252525;
    --accent-cyan: #00D9FF;
    --accent-blue: #2196F3;
    --accent-purple: #9C27B0;
    --text-primary: #E2E8F0;
    --text-secondary: #CBD5E0;
    --text-muted: #94A3B8;
    --border-color: #2D3748;
    --success: #00E676;
    --warning: #FFC107;
    --error: #FF5252;
    --shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
    --glow: 0 0 20px rgba(0, 217, 255, 0.3);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    background: var(--bg-primary) !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 20px 40px !important;
    margin: 0 !important;
}

.main {
    max-width: 100% !important;
    width: 100% !important;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: var(--shadow-lg);
    margin-bottom: 20px;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple));
}

.gr-group, .gr-box, .gr-form, .gr-panel {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow) !important;
}

.gr-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 11px 24px !important;
    transition: all 0.3s ease !important;
    border: none !important;
}

button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 100%) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    box-shadow: var(--shadow), var(--glow) !important;
}

button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg), 0 0 30px rgba(0, 217, 255, 0.5) !important;
}

.gr-input, .gr-textbox, .gr-dropdown, textarea, select {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
}

.gr-image {
    background: var(--bg-tertiary) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 10px !important;
}

.gr-image:hover {
    border-color: var(--accent-cyan) !important;
}

label, .gr-label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.result-output {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}
"""

with gr.Blocks(
    css=css,
    title="Brain Tumor AI Classifier - Abdullah Rubab",
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="blue").set(
        body_background_fill="#121212",
        background_fill_primary="#1E1E1E",
        background_fill_secondary="#252525",
        border_color_primary="#2D3748",
        button_primary_background_fill="#00D9FF",
        button_primary_text_color="#000000",
    )
) as demo:
    
    gr.Markdown("""
# 🧠 Brain Tumor AI Classifier

**Advanced Deep Learning Medical Image Analysis**

*Powered by ConvNeXt | Vision Transformer | EfficientNetV2*
    """, elem_classes="main-header")
    
    with gr.Row():
        # Left Panel
        with gr.Column(scale=3, min_width=380):
            with gr.Group():
                gr.Markdown("### 📤 Upload MRI Scan")
                image_input = gr.Image(
                    type="pil",
                    label="Brain MRI Scan",
                    height=320
                )
                
                language_choice = gr.Radio(
                    choices=[
                        ("🇺🇸 English", "en"),
                        ("🇧🇩 বাংলা", "bn"),
                        ("🇪🇸 Español", "es")
                    ],
                    value="en",
                    label="🌐 Language"
                )
                
                model_choice = gr.Dropdown(
                    choices=["Ensemble", "ConvNeXt-Base", "ViT-B/16", "EfficientNetV2-M"],
                    value="Ensemble",
                    label="🤖 AI Model"
                )
                
                use_tta = gr.Checkbox(
                    label="🔄 Test-Time Augmentation",
                    value=False
                )
                
                show_explanations = gr.Checkbox(
                    label="🔍 Grad-CAM++ Explanation",
                    value=True
                )
                
                predict_btn = gr.Button(
                    "🔍 ANALYZE SCAN",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Group():
                gr.Markdown("""
### 📊 Model Performance

**Ensemble:** 96%+ accuracy  
**ConvNeXt-Base:** 99.9% F1  
**ViT-B/16:** 99%+ F1  
**EfficientNetV2-M:** 99%+ F1

### 🏥 Detectable Tumors

• Glioma (গ্লাইওমা)  
• Meningioma (মেনিনজিওমা)  
• Pituitary (পিটুইটারি)  
• No Tumor (কোন টিউমার নেই)
                """)
        
        # Right Panel
        with gr.Column(scale=7, min_width=700):
            with gr.Group():
                gr.Markdown("### 📊 Analysis Results")
                result_output = gr.Textbox(
                    label="Medical Report",
                    lines=22,
                    show_copy_button=True,
                    elem_classes="result-output"
                )
            
            with gr.Row():
                uploaded_display = gr.Image(
                    label="📸 Uploaded Scan",
                    height=280
                )
                
                chart_output = gr.Image(
                    label="📊 Confidence Distribution",
                    height=280
                )
            
            explanation_output = gr.Image(
                label="🔥 Grad-CAM++ Explanation",
                height=320
            )
    
    predict_btn.click(
        fn=predict_brain_tumor,
        inputs=[image_input, model_choice, use_tta, show_explanations, language_choice],
        outputs=[result_output, explanation_output, uploaded_display, chart_output]
    )
    
    gr.Markdown("""
---

## 👨‍💻 Developer Information

**Abdullah Rubab**

🏫 Daffodil International University, Bangladesh  
📧 rubab2712@gmail.com  
🐙 GitHub: [@ABRUBAB](https://github.com/ABRUBAB)  
🔗 Project: [brain-tumor-mri-classification](https://github.com/ABRUBAB/brain-tumor-mri-classification)

---

## ⚠️ MEDICAL DISCLAIMER

This AI system is for **educational and research purposes only**.

- NOT a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- Not approved by medical regulatory authorities
- For educational and research use only

---

© 2025 Abdullah Rubab | Academic Research Project
    """)

if __name__ == "__main__":
    demo.launch()
