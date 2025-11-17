#!/usr/bin/env python3
"""
Single Image Prediction using trained ViT-B/16 model
For Poultry Disease Classification
"""

import os
import sys
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import json
import numpy as np

def predict_image(image_path, model_path="./vit_poultry_results/final_model"):
    """
    Predict disease class for a single image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model directory
    
    Returns:
        dict with prediction results
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first using train_vit_b16.py")
        return None
    
    # Load model and feature extractor
    print(f"🔮 Loading model from {model_path}...")
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load and preprocess image
    print(f"📸 Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Extract features
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    print("🤖 Making prediction...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    
    # Get prediction
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, min(3, len(probabilities)))
    top3_predictions = [
        {
            'class': model.config.id2label[idx.item()],
            'confidence': prob.item()
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]
    
    # Prepare results
    results = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3_predictions': top3_predictions,
        'all_probabilities': {
            model.config.id2label[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }
    }
    
    return results


def print_results(results):
    """Pretty print prediction results"""
    if results is None:
        return
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n📸 Image: {results['image_path']}")
    print(f"\n🎯 Predicted Disease: {results['predicted_class'].upper()}")
    print(f"   Confidence: {results['confidence']*100:.2f}%")
    
    print(f"\n📊 Top 3 Predictions:")
    for i, pred in enumerate(results['top3_predictions'], 1):
        bar = "█" * int(pred['confidence'] * 50)
        print(f"   {i}. {pred['class']:<20} {pred['confidence']*100:>6.2f}% {bar}")
    
    print("\n📈 All Class Probabilities:")
    sorted_probs = sorted(results['all_probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        bar = "▓" * int(prob * 30)
        print(f"   {cls:<20} {prob*100:>6.2f}% {bar}")
    
    print("=" * 60 + "\n")


def main():
    """Main function for command line usage"""
    
    if len(sys.argv) < 2:
        print("Usage: python predict_single.py <image_path> [model_path]")
        print("\nExample:")
        print("  python predict_single.py test_image.jpg")
        print("  python predict_single.py test_image.jpg ./vit_poultry_results/final_model")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "./vit_poultry_results/final_model"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    # Make prediction
    results = predict_image(image_path, model_path)
    
    # Print results
    print_results(results)
    
    # Optionally save results to JSON
    output_json = image_path.rsplit('.', 1)[0] + '_prediction.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {output_json}")


if __name__ == '__main__':
    main()
