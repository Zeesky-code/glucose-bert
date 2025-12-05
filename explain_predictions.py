import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/Users/mac2/Downloads/diabetes_datasets/glucose_bert')
from glucose_bert import GlucoseBERT, CONFIG
from glucose_bert_finetune import GlucoseForecaster, ForecastDataset, FT_CONFIG

OUTPUT_DIR = 'explainability_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_model():
    base_model = GlucoseBERT(CONFIG['d_model'], CONFIG['nhead'], CONFIG['num_layers'])
    model = GlucoseForecaster(base_model)
    model.load_state_dict(torch.load('glucose_forecaster.pth', map_location='cpu'))
    model.eval()
    return model

def prepare_data():
    dataset = ForecastDataset(FT_CONFIG['data_dir'], FT_CONFIG['seq_len'], FT_CONFIG['pred_len'])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_samples = []
    test_targets = []
    for i in range(min(100, len(test_dataset))):
        sample = test_dataset[i]
        test_samples.append(sample['input'].numpy())
        test_targets.append(sample['target'].numpy())
    
    return np.array(test_samples), np.array(test_targets)

def compute_gradient_attribution(model, input_tensor):
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    output = model(input_tensor)
    output.backward()
    
    attributions = (input_tensor.grad * input_tensor).detach().numpy()
    
    return attributions

def explain_predictions():
    print("Loading model and data...")
    model = load_model()
    X_test, y_test = prepare_data()
    
    print(f"Test set: {X_test.shape}")
    print("Computing gradient-based attributions...")
    
    test_samples = X_test[:10]
    attributions_list = []
    
    for sample in test_samples:
        input_tensor = torch.FloatTensor(sample).unsqueeze(0)
        attr = compute_gradient_attribution(model, input_tensor)
        attributions_list.append(attr[0])
    
    attributions = np.array(attributions_list)
    
    print("Generating visualizations...")
    
    for i in range(min(3, len(test_samples))):
        plt.figure(figsize=(14, 6))
        
        time_steps = np.arange(len(test_samples[i]))
        
        plt.subplot(1, 2, 1)
        plt.plot(time_steps, test_samples[i].flatten(), label='Glucose History', linewidth=2, color='steelblue')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Time Steps (15 min intervals ago)')
        plt.ylabel('Normalized Glucose')
        plt.title(f'Patient {i+1}: Input Sequence (2h history)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        attr_values = attributions[i].flatten()
        colors = ['red' if v < 0 else 'green' for v in attr_values]
        plt.bar(time_steps, attr_values, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Time Steps (15 min intervals ago)')
        plt.ylabel('Attribution (Influence on Prediction)')
        plt.title(f'Patient {i+1}: Which Moments Mattered Most?')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'explanation_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved explanation for patient {i+1}")
    
    mean_abs_attr = np.mean(np.abs(attributions), axis=0).flatten()
    
    plt.figure(figsize=(14, 6))
    time_steps = np.arange(len(mean_abs_attr))
    plt.bar(time_steps, mean_abs_attr, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Time Steps (15 min intervals ago)')
    plt.ylabel('Mean |Attribution|')
    plt.title('Overall Feature Importance: Which Parts of History Matter Most for Forecasting?')
    plt.grid(True, alpha=0.3, axis='y')
    
    top_5_indices = np.argsort(mean_abs_attr)[-5:][::-1]
    for idx in top_5_indices:
        minutes_ago = (len(mean_abs_attr) - idx) * 15
        plt.text(idx, mean_abs_attr[idx], f'{minutes_ago}m ago',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overall_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
    print("\nTop 5 Most Important Time Points:")
    for rank, idx in enumerate(top_5_indices):
        minutes_ago = (len(mean_abs_attr) - idx) * 15
        print(f"  {rank+1}. {minutes_ago} minutes ago (Attribution: {mean_abs_attr[idx]:.4f})")

if __name__ == "__main__":
    explain_predictions()
