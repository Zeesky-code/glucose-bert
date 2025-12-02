import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from glucose_bert import GlucoseBERT, CONFIG
from glucose_bert_finetune import GlucoseForecaster, ForecastDataset, FT_CONFIG

EVAL_CONFIG = {
    'device': 'cpu',
    'model_path': 'glucose_forecaster.pth',
    'output_dir': '../analysis_results'
}

if not os.path.exists(EVAL_CONFIG['output_dir']):
    os.makedirs(EVAL_CONFIG['output_dir'])

def plot_forecasts(model, dataloader, device, num_plots=5):
    model.eval()
    inputs_list = []
    targets_list = []
    preds_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)
            
            inputs_list.append(inputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            preds_list.append(outputs.cpu().numpy())
            
            if len(inputs_list) * inputs.shape[0] >= num_plots:
                break
                
    inputs = np.concatenate(inputs_list)[:num_plots]
    targets = np.concatenate(targets_list)[:num_plots]
    preds = np.concatenate(preds_list)[:num_plots]
    
    for i in range(num_plots):
        plt.figure(figsize=(10, 6))
        
        seq_len = inputs.shape[1]
        x_input = np.arange(seq_len)
        plt.plot(x_input, inputs[i].flatten(), label='History (24h)', color='blue')
        
        x_target = seq_len + 1 
        
        plt.scatter(x_target, targets[i], color='green', label='Actual (t+30m)', s=100, marker='o')
        plt.scatter(x_target, preds[i], color='red', label='Predicted (t+30m)', s=100, marker='x')
        
        plt.title(f'Glucose Forecast Example {i+1}')
        plt.xlabel('Time Steps (15 min intervals)')
        plt.ylabel('Normalized Glucose')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(EVAL_CONFIG['output_dir'], f'forecast_example_{i+1}.png'))
        plt.close()

def main():
    print(f"Initializing Evaluator on {EVAL_CONFIG['device']}...")
    
    base_model = GlucoseBERT(CONFIG['d_model'], CONFIG['nhead'], CONFIG['num_layers'])
    model = GlucoseForecaster(base_model)
    model.load_state_dict(torch.load(EVAL_CONFIG['model_path'], map_location=EVAL_CONFIG['device']))
    model.to(EVAL_CONFIG['device'])
    
    dataset = ForecastDataset(FT_CONFIG['data_dir'], FT_CONFIG['seq_len'], FT_CONFIG['pred_len'])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    print("Generating Forecast Plots...")
    plot_forecasts(model, test_loader, EVAL_CONFIG['device'], num_plots=5)
    
    print(f"Evaluation complete. Results saved to {EVAL_CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
