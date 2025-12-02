import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import os
import math
from glucose_bert import GlucoseBERT, CONFIG


FT_CONFIG = {
    'data_dir': '../Shanghai_T1DM',
    'seq_len': 96,
    'pred_len': 2,
    'batch_size': 32,
    'lr': 5e-5,
    'epochs': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pretrained_path': 'glucose_bert_pretrained.pth'
}

class ForecastDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = self._load_data(data_dir)

    def _load_data(self, data_dir):
        files = glob.glob(os.path.join(data_dir, '*.xls*'))
        all_sequences = []
        
        for f in files:
            df = pd.read_excel(f)
            if 'CGM (mg / dl)' not in df.columns:
                continue
                
            glucose = pd.to_numeric(df['CGM (mg / dl)'], errors='coerce').dropna().values
            
            if len(glucose) > 0:
                glucose = (glucose - np.mean(glucose)) / (np.std(glucose) + 1e-5)
                
                for i in range(len(glucose) - self.seq_len - self.pred_len):
                    seq = glucose[i:i+self.seq_len]
                    target = glucose[i+self.seq_len+self.pred_len-1] # Point prediction
                    all_sequences.append((seq, target))
                    
        return all_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return {
            'input': torch.tensor(seq, dtype=torch.float32).unsqueeze(-1),
            'target': torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        }

class GlucoseForecaster(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model
        self.forecast_head = nn.Linear(CONFIG['d_model'], 1)

    def forward(self, x):
        src = x.permute(1, 0, 2)
        embedded = self.encoder.input_proj(src)
        embedded = self.encoder.pos_encoder(embedded)
        features = self.encoder.transformer_encoder(embedded)
        
        last_token_feat = features[-1, :, :]
        
        return self.forecast_head(last_token_feat)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    print(f"Initializing Forecaster on {FT_CONFIG['device']}...")
    
    base_model = GlucoseBERT(CONFIG['d_model'], CONFIG['nhead'], CONFIG['num_layers'])
    base_model.load_state_dict(torch.load(FT_CONFIG['pretrained_path'], map_location=FT_CONFIG['device']))
    
    model = GlucoseForecaster(base_model).to(FT_CONFIG['device'])
    
    dataset = ForecastDataset(FT_CONFIG['data_dir'], FT_CONFIG['seq_len'], FT_CONFIG['pred_len'])

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=FT_CONFIG['lr'])
    criterion = nn.MSELoss()
    
    print("Starting Fine-tuning...")
    for epoch in range(FT_CONFIG['epochs']):
        loss = train(model, train_loader, optimizer, criterion, FT_CONFIG['device'])
        print(f"Epoch {epoch+1}/{FT_CONFIG['epochs']} | Loss: {loss:.6f}")
        
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(FT_CONFIG['device'])
            targets = batch['target'].to(FT_CONFIG['device'])
            outputs = model(inputs)
            total_mse += criterion(outputs, targets).item()
            
    rmse = math.sqrt(total_mse / len(test_loader))
    print(f"Test RMSE: {rmse:.4f}")
    
    torch.save(model.state_dict(), 'glucose_forecaster.pth')

if __name__ == "__main__":
    main()