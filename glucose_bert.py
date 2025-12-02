import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import os
import math

CONFIG = {
    'data_dir': '../Shanghai_T1DM',
    'seq_len': 96,  
    'mask_ratio': 0.15,
    'batch_size': 32,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 3,
    'lr': 1e-4,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class GlucoseDataset(Dataset):
    def __init__(self, data_dir, seq_len, mask_ratio):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
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
                
                # Create sliding windows
                for i in range(len(glucose) - self.seq_len):
                    all_sequences.append(glucose[i:i+self.seq_len])
                    
        return np.array(all_sequences, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        
        # Masking
        mask = np.random.rand(self.seq_len) < self.mask_ratio
        masked_seq = seq.copy()
        masked_seq[mask] = 0 
        
        return {
            'masked_input': torch.tensor(masked_seq).unsqueeze(-1),
            'target': torch.tensor(seq).unsqueeze(-1),              
            'mask': torch.tensor(mask)                             
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class GlucoseBERT(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4),
            num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        prediction = self.output_head(output)
        
        return prediction.permute(1, 0, 2)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        inputs = batch['masked_input'].to(device)
        targets = batch['target'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        outputs_masked = outputs[mask]
        targets_masked = targets[mask]
        
        if len(targets_masked) > 0:
            loss = criterion(outputs_masked, targets_masked)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    print(f"Initializing GlucoseBERT on {CONFIG['device']}...")
    
    dataset = GlucoseDataset(CONFIG['data_dir'], CONFIG['seq_len'], CONFIG['mask_ratio'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    print(f"Loaded {len(dataset)} sequences.")
    
    model = GlucoseBERT(CONFIG['d_model'], CONFIG['nhead'], CONFIG['num_layers']).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    print("Starting Pre-training...")
    for epoch in range(CONFIG['epochs']):
        loss = train(model, dataloader, optimizer, criterion, CONFIG['device'])
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {loss:.6f}")
            
    torch.save(model.state_dict(), 'glucose_bert_pretrained.pth')
    print("Pre-training complete. Model saved.")

if __name__ == "__main__":
    main()
