"""
Model implementations: Baseline LSTM and Enhanced Seq2Seq.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, encoder_outputs):
        # encoder_outputs: [batch, seq_len, hidden]
        attn_logits = self.attention(encoder_outputs)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_logits, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)  # [batch, hidden]
        return context, attn_weights

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, encoder_hidden=267, decoder_hidden=267,
                 encoder_layers=1, decoder_layers=1, encoder_dropout=0,
                 decoder_dropout=0, output_size=1):
        super().__init__()
        
        self.encoder = nn.GRU(
            input_size=input_size, 
            hidden_size=encoder_hidden,
            num_layers=encoder_layers, 
            batch_first=True,
            dropout=encoder_dropout if encoder_layers > 1 else 0
        )

        self.attention = AttentionLayer(encoder_hidden)
        
        self.decoder_cell = nn.GRUCell(
            input_size=output_size + encoder_hidden,  
            hidden_size=decoder_hidden
        )
        
        self.fc_out = nn.Linear(decoder_hidden, output_size)
        self.decoder_output_dropout = nn.Dropout(decoder_dropout)
        self.encoder_hidden = encoder_hidden
        self.output_size = output_size
    
    def forward(self, x_features, y_features):
        batch_size = x_features.shape[0]
        forecast_len = y_features.shape[1]
        
        # 1. Encoding
        enc_outputs, hidden = self.encoder(x_features)
        
        # 2. Attention
        context, _ = self.attention(enc_outputs)  # context: [batch, encoder_hidden]
        
        # 3. Decoding
        dec_hidden = hidden[-1, :, :]  # [batch_size, hidden_size]
        
        predictions = torch.zeros(batch_size, forecast_len, device=x_features.device)
        
        last_value = torch.zeros(batch_size, 1, device=x_features.device)
        
        # Decoding từng bước
        for t in range(forecast_len):
            dec_input = torch.cat([
                last_value,  # [batch, 1]
                context  # [batch, encoder_hidden]
            ], dim=1)  # [batch, 1 + encoder_hidden]
            
            # GRUCell forward
            dec_hidden = self.decoder_cell(dec_input, dec_hidden)
         
            pred_t = self.fc_out(self.decoder_output_dropout(dec_hidden))  # [batch, 1]
            predictions[:, t] = pred_t.squeeze(-1)
            
            last_value = pred_t 
        
        return predictions
    
    def predict(self, x_features, y_features):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x_features, y_features)
        return predictions

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)

        last_out = out[:, -1, :] 
        prediction = self.fc(last_out)
        return prediction
    
    def predict(self, series: np.ndarray):
        self.eval()
        input_tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            output = self.forward(input_tensor)
        pred = output.squeeze().cpu().numpy()
        return pred
