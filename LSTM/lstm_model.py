import math
import torch
import torch.nn as nn
from .lstm_config import( TICKER_EMB_DIM, MARKET_EMB_DIM, REGION_EMB_DIM, INTERVAL_EMB_DIM,
                        LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,BIDIRECTIONAL, ATTN_HIDDEN, HEAD_HIDDEN, HEAD_DROPOUT, )


class  PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe_tensor = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe_tensor[:, :x.size(1), :]


class MultiHeadTemporalAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, temperature=0.5, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.scale = math.sqrt(self.head_dim) * temperature
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def  forward(self, x):
        B, T, D = x.shape
        x = self.pos_enc(x)
        Q = self.Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), neg_inf)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = self.norm(out + x)

        context = out[:, -1, :]

        attn_viz = weights[:, :, -1, :].mean(dim=1)

        return context, attn_viz

class PredictionHead(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MarketLSTM(nn.Module):

    def __init__(self, num_features: int, num_tickers: int, num_markets: int, num_regions: int, num_intervals: int):
        super().__init__()
        self.ticker_emb = nn.Embedding(num_tickers + 1, TICKER_EMB_DIM, padding_idx=0)
        self.market_emb = nn.Embedding(num_markets + 1, MARKET_EMB_DIM, padding_idx=0)
        self.region_emb = nn.Embedding(num_regions + 1, REGION_EMB_DIM, padding_idx=0)
        self.interval_emb = nn.Embedding(num_intervals + 1, INTERVAL_EMB_DIM, padding_idx=0)

        emb_total = TICKER_EMB_DIM + MARKET_EMB_DIM + REGION_EMB_DIM + INTERVAL_EMB_DIM

        self.input_proj = nn.Sequential(
            nn.Linear(num_features + emb_total, LSTM_HIDDEN),
            nn.LayerNorm(LSTM_HIDDEN),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size = LSTM_HIDDEN,
            hidden_size = LSTM_HIDDEN,
            num_layers = LSTM_LAYERS,
            dropout = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
            bidirectional = BIDIRECTIONAL,
            batch_first = True,
        )

        lstm_out_dim = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)

        self.attention = MultiHeadTemporalAttention(
            hidden_dim=lstm_out_dim,
            num_heads=8,
            temperature=0.5,
            dropout=0.1)

        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.price_head = PredictionHead(lstm_out_dim, HEAD_HIDDEN, 1, HEAD_DROPOUT)
        self.dir_head = PredictionHead(lstm_out_dim, HEAD_HIDDEN, 1, HEAD_DROPOUT)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_hh" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x_num: torch.Tensor, x_emb: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, _ = x_num.shape

        t_emb = self.ticker_emb(x_emb[:, 0])
        m_emb = self.market_emb(x_emb[:, 1])
        r_emb = self.region_emb(x_emb[:, 2])
        i_emb = self.interval_emb(x_emb[:, 3])
        emb = torch.cat([t_emb, m_emb, r_emb, i_emb], dim=-1)
        emb_expanded = emb.unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([x_num, emb_expanded], dim=-1)

        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)

        context, attn_weights = self.attention(lstm_out)
        context = self.dropout(self.norm(context))

        price_pred = self.price_head(context)
        dir_pred = self.dir_head(context)

        return price_pred, dir_pred, attn_weights
