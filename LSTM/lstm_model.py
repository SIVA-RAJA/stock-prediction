import torch
import torch.nn as nn
from lstm_config import( TICKER_EMB_DIM, MARKET_EMB_DIM, REGION_EMB_DIM, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,BIDIRECTIONAL, ATTN_HIDDEN, HEAD_HIDDEN, HEAD_DROPOUT, )

class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.v(torch.tanh(self.W(lstm_out)))
        weights = torch.softmax(score, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights.squeeze(-1)


class PredictionHead(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MarketLSTM(nn.Module):

    def __init__(self, num_features: int, num_tickers: int, num_markets: int, num_regions: int, ):
        super().__init__()
        self.ticker_emb = nn.Embedding(num_tickers + 1, TICKER_EMB_DIM, padding_idx=0)
        self.market_emb = nn.Embedding(num_markets + 1, MARKET_EMB_DIM, padding_idx=0)
        self.region_emb = nn.Embedding(num_regions + 1, REGION_EMB_DIM, padding_idx=0)

        emb_total = TICKER_EMB_DIM + MARKET_EMB_DIM + REGION_EMB_DIM

        self.input_proj = nn.Sequential(
            nn.Linear(num_features + emb_total, LSTM_HIDDEN),
            nn.LayerNorm(LSTM_HIDDEN),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size = LSTM_HIDDEN,
            hidden_siz = LSTM_HIDDEN,
            num_layers = LSTM_LAYERS,
            dropout = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
            bidirectional = BIDIRECTIONAL,
            batch_first = True,
        )

        lstm_out_dim = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)

        self.attention = AdditiveAttention(lstm_out_dim, ATTN_HIDDEN)
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.price_head = PredictionHead(lstm_out_dim, HEAD_HIDDEN, out_dim=1, dropout=HEAD_DROPOUT)
        self.dir_head = PredictionHead(lstm_out_dim, HEAD_HIDDEN, out_dim=1, dropout=HEAD_DROPOUT)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif"weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name or "bias_hh" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x_num: torch.Tensor, x_emb: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, _ = x_num.shape

        t_emb = self.ticker_emb(x_emb[:, 0])
        m_emb = self.market_emb(x_emb[:, 1])
        r_emb = self.region_emb(x_emb[:, 2])
        emb = torch.cat([t_emb, m_emb, r_emb], dim=-1)

        emb_expanded = emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_num, emb_expanded], dim=-1)

        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)

        context, attn_weights = self.attention(lstm_out)
        context = self.dropout(self.norm(context))

        price_pred = self.price_head(context)
        dir_pred = self.sigmoid(self.dir_head(context))

        return price_pred, dir_pred, attn_weights
