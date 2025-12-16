import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from .models.tokenizer.tokenizer import Tokenizer
from .models.blocks.encoder_layer import EncoderLayer
from .models.blocks.decoder_layer import DecoderLayer
from .models.layers.layer_norm import LayerNorm
from .models.layers.positional_encoding import PositionalEncoding

@dataclass
class MachineTranslatorConfig:
    block_size: int = 192
    vocab_size: int = 32000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True
    
class MachineTranslator(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0),
            wpe = PositionalEncoding(config),
            dropout = nn.Dropout(config.dropout),
            enc_h = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)]),
            dec_h = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        self.apply(self._init_weights)
        
        # Thực hiện khai báo special scaled cho ma trận Output của Attention, theo GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("proj_output.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Báo cáo số lượng tham số
        print(f"Số lượng tham số: {self.get_num_params() / 1e6:.2f}M")
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def generate_mask(self, src, tgt):
        """
            Tạo padding mask cho câu nguồn và câu đích
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seg_length = tgt.size(1)
        tril = torch.tril(torch.ones(seg_length, seg_length)).bool()
        tgt_mask = tgt_mask & tril
        return src_mask, tgt_mask
     
    def forward(self, src, tgt, labels=None):
        b, t = src.size()
        assert t <= self.config.block_size, f"Không thể truyền vào chuỗi có độ dài {t}, kích thước của một block là {self.config.block_size}"

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_tok_emb = self.transformer.wte(src) * math.sqrt(self.config.n_embd)
        tgt_tok_emb = self.transformer.wte(tgt) * math.sqrt(self.config.n_embd)
        src_pos_emb = self.transformer.wpe(src_tok_emb)
        tgt_pos_emb = self.transformer.wpe(tgt_tok_emb)
        
        src_x = self.transformer.dropout(src_pos_emb)        
        for block in self.transformer.enc_h:
            src_x = block(src_x, src_mask)

        tgt_x = self.transformer.dropout(tgt_pos_emb)
        for block in self.transformer.dec_h:
            tgt_x = block(tgt_x, src_x, src_mask, tgt_mask)
        tgt_x = self.transformer.ln_f(tgt_x)
        
        if labels is not None:
            logits = self.lm_head(tgt_x)
            loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1), ignore_index=0)
        else:
            logits = self.lm_head(tgt_x[:, [-1], :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def translate(self, src, max_seq_length=192):
        tokenizer = Tokenizer("en-vi")
        
        # Thêm kí tự đệm vào các chuỗi
        src_enc = tokenizer.encode(src)
        src_enc += [tokenizer.eos_id()] + [tokenizer.pad_id()] * (self.config.block_size - len(src_enc) - 1)
        src_input = torch.tensor([src_enc], dtype=torch.long)
        
        tgt_enc = [tokenizer.bos_id()] + [tokenizer.pad_id()] * (self.config.block_size - 1)
        tgt_input = torch.tensor([tgt_enc], dtype=torch.long)
        
        # Quy trình dịch
        index = 0
        for _ in range(max_seq_length):
            logits, _ = self(src_input, tgt_input)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next == tokenizer.eos_id() or index == max_seq_length - 1:
                break
            tgt_input[:, index + 1] = idx_next
            index += 1
            
        tgt = tokenizer.decode(tgt_input.squeeze(0).tolist())
        return tgt