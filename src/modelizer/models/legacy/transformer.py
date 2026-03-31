import torch

from typing import Union, Tuple, List
from math import log as math_log, sqrt as math_sqrt


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_size: int = 5000):
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError(f"emb_size must be even, got {emb_size} instead.")
        if not isinstance(dropout, float) or dropout < 0 or dropout >= 1:
            raise TypeError(f"dropout must be a non-negative float, got {dropout} instead.")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got {max_size} instead.")

        den = torch.exp(- torch.arange(0, emb_size, 2) * math_log(10000) / emb_size)
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        pos_embedding = torch.zeros((max_size, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        seq_len = token_embedding.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum position encoding length {self.pos_embedding.size(1)}")
        return self.dropout(token_embedding + self.pos_embedding[:, :seq_len, :])


class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size} instead.")
        if not isinstance(emb_size, int) or emb_size <= 0:
            raise ValueError(f"emb_size must be a positive integer, got {emb_size} instead.")

        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.register_buffer("sqrt_emb_size", torch.tensor(math_sqrt(float(emb_size)), dtype=torch.float32))

    def forward(self, tokens: torch.Tensor):
        if torch.any(tokens >= self.embedding.num_embeddings) or torch.any(tokens < 0):
            raise ValueError(f"Token indices must be in range [0, {self.embedding.num_embeddings}). Got tokens: {tokens}")
        return self.embedding(tokens.long()) * self.sqrt_emb_size


class Transformer(torch.nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 enc_layers: int,
                 dec_layers: int,
                 embedding_size: int,
                 feedforward_size: int,
                 head_count: int,
                 dropout: float,
                 source_max_len: int = 5000,
                 target_max_len: int = 5000):
        super().__init__()

        if not isinstance(source_vocab_size, int) or source_vocab_size <= 0:
            raise ValueError(f"source_vocab_size must be a positive integer, got {source_vocab_size} instead.")
        if not isinstance(target_vocab_size, int) or target_vocab_size <= 0:
            raise ValueError(f"target_vocab_size must be a positive integer, got {target_vocab_size} instead.")
        if not isinstance(enc_layers, int) or enc_layers < 1:
            raise ValueError(f"enc_layers must be a positive integer, got {enc_layers} instead.")
        if not isinstance(dec_layers, int) or dec_layers < 1:
            raise ValueError(f"dec_layers must be a positive integer, got {dec_layers} instead.")
        if not isinstance(embedding_size, int) or embedding_size <= 0:
            raise ValueError(f"embedding_size must be a positive integer, got {embedding_size} instead.")
        if not isinstance(feedforward_size, int) or feedforward_size <= 0:
            raise ValueError(f"feedforward_size must be a positive integer, got {feedforward_size} instead.")
        if not isinstance(head_count, int) or head_count < 1:
            raise ValueError(f"head_count must be a positive integer, got {head_count} instead.")
        if not isinstance(dropout, float) or dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be a float in [0, 1), got {dropout} instead.")
        if not isinstance(source_max_len, int) or source_max_len <= 0:
            raise ValueError(f"source_max_len must be a positive integer, got {source_max_len} instead.")
        if not isinstance(target_max_len, int) or target_max_len <= 0:
            raise ValueError(f"target_max_len must be a positive integer, got {target_max_len} instead.")
        if embedding_size % head_count != 0:
            raise ValueError(f"embedding_size ({embedding_size}) must be divisible by head_count ({head_count})")

        dropout = 0.0 if enc_layers + dec_layers == 2 else dropout
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.src_token_embedding = TokenEmbedding(source_vocab_size, embedding_size)
        self.tgt_token_embedding = TokenEmbedding(target_vocab_size, embedding_size)
        self.src_positional_encoding = PositionalEncoding(embedding_size, dropout=dropout, max_size=source_max_len)
        self.tgt_positional_encoding = PositionalEncoding(embedding_size, dropout=dropout, max_size=target_max_len)
        self.transformer = torch.nn.Transformer(d_model=embedding_size,
                                                nhead=head_count,
                                                num_encoder_layers=enc_layers,
                                                num_decoder_layers=dec_layers,
                                                dim_feedforward=feedforward_size,
                                                dropout=dropout,
                                                batch_first=True)
        self.generator = torch.nn.Linear(embedding_size, target_vocab_size)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        src_embedding = self.src_positional_encoding(self.src_token_embedding(src))
        tgt_embedding = self.tgt_positional_encoding(self.tgt_token_embedding(tgt))

        # Normalizing masks to the same type (float additive) to avoid PyTorch deprecation warning.
        float_dtype = src_embedding.dtype
        if src_mask is not None and src_mask.dtype == torch.bool:
            src_mask = src_mask.float().masked_fill(src_mask, float('-inf'))
        if tgt_mask is not None and tgt_mask.dtype == torch.bool:
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))
        if src_padding_mask is not None and src_padding_mask.dtype == torch.bool:
            src_padding_mask = torch.zeros_like(src_padding_mask, dtype=float_dtype).masked_fill(src_padding_mask, float('-inf'))
        if tgt_padding_mask is not None and tgt_padding_mask.dtype == torch.bool:
            tgt_padding_mask = torch.zeros_like(tgt_padding_mask, dtype=float_dtype).masked_fill(tgt_padding_mask, float('-inf'))
        if memory_key_padding_mask is not None and memory_key_padding_mask.dtype == torch.bool:
            memory_key_padding_mask = torch.zeros_like(memory_key_padding_mask, dtype=float_dtype).masked_fill(memory_key_padding_mask, float('-inf'))

        outs = self.transformer(src_embedding, tgt_embedding, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
        self.to(device)
        src_embedding = self.src_positional_encoding(self.src_token_embedding(src.to(device)))
        src_mask = src_mask.to(device) if src_mask is not None else None

        with torch.no_grad():
            encoder_output = self.transformer.encoder(src_embedding, src_mask)
        return encoder_output

    def decode(self, tgt: torch.Tensor, encoder_outputs: torch.Tensor, tgt_mask: torch.Tensor = None, device=torch.device('cpu')) -> torch.Tensor:
        self.to(device)
        seq_len = tgt.shape[1]
        if tgt_mask is None:
            tgt_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_embedding = self.tgt_positional_encoding(self.tgt_token_embedding(tgt.to(device)))
        tgt_mask = tgt_mask.to(device) if tgt_mask is not None else None
        encoder_outputs = encoder_outputs.to(device)
        with torch.no_grad():
            decoder_output = self.transformer.decoder(tgt_embedding, encoder_outputs, tgt_mask)
        return decoder_output

    def greedy_decode(self, tgt: torch.Tensor, encoder_outputs: torch.Tensor, device=torch.device('cpu'), **_) -> Union[int, torch.Tensor]:
        decoder_output = self.decode(tgt, encoder_outputs, device=device).transpose(0, 1)
        last_step = decoder_output[-1]
        logits = self.generator(last_step)
        idx = torch.max(logits, dim=1)[1]
        return idx.item() if idx.shape[0] == 1 else idx

    def beam_decode(self, tgt: torch.Tensor, encoder_outputs: torch.Tensor, device=torch.device('cpu'), beam_size: int = 2) -> Union[Tuple[List[float], List[int]], Tuple[torch.Tensor, torch.Tensor]]:
        decoder_output = self.decode(tgt, encoder_outputs, device=device).transpose(0, 1)
        last_step = decoder_output[-1]
        logits = self.generator(last_step)
        values, indices = torch.topk(logits, beam_size, dim=1)
        return (values[0].tolist(), indices[0].tolist()) if logits.size(0) == 1 else (values, indices)
