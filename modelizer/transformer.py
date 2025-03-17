import torch
from math import log as math_log, sqrt as math_sqrt


# based on https://pytorch.org/tutorials/beginner/translation_transformer.html
class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_size: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.size = max_size
        den = torch.exp(- torch.arange(0, emb_size, 2) * math_log(10000) / emb_size)
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        pos_embedding = torch.zeros((max_size, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.sqrt_emb_size = math_sqrt(emb_size)
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * self.sqrt_emb_size


class Transformer(torch.nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 embedding_size: int,
                 feedforward_size: int,
                 head_count: int,
                 dropout: float, ** kwargs):
        super(Transformer, self).__init__()
        pos_encoding_size = kwargs.setdefault("pos_encoding_size", 5000)
        dropout = 0.0 if num_encoder_layers + num_decoder_layers == 2 else dropout
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.src_token_embedding = TokenEmbedding(source_vocab_size, embedding_size)
        self.trg_token_embedding = TokenEmbedding(target_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout, max_size=pos_encoding_size)
        self.transformer = torch.nn.Transformer(d_model=embedding_size,
                                                nhead=head_count,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=feedforward_size,
                                                dropout=dropout,
                                                batch_first=False)
        self.generator = torch.nn.Linear(embedding_size, target_vocab_size)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                trg_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                trg_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        src_embedding = self.positional_encoding(self.src_token_embedding(src))
        trg_embedding = self.positional_encoding(self.trg_token_embedding(trg))
        outs = self.transformer(src_embedding, trg_embedding, src_mask, trg_mask, None,
                                src_padding_mask, trg_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, device=torch.device('cpu')):
        with torch.no_grad():
            src_embedding = self.positional_encoding(self.src_token_embedding(src)).to(device)
            encoder_output = self.transformer.encoder(src_embedding, src_mask)
        return encoder_output

    def decode(self, trg: torch.Tensor, encoder_outputs: torch.Tensor, device=torch.device('cpu')):
        trg_mask = (torch.triu(torch.ones((trg.shape[0], trg.shape[0]), device=device)) == 1).transpose(0, 1)
        trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
        with torch.no_grad():
            decoder_output = self.transformer.decoder(self.positional_encoding(self.trg_token_embedding(trg)), encoder_outputs, trg_mask)
        return decoder_output

    def greedy_decode(self, trg: torch.Tensor, encoder_outputs: torch.Tensor, device=torch.device('cpu')):
        decoder_output = self.decode(trg, encoder_outputs, device).transpose(0, 1)
        return torch.max(self.generator(decoder_output[:, -1]), dim=1)[1].item()

    def beam_decode(self, trg: torch.Tensor, encoder_outputs: torch.Tensor, device=torch.device('cpu'),
                    beam_size: int = 2, ):
        decoder_output = self.decode(trg, encoder_outputs, device).transpose(0, 1)
        decoder_output = self.generator(decoder_output[:, -1])
        return torch.topk(decoder_output, beam_size, dim=1)  # (probabilities, indices)

    def get_pos_encoding_size(self):
        return self.positional_encoding.size
