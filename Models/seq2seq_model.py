import torch
from torch import nn
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_hidden, dim_feedforward, num_layers, dropout_probability=0.1):
        super().__init__()

        self.embd_layer = nn.Embedding(vocab_size, dim_embed)
        self.dropout = nn.Dropout(dropout_probability)
        self.rnn = nn.GRU(dim_embed, dim_hidden, num_layers=num_layers,
                            dropout=dropout_probability,batch_first=True,
                              bidirectional=True)
        self.ff = nn.Sequential(nn.Linear(dim_hidden*2, dim_feedforward),
                                nn.ReLU(),
                                nn.Linear(dim_feedforward, dim_hidden),
                                nn.Dropout(dropout_probability))

    def forward(self, x):
        embds = self.dropout(self.embd_layer(x))
        output, hidden = self.rnn(embds)
        ## hidden[-2,:,:]: hidden state for the forward direction of the last layer.
        ## hidden[-1,:,:]: hidden state for the backward direction of the last layer.
        last_hidden = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=-1)
        projected_hidden = self.ff(last_hidden)
        return projected_hidden



class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_hidden, num_layers, dropout_probability=0.1):
        super().__init__()

        self.embd_layer = nn.Embedding(vocab_size, dim_embed)
        self.dropout = nn.Dropout(dropout_probability)
        self.rnn = nn.GRU(dim_embed, dim_hidden, num_layers=num_layers,
                            dropout=dropout_probability, batch_first=True)
        self.ffw = nn.Linear(dim_hidden, dim_hidden)
        
    def forward(self, x, hidden_t_1):
        embds = self.dropout(self.embd_layer(x))
        output, hidden_t = self.rnn(embds, hidden_t_1)
        out = self.ffw(hidden_t[-1])
        return out, hidden_t


class Seq2seq_no_attention(nn.Module):
    def __init__(self, vocab_size:int, dim_embed:int, dim_model:int, dim_feedforward:int, num_layers:int, dropout_probability:float):
        super(Seq2seq_no_attention, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.encoder = Encoder(vocab_size, dim_embed, dim_model, dim_feedforward, num_layers, dropout_probability)
        self.decoder = Decoder(vocab_size, dim_embed, dim_model, num_layers, dropout_probability)
        self.classifier = nn.Linear(dim_model, vocab_size)
        ## weight sharing between classifier and embed_shared_src_trg_cls
        self.encoder.embd_layer.weight = self.classifier.weight
        self.decoder.embd_layer.weight = self.classifier.weight

    def forward(self, source, target, pad_tokenId):
        # target = <s> text </s>
        teacher_force_ratio = 0.5
        B, T = target.size()
        total_logits = torch.zeros(B, T, self.vocab_size, device=source.device) # (B,T,vocab_size)

        context = self.encoder(source) # (B, dim_model)
        ## We will pass the hiddens for each layer of the decoder (inspired by Attention is all you need paper)
        context = context.unsqueeze(0).repeat(self.num_layers,1,1) # (numlayer, B, dim_model)
        step_token = target[:, [0]]
        for step in range(1, T):
            out, context = self.decoder(step_token, context)
            logits = self.classifier(out)
            total_logits[:, step] = logits
            top1 = logits.argmax(-1, keepdim=True)
            step_token = target[:, [step]] if teacher_force_ratio > random.random() else top1

        flat_logits = total_logits[:,1:,:].reshape(-1, total_logits.size(-1))
        flat_targets = target[:,1:].reshape(-1)
        loss = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=pad_tokenId) if target is not None else None
        return total_logits, loss
    
    @torch.no_grad
    def translate(self, source:torch.Tensor, sos_tokenId: int, eos_tokenId:int, pad_tokenId, max_tries=50):
        targets_hat = [sos_tokenId]
        context = self.encoder(source.unsqueeze(0))
        context = context.unsqueeze(0).repeat(self.num_layers,1,1)
        for step in range(1, max_tries):
            x = torch.tensor([targets_hat[step]]).unsqueeze(0).to(source.device)
            out, context = self.decoder(x, context)
            logits = self.classifier(out)
            top1 = logits.argmax(-1)
            targets_hat.append(top1.item())
            if top1 == eos_tokenId:
                return targets_hat
        return targets_hat