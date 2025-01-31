import sentencepiece as spm
import torch

## Tokenizer
class Callable_tokenizer():
    def __init__(self, tokenizer_path):
        self.path = tokenizer_path
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
    def __call__(self, text):
        return self.tokenizer.Encode(text)

    def get_tokenId(self, token_name):
        return self.tokenizer.piece_to_id(token_name)

    def get_tokenName(self, id):
        return self.tokenizer.id_to_piece(id)

    def decode(self, tokens_list):
        return self.tokenizer.Decode(tokens_list)

    def __len__(self):
        return len(self.tokenizer)

    def user_tokenization(self, text):
        return self(text) + [self.get_tokenId('</s>')]


@torch.no_grad
def greedy_decode(model:torch.nn.Module, source_tensor:torch.Tensor, sos_tokenId: int, eos_tokenId:int, pad_tokenId, max_tries=50):
    model.eval()
    device = source_tensor.device
    target_tensor = torch.tensor([sos_tokenId]).unsqueeze(0).to(device)

    for i in range(max_tries):
        logits, _ = model(source_tensor, target_tensor, pad_tokenId)
        # Greedy decoding
        top1 = logits[:,-1,:].argmax(dim=-1, keepdim=True)
        # Append predicted token
        target_tensor = torch.cat([target_tensor, top1], dim=1)
        # Stop if predict <EOS>
        if top1.item() == eos_tokenId:
            break
    return target_tensor.squeeze(0).tolist()


def en_translate_ar(text, model, tokenizer):
    source_tensor = torch.tensor(tokenizer(text)).unsqueeze(0)
    target_tokens = greedy_decode(model, source_tensor,
                                  tokenizer.get_tokenId('<s>'),
                                  tokenizer.get_tokenId('</s>'),
                                  tokenizer.get_tokenId('<pad>'), 30)
    
    return tokenizer.decode(target_tokens)