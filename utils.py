import math
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from Tokenizers.Tokenizers import Callable_tokenizer
import matplotlib.pyplot as plt
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score


## Dataset
class MT_Dataset(Dataset):
    """
    Dataset class for machine translation tasks.
    
    Args:
        src_sentences_list (List[str]): List of source sentences.
        trg_sentences_list (List[str]): List of target sentences.
        callable_tokenizer (Callable): Tokenizer.
        reversed_input (bool): Whether to reverse the input sequence. Default is False.
    """
    def __init__(self, input_sentences_list:list, target_sentences_list:list, callable_tokenizer:Callable_tokenizer, reversed_input=False):
        super(MT_Dataset, self).__init__()

        assert len(input_sentences_list) == len(target_sentences_list), (f"Lengths mismatched: input has {len(input_sentences_list)} sentences, "f"but targets has {len(target_sentences_list)} sentences.")
        
        self.input_sentences_list = input_sentences_list
        self.target_sentences_list = target_sentences_list
        self.callable_tokenizer = callable_tokenizer
        self.reversed_input = reversed_input
        # self.maxlen = maxlen
        self.sos = self.callable_tokenizer.get_tokenId('<s>')
        self.eos = self.callable_tokenizer.get_tokenId('</s>')

    def __len__(self):
        return len(self.input_sentences_list)

    def __getitem__(self, index):
        input, target = self.input_sentences_list[index], self.target_sentences_list[index]

        input_tokens = torch.tensor(self.callable_tokenizer(input))
        target_tokens_forward = torch.tensor([self.sos] + self.callable_tokenizer(target) + [self.eos])
        # target_tokens_loss = torch.tensor(self.callable_tokenizer(target) + [self.eos])

        if self.reversed_input: input_tensor_tokens = input_tensor_tokens.flip(0)

        return input_tokens, target_tokens_forward#, target_tokens_loss
 
    
## Collator
class MyCollate():
    def __init__(self, pad_value, batch_first=True):
        self.pad_value = pad_value
        self.batch_first = batch_first

    def __call__(self, data):
        src_stentences = [ex[0] for ex in data]
        trg_stentences_forward = [ex[1] for ex in data]
        # trg_stentences_loss = [ex[2] for ex in data]

        padded_src_stentences = pad_sequence(src_stentences, batch_first=self.batch_first,
                                                      padding_value=self.pad_value)
        padded_trg_stentences_forward = pad_sequence(trg_stentences_forward, batch_first=self.batch_first,
                                                      padding_value=self.pad_value)
        # padded_trg_stentences_loss = pad_sequence(trg_stentences_loss, batch_first=self.batch_first,
        #                                               padding_value=self.pad_value)
        return padded_src_stentences, padded_trg_stentences_forward#, padded_trg_stentences_loss
    

def get_parameters_info(model):
    names = []
    trainable = []
    nontrainable = []

    for name, module, in model.named_children():
        names.append(name)
        trainable.append(sum(p.numel() for p in module.parameters() if p.requires_grad==True))
        nontrainable.append(sum(p.numel() for p in module.parameters() if p.requires_grad==False))

    names.append("TotalParams")
    trainable.append(sum(trainable))
    nontrainable.append(sum(nontrainable))
    return names, trainable, nontrainable


def plot_loss(train_class_losses, val_class_losses, steps, plots_dir, model_name):

    fig = plt.figure(figsize=(8, 3))
    plt.plot(steps, train_class_losses, label="Training Loss")
    plt.plot(steps, val_class_losses, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()  # Add a legend to differentiate the lines
    plot_path = os.path.join(plots_dir, f'{model_name}_losses.png')
    plt.savefig(plot_path, dpi=300)
    # Close the plot to prevent it from displaying
    plt.close(fig)


def save_checkpoint(model:torch.nn.Module, optimizer, save_dir:str, run_name:str, step:int, in_onnx=False):
    model_path = os.path.join(save_dir, f"{run_name}_step_{step}.pth")
    if in_onnx:
        ## onnx
        print("Saving in Onnx format not supported for now.")
    else:
        ## pytorch
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
    print(f"Checkpoint saved at: {model_path}")


def compute_metrics(references:torch.Tensor, candidates:torch.Tensor):
    batch_size = candidates.size(0)
    total_bleu = 0
    smoothing = SmoothingFunction().method2  # Use smoothing to handle zero n-gram overlaps
    for i in range(batch_size):
        mask_i = references[i]!=0
        candidate = candidates[i][mask_i].tolist()
        references_one = [references[i][mask_i].tolist()]
        bleu_score = sentence_bleu(references_one, candidate, weights=[0.33,0.33,0.33,0.0], smoothing_function=smoothing)
        # print(round(bleu_score, 4))
        total_bleu += bleu_score
    bleu = total_bleu / batch_size
    accuracy = accuracy_score(references.reshape(-1), candidates.reshape(-1))
    
    return  {"Accuracy": accuracy, "Bleu": bleu}


class CosineScheduler():
    def __init__(self, max_steps:int, warmup_steps:int, max_lr:float, min_lr:float):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step):
        if step < self.warmup_steps:
            ## linear warmup
            return self.max_lr*(step+1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        else:
            decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
            return self.min_lr + coeff * (self.max_lr - self.min_lr)