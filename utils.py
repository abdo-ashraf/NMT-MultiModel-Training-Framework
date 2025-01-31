import math
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from Tokenizers.Tokenizers import Callable_tokenizer
import matplotlib.pyplot as plt
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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


# def plot_history(history, save_plots_dir, model_type):

#     fig = plt.figure(figsize=(8, 5))
#     plt.plot(history['steps'], history['train_loss'], label="Training Loss")
#     plt.plot(history['steps'], history['valid_loss'], label="Validation Loss")
#     plt.plot(history['steps'], history['valid_accuracy'], label="Validation Accuracy")
#     plt.plot(history['steps'], history['valid_bleu'], label="Validation Bleu")
#     plt.xlabel("Step")
#     plt.ylabel("Scores")
#     plt.legend()  # Add a legend to differentiate the lines
#     plot_path = os.path.join(save_plots_dir, f'{model_type}_history.png')
#     print(plot_path)
#     plt.savefig(plot_path, dpi=300)
#     # Close the plot to prevent it from displaying
#     plt.close(fig)
def plot_history(history, test_history, save_plots_dir, model_type):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Get the last step for test performance alignment
    last_step = history['steps'][-1]

    # Plot Losses
    axes[0].plot(history['steps'], history['train_loss'], label="Training Loss")
    axes[0].plot(history['steps'], history['valid_loss'], label="Validation Loss")
    if test_history:
        axes[0].scatter(last_step, test_history['test_loss'], color='red', marker='o', label="Test Loss", edgecolors='black', s=100)
        axes[0].axhline(y=test_history['test_loss'], color='red', linestyle='--')  # Add test loss as a horizontal line
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training, Validation, and Test Loss") if test_history else axes[0].set_title("Training and Validation Loss")

    # Plot Accuracy and BLEU
    axes[1].plot(history['steps'], history['valid_accuracy'], label="Validation Accuracy")
    axes[1].plot(history['steps'], history['valid_bleu'], label="Validation BLEU")
    if test_history:
        axes[1].scatter(last_step, test_history['test_accuracy'], color='green', marker='o', label="Test Accuracy", edgecolors='black', s=100)
        axes[1].scatter(last_step, test_history['test_bleu'], color='blue', marker='o', label="Test BLEU", edgecolors='black', s=100)
        axes[1].axhline(y=test_history['test_accuracy'], color='green', linestyle='--')  # Add test accuracy as a horizontal line
        axes[1].axhline(y=test_history['test_bleu'], color='green', linestyle='--')  # Add test BLEU as a horizontal line
    
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Scores")
    axes[1].legend()
    axes[1].set_title("Validation and Test Accuracy/BLEU") if test_history else axes[1].set_title("Validation Accuracy/BLEU")

    plt.tight_layout()  # Adjust layout to prevent overlap

    plot_path = os.path.join(save_plots_dir, f'{model_type}_history.png')
    plt.savefig(plot_path, dpi=300)
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


def compute_metrics(references:torch.Tensor, candidates:torch.Tensor, ignore_index:int):
    batch_size = candidates.size(0)
    total_bleu = 0
    total_accu = 0
    smoothing = SmoothingFunction().method2  # Use smoothing to handle zero n-gram overlaps
    for i in range(batch_size):
        mask_i = references[i]!=ignore_index
        candidate = candidates[i][mask_i].tolist()
        references_one = [references[i][mask_i].tolist()]
        bleu_score = sentence_bleu(references_one, candidate, weights=[0.25,0.25,0.25,0.25], smoothing_function=smoothing)
        accu_score = sentence_bleu(references_one, candidate, weights=[1.0,0.0,0.0,0.0], smoothing_function=smoothing)
        total_bleu += bleu_score
        total_accu += accu_score
    bleu = total_bleu / batch_size
    accuracy = total_accu / batch_size
    # accuracy = accuracy_score(references.cpu().reshape(-1), candidates.cpu().reshape(-1))
    
    return  {"accuracy": accuracy, "bleu": bleu}


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