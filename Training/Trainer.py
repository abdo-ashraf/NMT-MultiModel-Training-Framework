import os
import torch
from tqdm import tqdm
from .TrainingArguments import TrainingArguments
from utils import MT_Dataset, MYCollate, save_checkpoint, plot_loss
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, args:TrainingArguments, model:torch.nn.Module,
                 train_ds:MT_Dataset, valid_ds:MT_Dataset,
                 collator:MYCollate, compute_metrics_func):
        self.args = args
        self.model = model.to(self.args.device)
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.collator = collator
        self.compute_metrics_func = compute_metrics_func

        self.generator = torch.manual_seed(self.args.seed) if self.args.seed else None
        self.train_loader = DataLoader(self.train_ds,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  collate_fn=self.collator,
                                  num_workers=self.args.cpu_num_workers,
                                  generator=self.generator,
                                  pin_memory=self.args.pin_memory)
        
        self.valid_loader = DataLoader(self.valid_ds,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  collate_fn=self.collator,
                                  num_workers=self.args.cpu_num_workers,
                                  generator=self.generator,
                                  pin_memory=self.args.pin_memory)

    def train(self, src_pad_tokenId=None, trg_pad_tokenId=None):

        print(f"Start Training {self.model.__class__.__name__} model...")
        if src_pad_tokenId:
            print('AdamW will be used for Transformer-based models')
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            print('Adam will be used for non-Transformer models')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.torch_compile:
            print(f"Compiling the model using torch.compile...")
            self.model = torch.compile(self.model)
            print(f"model Compilation done.")

        if self.args.precision == 'high':
            print("Using BF16")     
        else: 
            print("Using TF16")

        train_losses = []
        valid_losses = []
        valid_metric = []
        steps = []
        step=0
        train_loader_iter = iter(self.train_loader)  # Create an iterator for the train_loader

        tqdm_loop = tqdm(total=self.args.max_steps, position=0)
        while step < self.args.max_steps:
            self.model.train()  # Set the model to training mode
            try:
                # Get the next batch
                data, labels_forward, labels_loss = next(train_loader_iter)
            except StopIteration:
                # Reinitialize the iterator when all batches are consumed
                train_loader_iter = iter(self.train_loader)
                data, labels_forward, labels_loss = next(train_loader_iter)
            # Get data
            data = data.to(self.args.device)
            labels_forward = labels_forward.to(self.args.device)
            labels_loss = labels_loss.to(self.args.device)
            # Forward
            if self.args.precision == 'high':
                with torch.autocast(device_type=self.args.device, dtype=torch.bfloat16):
                    logits, loss = self.model(data, labels_forward, labels_loss, src_pad_tokenId=src_pad_tokenId, trg_pad_tokenId=trg_pad_tokenId)
            else:
                logits, loss = self.model(data, labels_forward, labels_loss, src_pad_tokenId=src_pad_tokenId, trg_pad_tokenId=trg_pad_tokenId)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update step
            step += 1
            tqdm_loop.update(1)
            tqdm_loop.set_description(f"Step [{step}/{self.args.max_steps}]")
            tqdm_loop.set_postfix_str(f'loss = {round(loss.item(), 4)}')


            if step % self.args.eval_steps == 0 or step == self.args.max_steps:
                train_losses.append(loss.item())
                steps.append(step)
                val_loss = self.evaluate(compute_metric_fn=self.compute_metrics_func)
                valid_losses.append(val_loss)
                print(f'Validation: Loss {val_loss:.4f}, Bleu Score (future work)%')
                
            # Save model at specific intervals
            if step % self.args.save_steps == 0 or step == self.args.max_steps:
                save_checkpoint(model=self.model,
                                optimizer=optimizer,
                                save_dir=self.args.save_models_dir,
                                run_name=self.args.run_name,
                                step=step,
                                in_onnx=self.args.onnx)

        tqdm_loop.close()
        print("Model Training Done.")

        plot_loss(train_losses, valid_losses, steps, self.args.save_plots_dir, self.args.run_name)

        return train_losses, valid_losses
    

    @torch.no_grad()
    def evaluate(self, compute_metric_fn):
        self.model.eval()

        total_loss = 0
        
        for data, labels_forward, labels_loss in self.valid_loader:
            data = data.to(self.args.device)
            labels_forward = labels_forward.to(self.args.device)
            labels_loss = labels_loss.to(self.args.device)

            class_logits, item_total_loss = self.model(data, labels_forward, labels_loss)
            total_loss += item_total_loss.item()

        avg_loss = total_loss / len(self.valid_loader)

        return avg_loss 