import torch
from tqdm import tqdm
from .TrainingArguments import TrainingArguments
from utils import MT_Dataset, MyCollate, save_checkpoint, CosineScheduler
from torch.utils.data import DataLoader
from collections import defaultdict


class Trainer():
    def __init__(self, args:TrainingArguments, model:torch.nn.Module,
                 train_ds:MT_Dataset, valid_ds:MT_Dataset,
                 collator:MyCollate, compute_metrics_func):
        
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
        
        self.lr_sch = CosineScheduler(max_steps=args.max_steps,
                                 warmup_steps=args.warmup_steps,
                                 max_lr=args.learning_rate,
                                 min_lr=args.learning_rate*args.lr_decay_ratio)

    def train(self):

        print(f"Start Training {self.model.__class__.__name__} model...")
        print(f'AdamW optimizer will be used will learning_rate={self.args.learning_rate}, weight_decay={self.args.weight_decay}')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        if self.args.torch_compile:
            print(f"Compiling the model using torch.compile...")
            self.model = torch.compile(self.model)
            print(f"model Compilation done.")

        if self.args.precision == 'high':
            print("Using BF16")     
        else: 
            print("Using TF16")

        history = defaultdict(list)
        # train_losses = []
        step=0
        best_valid_bleu = float("-inf")  # Assuming BLEU score is always non-negative
        train_loader_iter = iter(self.train_loader)  # Create an iterator for the train_loader

        tqdm_loop = tqdm(total=self.args.max_steps, position=0)
        self.model = self.model.train()  # Set the model to training mode
        while step < self.args.max_steps:
            try:
                # Get the next batch
                data, labels_forward = next(train_loader_iter)
            except StopIteration:
                # Reinitialize the iterator when all batches are consumed
                train_loader_iter = iter(self.train_loader)
                data, labels_forward = next(train_loader_iter)
            # Get data
            data = data.to(self.args.device)
            labels_forward = labels_forward.to(self.args.device)
            
            # Forward
            if self.args.precision == 'high':
                with torch.autocast(device_type=self.args.device, dtype=torch.bfloat16):
                    logits, loss = self.model(source=data,
                                              target=labels_forward,
                                              pad_tokenId=self.collator.pad_value)
            else:
                logits, loss = self.model(source=data,
                                          target=labels_forward,
                                          pad_tokenId=self.collator.pad_value)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            # train_losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            curr_lr = self.lr_sch.get_lr(step=step)
            for group in optimizer.param_groups:
                group['lr'] = curr_lr
            optimizer.step()

            # Update step
            step += 1
            tqdm_loop.update(1)
            tqdm_loop.set_description(f"Step [{step}/{self.args.max_steps}]")
            tqdm_loop.set_postfix_str(f'loss={round(loss.item(), 4)}')

            if self.args.eval_steps != 0 and self.args.eval_steps is not None:
                if step % self.args.eval_steps == 0 or step == self.args.max_steps:
                    # mean_loss = round(sum(train_losses)/len(train_losses), 4)
                    # train_losses = []
                    # history['train_loss'].append(mean_loss)
                    history['train_loss'].append(round(loss.item(), 4))
                    history['steps'].append(step)
                    metrics = self.evaluate()
                    for metric, value in metrics.items():
                        history[metric].append(value)
                    print(f'\n  Validation step-{step}: {metrics}')
                    # Check if the current BLEU score is better than or equal to the best BLEU score
                    if metrics['valid_bleu'] >= best_valid_bleu:
                        print(f"    BLEU score improved from {best_valid_bleu} to {metrics['valid_bleu']}")
                        best_valid_bleu = metrics['valid_bleu']  # Update the best BLEU score
                        # Save the model checkpoint
                        save_checkpoint(model=self.model,
                                        optimizer=optimizer,
                                        save_dir=self.args.save_models_dir,
                                        run_name=self.args.run_name,
                                        in_onnx=self.args.onnx)
                    self.model = self.model.train()

        tqdm_loop.close()
        print("Model Training Done.")
        return history
    

    @torch.no_grad()
    def evaluate(self, dataloader=None, set_name='valid'):
        loader = self.valid_loader if dataloader is None else dataloader
        self.model = self.model.eval()
        results_dict = defaultdict(list)
        for data, labels_forward in loader:
            data = data.to(self.args.device)
            labels_forward = labels_forward.to(self.args.device)

            if self.args.precision == 'high':
                with torch.autocast(device_type=self.args.device, dtype=torch.bfloat16):
                    class_logits, item_total_loss = self.model(source=data,
                                                               target=labels_forward,
                                                               pad_tokenId=self.collator.pad_value)
            else:
                class_logits, item_total_loss = self.model(source=data,
                                                           target=labels_forward,
                                                           pad_tokenId=self.collator.pad_value)

            candidates = torch.argmax(class_logits, dim=-1)
            metrics_dict = self.compute_metrics_func(labels_forward[:,1:], candidates[:,:-1], self.collator.pad_value)
            metrics_dict['loss'] = item_total_loss.item()

            for metric, value in metrics_dict.items():
                results_dict[set_name+"_"+metric].append(value)

        to_return = {}
        for name, values_list in results_dict.items():
            if 'accuracy' in name.lower() or 'bleu' in name.lower():
                to_return[name] = round(sum(values_list)/len(values_list), 4)
                # to_return[name] = round(sum(values_list)/len(values_list)*100, 2)
            else:
                to_return[name] = round(sum(values_list)/len(values_list), 4)
        return to_return