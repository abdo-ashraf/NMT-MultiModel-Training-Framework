import os
import torch
from .TrainingArguments import TrainingArguments
from utils.data_utils import MT_Dataset, MYCollate
from utils.model_utils import training_loop, plot_loss
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

    def train(self):

        print("Starting Model Training...")
        class_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.collator.pad_value)  # Classification loss
        if hasattr(self.model, 'maxlen'):
            print('AdamW will be used for Transformer-based models')
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            print('Adam will be used for non-Transformer models')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        train_class_losses, val_class_losses = training_loop(model=self.model,
                                                            criterion=class_criterion,
                                                            optimizer=optimizer,
                                                            train_loader=self.train_loader,
                                                            valid_loader=self.valid_loader,
                                                            epochs=self.args.num_train_epochs,
                                                            device=self.args.device)
        print("Model Training Done.")

        plot_loss(train_class_losses, val_class_losses, self.args.save_plots_dir, self.args.run_name)

        ## Save Entire Model
        model_path = os.path.join(self.args.save_models_dir, self.args.run_name)
        if self.args.onnx:
            print("Converting To ONNX framework...")
            ## onnx
        else:
            ## pytorch
            torch.save({'epoch': self.args.num_train_epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "Training losses": train_class_losses,
                        "Validation losses": val_class_losses}, f"{model_path}.pth")

        return train_class_losses, val_class_losses
