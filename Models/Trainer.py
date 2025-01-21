import os
import torch
from Models.TrainingArguments import TrainingArguments
from utils.data_utils import MT_Dataset, MYCollate
from utils.model_utils import training, get_parameters_info, plot_loss
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, args:TrainingArguments, model:torch.nn.Module,
                 train_ds:MT_Dataset, valid_ds:MT_Dataset,
                 collator:MYCollate, compute_metrics_func):
        self.args = args
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.collator = collator
        self.compute_metrics_func = compute_metrics_func


    def train(self):

        pin_memory = True if self.args.device == 'cuda' else False
        generator = torch.manual_seed(self.args.seed) if self.args.seed else None

        train_loader = DataLoader(self.train_ds,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  collate_fn=self.collator,
                                  num_workers=self.args.cpu_num_workers,
                                  generator=generator,
                                  pin_memory=pin_memory)

        valid_loader = DataLoader(self.valid_ds,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  collate_fn=self.collator,
                                  num_workers=self.args.cpu_num_workers,
                                  generator=generator,
                                  pin_memory=pin_memory)

        names, tr, nontr = get_parameters_info(model=model)
        print(f"{'Module':<15}{'Trainable':>15}{'Non-Trainable':>15}")
        for n, ttp, ntp in zip(names, tr, nontr):
            print(f"{n:<15}{ttp:>15,}{ntp:>15,}")
        model.to(self.args.device)

        print("Starting Model Training...")
        class_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.collator.pad_value)  # Classification loss
        train_class_losses, val_class_losses, model, optim, epochs = training(model=model, criterion=class_criterion,
                                                        optimizer=optim, train_loader=train_loader,
                                                        valid_loader=valid_loader, epochs=self.args.num_train_epochs, device=self.args.device)
        print("Model Training Done.")

        plot_loss(train_class_losses, val_class_losses, self.args.plots_dir, self.args.run_name)

        ## Save Entire Model
        model_path = os.path.join(self.args.output_dir, self.args.run_name)
        if self.args.onnx:
            print("Converting To ONNX framework...")
            ## onnx
        else:
            ## pytorch
            torch.save({'epoch': self.args.num_train_epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.args.optimizer.state_dict(),
                        "Training losses": train_class_losses,
                        "Validation losses": val_class_losses}, f"{model_path}.pth")

        return train_class_losses, val_class_losses
