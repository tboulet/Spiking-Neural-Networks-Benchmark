from utils import set_seed
from datetime import datetime
from uuid import uuid4
import os

import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from snn_benchmark.configs import Config
from snn_benchmark.data.base_datasets import Augs

eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.build_model()
        self.init_model()

        self.init_pos = []
        if self.config.model_type.startswith("snn_delays"):
            for i in range(len(self.blocks)):
                self.init_pos.append(
                    np.copy(self.blocks[i][0][0].P.cpu().detach().numpy())
                )

    def optimizers(self):
        ##################################
        #  returns a list of optimizers
        ##################################
        optimizers_return = []

        if self.config.model_type in ["snn_delays", "snn_delays_lr0", "snn"]:
            if self.config.optimizer_w == "adam":
                optimizers_return.append(
                    optim.Adam(
                        [
                            {
                                "params": self.weights,
                                "lr": self.config.lr_w,
                                "weight_decay": self.config.weight_decay,
                            },
                            {
                                "params": self.weights_plif,
                                "lr": self.config.lr_w,
                                "weight_decay": self.config.weight_decay,
                            },
                            {
                                "params": self.weights_bn,
                                "lr": self.config.lr_w,
                                "weight_decay": 0,
                            },
                        ]
                    )
                )
            if self.config.model_type == "snn_delays":
                if self.config.optimizer_pos == "adam":
                    optimizers_return.append(
                        optim.Adam(
                            self.positions, lr=self.config.lr_pos, weight_decay=0
                        )
                    )
        elif self.config.model_type == "ann":
            if self.config.optimizer_w == "adam":
                optimizers_return.append(
                    optim.Adam(
                        self.model.parameters(), lr=self.config.lr_w, betas=(0.9, 0.999)
                    )
                )

        return optimizers_return

    def schedulers(self, optimizers):
        ##################################
        #  returns a list of schedulers
        #  if self.config.scheduler_x is none:  list will be empty
        ##################################
        schedulers_return = []

        if self.config.model_type in ["snn_delays", "snn_delays_lr0", "snn"]:
            if self.config.scheduler_w == "one_cycle":
                schedulers_return.append(
                    torch.optim.lr_scheduler.OneCycleLR(
                        optimizers[0],
                        max_lr=self.config.max_lr_w,
                        total_steps=self.config.epochs,
                    )
                )
            elif self.config.scheduler_w == "cosine_a":
                schedulers_return.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizers[0], T_max=self.config.t_max_w
                    )
                )

            if self.config.model_type == "snn_delays":
                if self.config.scheduler_pos == "one_cycle":
                    schedulers_return.append(
                        torch.optim.lr_scheduler.OneCycleLR(
                            optimizers[1],
                            max_lr=self.config.max_lr_pos,
                            total_steps=self.config.epochs,
                        )
                    )
                elif self.config.scheduler_pos == "cosine_a":
                    schedulers_return.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizers[1], T_max=self.config.t_max_pos
                        )
                    )

        elif self.config.model_type == "ann":
            if self.config.scheduler_w == "one_cycle":
                schedulers_return.append(
                    torch.optim.lr_scheduler.OneCycleLR(
                        optimizers[0],
                        max_lr=self.config.max_lr_w,
                        total_steps=self.config.epochs,
                    )
                )

        return schedulers_return

    def calc_loss(self, output, y):
        if self.config.loss == "mean":
            m = torch.mean(output, 0)
        elif self.config.loss == "max":
            m, _ = torch.max(output, 0)
        elif self.config.loss == "spike_count":
            m = torch.sum(output, 0)
        elif self.config.loss == "sum":
            softmax_fn = nn.Softmax(dim=2)
            m = torch.sum(softmax_fn(output), 0)

        # probably better to add it in init, or in general do it one time only
        if self.config.loss_fn == "CEloss":
            # compare using this to directly using nn.CrossEntropyLoss

            CEloss = nn.CrossEntropyLoss()
            loss = CEloss(m, y)
            # log_softmax_fn = nn.LogSoftmax(dim=1)
            # loss_fn = nn.NLLLoss()
            # log_p_y = log_softmax_fn(m)
            # loss = loss_fn(log_p_y, y)

            return loss

    def calc_metric(self, output, y):
        # mean accuracy over batch
        if self.config.loss == "mean":
            m = torch.mean(output, 0)
        elif self.config.loss == "max":
            m, _ = torch.max(output, 0)
        elif self.config.loss == "spike_count":
            m = torch.sum(output, 0)
        elif self.config.loss == "sum":
            softmax_fn = nn.Softmax(dim=2)
            m = torch.sum(softmax_fn(output), 0)

        return np.mean(
            (torch.max(y, 1)[1] == torch.max(m, 1)[1]).detach().cpu().numpy()
        )

    def fine_tune(self, train_loader, valid_loader, test_loader, device):
        # if self.config.spiking_neuron_type == 'plif' and self.config.spiking_neuron_type_finetuning == 'lif':

        self.config.DCLSversion = "max"
        self.config.model_type = "snn_delays_lr0"

        self.config.lr_w = self.config.lr_w_finetuning
        self.config.max_lr_w = self.config.max_lr_w_finetuning

        self.config.dropout_p = self.config.dropout_p_finetuning
        self.config.stateful_synapse_learnable = (
            self.config.stateful_synapse_learnable_finetuning
        )
        self.config.spiking_neuron_type = self.config.spiking_neuron_type_finetuning
        self.config.epochs = self.config.epochs_finetuning

        self.config.final_epoch = 0

        self.config.wandb_run_name = self.config.wandb_run_name_finetuning
        self.config.wandb_group_name = self.config.wandb_group_name_finetuning

        self.__init__(self.config)
        self.to(device)
        self.load_state_dict(torch.load(self.config.save_model_path), strict=False)

        for i in range(len(self.blocks)):
            self.blocks[i][0][0].SIG *= 0

            # if i < len(self.blocks) - 2:
            #     self.blocks[i][0][0].weight.requires_grad = False
            #     if self.config.use_batchnorm:
            #         self.blocks[i][0][1].weight.requires_grad = False
            #         self.blocks[i][0][1].bias.requires_grad = False

        self.round_pos()

        self.config.save_model_path = self.config.save_model_path_finetuning
        self.train_model(train_loader, valid_loader, test_loader, device)

    def train_model(self, train_loader, valid_loader, test_loader, device):
        #######################################################################################
        #           Main Training Loop for all models
        #
        #
        #
        ##################################    Initializations    #############################

        set_seed(self.config.seed)

        if self.config.use_wandb:
            cfg = {k: v for k, v in dict(vars(Config)).items() if "__" not in k}

            wandb.login(key=self.config.wandb_token)

            wandb.init(
                project=self.config.wandb_project_name,
                name=self.config.wandb_run_name,
                config=cfg,
                group=self.config.wandb_group_name,
            )

        optimizers = self.optimizers()
        schedulers = self.schedulers(optimizers)

        augmentations = Augs(self.config)

        ##################################    Train Loop    ##############################

        loss_epochs = {"train": [], "valid": [], "test": []}
        metric_epochs = {"train": [], "valid": [], "test": []}
        best_metric_val = 0  # 1e6
        best_metric_test = 0  # 1e6
        best_loss_val = 1e6

        pre_pos_epoch = self.init_pos.copy()
        pre_pos_5epochs = self.init_pos.copy()
        batch_count = 0
        for epoch in range(self.config.epochs):
            self.train()
            # last element in the tuple corresponds to the collate_fn return
            loss_batch, metric_batch = [], []
            pre_pos = pre_pos_epoch.copy()

            for i, (x, y, *_) in enumerate(tqdm(train_loader)):
                # x for shd and ssc is: (batch, time, neurons)
                y = F.one_hot(y.to(torch.int64), self.config.n_outputs).float()

                if self.config.augment:
                    x, y = augmentations(x, y)

                # Adapt to right shape
                if len(x.shape) == 3:
                    x = x.permute(1, 0, 2).float().to(device)  # (time, batch, neurons)
                elif len(x.shape) == 2:
                    x = (
                        x.unsqueeze(0).float().to(device)
                    )  # (1, batch, neurons)    # (1, 105, 4)

                y = y.to(device)

                for opt in optimizers:
                    opt.zero_grad()
                output = self.forward(x)
                loss = self.calc_loss(output, y)

                loss.backward()
                for opt in optimizers:
                    opt.step()

                metric = self.calc_metric(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                self.reset_model(train=True)

                if self.config.use_wandb and self.config.model_type == "snn_delays":
                    wandb_pos_log = {}
                    for b in range(len(self.blocks)):
                        curr_pos = self.blocks[b][0][0].P.cpu().detach().numpy()
                        wandb_pos_log[f"dpos{b}"] = np.abs(curr_pos - pre_pos[b]).mean()
                        pre_pos[b] = curr_pos.copy()

                    wandb_pos_log.update({"batch": batch_count})
                    wandb.log(wandb_pos_log)
                    batch_count += 1

            if self.config.model_type == "snn_delays":
                pos_logs = {}
                for b in range(len(self.blocks)):
                    pos_logs[f"dpos{b}_epoch"] = np.abs(
                        pre_pos[b] - pre_pos_epoch[b]
                    ).mean()
                    pre_pos_epoch[b] = pre_pos[b].copy()

                if epoch % 5 == 0 and epoch > 0:
                    for b in range(len(self.blocks)):
                        pos_logs[f"dpos{b}_5epochs"] = np.abs(
                            pre_pos[b] - pre_pos_5epochs[b]
                        ).mean()
                        pre_pos_5epochs[b] = pre_pos[b].copy()

            loss_epochs["train"].append(np.mean(loss_batch))
            metric_epochs["train"].append(np.mean(metric_batch))

            for scheduler in schedulers:
                scheduler.step()
            self.decrease_sig(epoch)

            ##################################    Eval Loop    #########################

            loss_valid, metric_valid = self.eval_model(valid_loader, device)

            loss_epochs["valid"].append(loss_valid)
            metric_epochs["valid"].append(metric_valid)

            if test_loader:
                loss_test, metric_test = self.eval_model(test_loader, device)
            else:
                # could be improved
                loss_test, metric_test = 100, 0

            loss_epochs["test"].append(loss_test)
            metric_epochs["test"].append(metric_test)

            ########################## Logging and Plotting  ##########################

            print(
                f"=====> Epoch {epoch} : \nLoss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100*metric_epochs['train'][-1]:.2f}%"
            )
            print(
                f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100*metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100*max(metric_epochs['valid'][-1], best_metric_val):.2f}%"
            )

            if test_loader:
                print(
                    f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100*metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100*max(metric_epochs['test'][-1], best_metric_test):.2f}%"
                )

            if self.config.use_wandb:
                lr_w = (
                    schedulers[0].get_last_lr()[0]
                    if self.config.scheduler_w != "none"
                    else self.config.lr_w
                )
                lr_pos = (
                    schedulers[1].get_last_lr()[0]
                    if self.config.model_type == "snn_delays"
                    and self.config.scheduler_pos != "none"
                    else self.config.lr_pos
                )

                wandb_logs = {
                    "Epoch": epoch,
                    "loss_train": loss_epochs["train"][-1],
                    "acc_train": metric_epochs["train"][-1],
                    "loss_valid": loss_epochs["valid"][-1],
                    "acc_valid": metric_epochs["valid"][-1],
                    "loss_test": loss_epochs["test"][-1],
                    "acc_test": metric_epochs["test"][-1],
                    "lr_w": lr_w,
                    "lr_pos": lr_pos,
                }

                try:
                    model_logs = self.get_model_wandb_logs()
                except:
                    model_logs = {}

                wandb_logs.update(model_logs)

                if self.config.model_type == "snn_delays":
                    wandb_logs.update(pos_logs)

                wandb.log(wandb_logs)

            os.makedirs(self.config.save_model_path, exist_ok=True)
            if (
                metric_valid > best_metric_val
            ):  #  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                print("# Saving best Metric model...")
                torch.save(
                    self.state_dict(),
                    self.config.save_model_path.replace("REPL", "Best_ACC"),
                )
                best_metric_val = metric_valid

            if (
                loss_valid < best_loss_val
            ):  #  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                print("# Saving best Loss model...")
                torch.save(
                    self.state_dict(),
                    self.config.save_model_path.replace("REPL", "Best_Loss"),
                )
                best_loss_val = loss_valid

            if (
                metric_test > best_metric_test
            ):  #  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                best_metric_test = metric_test

        if self.config.use_wandb:
            wandb.run.finish()

    def eval_model(self, loader, device):
        # the creation of a temporary checkpoint
        # should be removed as it can lead to errors
        torch.save(self.state_dict(), eventid + ".pt")
        self.eval()
        with torch.no_grad():
            if self.config.model_type.startswith("snn_delays"):
                for i in range(len(self.blocks)):
                    self.blocks[i][0][0].SIG *= 0
                    self.blocks[i][0][0].version = "max"
                    self.blocks[i][0][0].DCK.version = "max"
                self.round_pos()

            loss_batch, metric_batch = [], []
            for i, (x, y, *_) in enumerate(tqdm(loader)):
                y = F.one_hot(y, self.config.n_outputs).float()

                # Adapt to right shape
                if len(x.shape) == 3:
                    x = x.permute(1, 0, 2).float().to(device)  # (time, batch, neurons)
                elif len(x.shape) == 2:
                    x = (
                        x.unsqueeze(0).float().to(device)
                    )  # (1, batch, neurons)    # (1, 105, 4)
                y = y.to(device)

                output = self.forward(x)

                loss = self.calc_loss(output, y)
                metric = self.calc_metric(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                self.reset_model(train=False)

            if self.config.DCLSversion == "gauss" and self.config.model_type != "snn":
                for i in range(len(self.blocks)):
                    self.blocks[i][0][0].version = "gauss"
                    self.blocks[i][0][0].DCK.version = "gauss"

            self.load_state_dict(torch.load(eventid + ".pt"), strict=True)
            if os.path.exists(eventid + ".pt"):
                os.remove(eventid + ".pt")
            else:
                print(f"File '{eventid + '.pt'}' does not exist.")

        return np.mean(loss_batch), np.mean(metric_batch)
