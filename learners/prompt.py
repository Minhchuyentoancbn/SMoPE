from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule, CosineSchedulerIter
from torch.autograd import Variable, Function


class Prompt(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config["prompt_param"]
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean
        )  # logits=cls_token if pen=True, else self.model.last(cls_token)
        logits = logits[:, : self.valid_out_dim]

        # ce with heuristic
        logits[:, : self.last_valid_out_dim] = -float(
            "inf"
        )  # TODO: this gives inf loss if self.memory_size > 0
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self):

        if len(self.config["gpuid"]) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            if self.model.prompt.task_count == 0:
                base_params = [
                    p
                    for name, p in self.model.prompt.named_parameters()
                    if p.requires_grad
                ]
            else:
                # base_params = [p for name, p in self.model.prompt.named_parameters() if 'mlp' not in name]
                # base_params = list(self.model.prompt.parameters())
                base_params = [
                    p
                    for name, p in self.model.prompt.named_parameters()
                    if p.requires_grad
                ]

            base_fc_params = list(self.model.last.parameters())
        base_params = {
            "params": base_params,
            "lr": self.config["lr"] * 5,
            "weight_decay": self.config["weight_decay"],
        }  # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {
            "params": base_fc_params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        optimizer_arg = [base_params, base_fc_params]

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total params:", total_params)
        tune_params = sum(p.numel() for p in base_params["params"]) + sum(
            p.numel() for p in base_fc_params["params"]
        )
        print("Tune params:", tune_params)
        tuned_percent = tune_params / total_params * 100
        print(f"Tune ratio: {tuned_percent:.2f}")
        prompt_params = sum(p.numel() for p in base_params["params"])
        print("Prompt params:", prompt_params)
        prompt_percent = prompt_params / total_params * 100
        print(f"Prompt ratio: {prompt_percent:.2f}")

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](optimizer_arg)

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )
        elif self.schedule_type == "coswm":
            # print(self.config)
            scheduler_cfg = {
                "base_value": [self.config["lr"] * 5, self.config["lr"]],
                "final_value": [1e-6, 1e-6],
                "optimizer": self.optimizer,
                "iter_step": self.config["iter_step"],
                "n_epochs": self.config["schedule"][-1],
                "last_epoch": -1,
                "warmup_epochs": self.config["schedule"][1],
                "start_warmup_value": 0,
                "freeze_iters": self.config["schedule"][0],
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config["gpuid"]) > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=self.config["gpuid"],
                output_device=self.config["gpuid"][0],
            )
        return self


# Our method
class VQPrompt(Prompt):
    def __init__(self, learner_config):
        super(VQPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="qt",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model


# Our method
class OnePrompt(Prompt):
    def __init__(self, learner_config):
        super(OnePrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="smope",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model

    def init_optimizer(self, epoch_factor=1):

        if len(self.config["gpuid"]) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            base_params = [
                p for name, p in self.model.prompt.named_parameters() if p.requires_grad
            ]

            base_fc_params = list(self.model.last.parameters())

        base_params = {
            "params": base_params,
            "lr": self.config["lr"] * 5,
            "weight_decay": self.config["weight_decay"],
        }  # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {
            "params": base_fc_params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        optimizer_arg = [base_params, base_fc_params]

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total params:", total_params)
        tune_params = sum(p.numel() for p in base_params["params"]) + sum(
            p.numel() for p in base_fc_params["params"]
        )
        print("Tune params:", tune_params)
        tuned_percent = tune_params / total_params * 100
        print(f"Tune ratio: {tuned_percent:.2f}")
        prompt_params = sum(p.numel() for p in base_params["params"])
        print("Prompt params:", prompt_params)
        prompt_percent = prompt_params / total_params * 100
        print(f"Prompt ratio: {prompt_percent:.2f}")

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](optimizer_arg)
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(self.optimizer, K=num_epochs)
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )
        elif self.schedule_type == "coswm":
            # print(self.config)
            scheduler_cfg = {
                "base_value": [self.config["lr"] * 5, self.config["lr"]],
                "final_value": [1e-6, 1e-6],
                "optimizer": self.optimizer,
                "iter_step": self.config["iter_step"],
                "n_epochs": num_epochs,
                "last_epoch": -1,
                "warmup_epochs": self.config["schedule"][1],
                "start_warmup_value": 0,
                "freeze_iters": self.config["schedule"][0],
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def update_model(self, inputs, targets, dense=False):
        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean, dense=dense
        )  # logits=cls_token if pen=True, else self.model.last(cls_token)

        logits = logits[:, : self.valid_out_dim]

        # ce with heuristic
        logits[:, : self.last_valid_out_dim] = -float(
            "inf"
        )  # TODO: this gives inf loss if self.memory_size > 0
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def learn_prompt(self, train_loader, batch_time, dense=False, epoch_factor=1):

        losses = AverageMeter()
        acc = AverageMeter()

        batch_timer = Timer()
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        if self.schedule_type == "coswm":  # step scheduler at each iter
            for epoch in range(num_epochs):
                self.epoch = epoch

                # for param_group in self.optimizer.param_groups:
                #     self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    # verify in train mode
                    self.model.train()
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y, dense=dense)
                    self.scheduler.step()

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(
                        output, y, task, acc, topk=(self.top_k,)
                    )  # already calculate train acc here? but logit range is narrow
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=self.epoch + 1, total=num_epochs
                    ),
                    end=" ",
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        loss=losses, acc=acc
                    )
                )

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
        else:
            for epoch in range(num_epochs):
                self.epoch = epoch

                if epoch > 0:
                    self.scheduler.step()
                # for param_group in self.optimizer.param_groups:
                #     self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y, dense=dense)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(
                        output, y, task, acc, topk=(self.top_k,)
                    )  # already calculate train acc here? but logit range is narrow
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=self.epoch + 1, total=num_epochs
                    )
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        loss=losses, acc=acc
                    )
                )

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
                # Cannot load, because in run.py, r<start_r is not allowed
                # all r in the loop, is not trained
                # I changed that in run.py to see effects
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log("Optimizer is reset!")
            self.init_optimizer()
        if need_train:

            # data weighting
            self.data_weighting(train_dataset)

            batch_time = AverageMeter()

            if self.task_count == 0:
                print("-" * 10)
                print("Initial training...")
                self.log("Optimizer is reset!")
                epoch_factor = 0.5
                self.init_optimizer(epoch_factor=epoch_factor)
                self.learn_prompt(
                    train_loader, batch_time, dense=True, epoch_factor=epoch_factor
                )
                self.log("Optimizer is reset!")
                self.init_optimizer()

            self.learn_prompt(train_loader, batch_time)

            print("-" * 10)
            print("Selecting Experts...")
            num_samples = 0

            for i, (x, y, task) in enumerate(train_loader):
                # verify in train mode
                self.model.eval()
                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                with torch.no_grad():
                    prompt_scores = self.model(x, train=False, return_attn=True)

                self.model.prompt.update_prompt(prompt_scores)
                num_samples += x.size(0)

            self.model.prompt.update_num_samples(num_samples)
            # self.model.prompt.print_freq()
            print("-" * 10)

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(
                self.memory_size, np.arange(self.last_valid_out_dim)
            )

        try:
            return batch_time.avg, need_train
        except:
            return None, need_train


# @inproceedings{smith2023coda,
#   title={CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning},
#   author={Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={11909--11919},
#   year={2023}
# }
class CODAPrompt(Prompt):
    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="coda", prompt_param=self.prompt_param
        )
        return model


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):
    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="dual", prompt_param=self.prompt_param
        )
        return model


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):
    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="l2p", prompt_param=self.prompt_param
        )
        return model
