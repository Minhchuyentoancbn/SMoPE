import os
import sys
import datetime
from tqdm import tqdm, trange
import math
from timm.utils import accuracy
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
from typing import Iterable
import learners
from utils import utils_tap
from utils.schedulers import CosineSchedulerIter


class Trainer:
    def __init__(self, args, seed, metric_keys, save_keys, round_id):

        # process inputs
        self.seed = seed
        self.round_id = round_id
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers

        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == "CIFAR100":
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32, 32, 3]
        elif args.dataset == "ImageNet_R":
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224, 224, 3]
            self.top_k = 1
        elif args.dataset == "CUB200":
            Dataset = dataloaders.iCUB200
            num_classes = 200
            self.dataset_size = [224, 224, 3]
            self.top_k = 1
        else:
            raise ValueError("Dataset not implemented!")

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if args.rand_split:
            print("=============================================")
            print("Shuffling....")
            print("pre-shuffle:" + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print("post-shuffle:" + str(class_order))
            print("=============================================")
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (
            args.max_task == -1 or len(self.tasks) < args.max_task
        ):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p : p + inc])
            self.tasks_logits.append(class_order_logits[p : p + inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1  # number of transforms per image
        if args.model_name.startswith("vit"):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(
            dataset=args.dataset,
            phase="train",
            aug=args.train_aug,
            resize_imnet=resize_imnet,
        )
        test_transform = dataloaders.utils.get_transform(
            dataset=args.dataset,
            phase="test",
            aug=args.train_aug,
            resize_imnet=resize_imnet,
        )
        self.train_dataset = Dataset(
            args.dataroot,
            train=True,
            lab=True,
            tasks=self.tasks,
            download_flag=True,
            transform=train_transform,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation,
        )
        self.test_dataset = Dataset(
            args.dataroot,
            train=False,
            tasks=self.tasks,
            download_flag=False,
            transform=test_transform,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation,
        )

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {
            "num_classes": num_classes,
            "lr": args.lr,
            "debug_mode": args.debug_mode == 1,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "schedule": args.schedule,
            "schedule_type": args.schedule_type,
            "iter_step": args.iter_step,  # add this for coswm
            "model_type": args.model_type,
            "model_name": args.model_name,
            "optimizer": args.optimizer,
            "gpuid": args.gpuid,
            "memory": args.memory,
            "temp": args.temp,
            "out_dim": num_classes,
            "overwrite": args.overwrite == 1,
            "DW": args.DW,
            "batch_size": args.batch_size,
            "upper_bound_flag": args.upper_bound_flag,
            "tasks": self.tasks_logits,
            "top_k": self.top_k,
            "prompt_param": [self.num_tasks, args.prompt_param],
            "pretrained_weight": args.pretrained_weight,
        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](
            self.learner_config
        )

        self.learner.print_model()

        # storing class mean and covariance
        # self.learner.cls_mean = dict()
        # self.learner.cls_cov = dict()
        self.num_classes = num_classes
        self.adaptive_pred = args.adaptive_pred
        self.n_centroids = args.n_centroids
        self.ca_method = args.ca_method
        self.crct_epochs = args.crct_epochs
        self.ca_lr = args.ca_lr
        self.ca_weight_decay = args.ca_weight_decay
        self.ca_batch_size_ratio = args.ca_batch_size_ratio

    def task_eval(self, t_index, local=False, task="acc"):

        val_name = self.task_names[t_index]
        print(f"validation split name (local {local}):", val_name)

        # eval
        self.test_dataset.load_dataset(
            t_index, train=True
        )  # train=True, only load task i data; else, load task 0~i data
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers,
            pin_memory=True,
        )
        if local:
            return self.learner.validation(
                test_loader, task_in=self.tasks_logits[t_index], task_metric=task
            )
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):

        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys:
            temp_table[mkey] = []
        temp_dir = self.log_dir + "/temp/"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # for each task
        for i in trange(self.max_task, desc="Task"):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print("======================", train_name, "=======================")

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[
                    self.learner_name
                ](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=int(self.workers),
                pin_memory=True,
            )

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()  # reinit all the prompt?

            # learn
            self.test_dataset.load_dataset(i, train=False)

            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers,
                pin_memory=True,
            )
            model_save_dir = (
                self.model_top_dir
                + "/models/repeat-"
                + str(self.round_id + 1)
                + "/task-"
                + self.task_names[i]
                + "/"
            )
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            avg_train_time, re_train = self.learner.learn_batch(
                train_loader, self.train_dataset, model_save_dir, test_loader
            )

            if self.adaptive_pred:
                # compute mean and variance
                self._compute_mean(model=self.learner.model, class_mask=self.tasks[i])

                # pseudo replay
                if i > 0:
                    self.train_task_adaptive_prediction(
                        model=self.learner.model, class_mask=self.tasks, task_id=i
                    )

            # save model
            if re_train:
                self.learner.save_model(model_save_dir)

            # evaluate acc -> NO NEED
            acc_table = []
            # acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i + 1):
                acc_table.append(
                    self.task_eval(j)
                )  # eval each task one-by-one, after learning a new task; on train dataset
            temp_table["acc"].append(np.mean(np.asarray(acc_table)))

            # self.learner.model.prompt.print_freq()

            print(f"Averaged accuracy for task {i + 1}: {temp_table['acc'][-1]}")

            if avg_train_time is not None:
                avg_metrics["time"]["global"][
                    i, self.round_id
                ] = avg_train_time  # time/epoch for each task
            # why not use avg_metrics to save other metrics such as 'acc'?

        return avg_metrics

    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict["global"]  # avg_metrics['acc']['global'] after training
        avg_acc_pt = acc_dict["pt"]
        # avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i + 1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]  # metric_table['acc']
                avg_acc_pt[j, i, self.round_id] = acc_table[val_name][
                    train_name
                ]  # metric_table['acc']
            avg_acc_history[i] = cls_acc_sum / (
                i + 1
            )  # metric_table['acc'], FAA of every task

        # Gather the final avg accuracy
        avg_acc_all[
            :, self.round_id
        ] = avg_acc_history  # metric_table['acc'] FAA? 'global'<-'pt'

        # repack dictionary and return
        return {"global": avg_acc_all, "pt": avg_acc_pt}

    def summarize_fr(self, fr_dict, acc_matrix):

        # unpack dictionary
        avg_fr_all = fr_dict["global"]

        avg_fr_history = [0] * self.max_task
        for task_id in range(self.max_task):
            if task_id > 0:
                avg_fr_history[task_id] = np.mean(
                    (np.max(acc_matrix[:, :task_id], axis=1) - acc_matrix[:, task_id])[
                        :task_id
                    ]
                )

        # Gather the final forgetting rate
        avg_fr_all[:, self.round_id] = avg_fr_history
        # repack dictionary and return
        return {"global": avg_fr_all}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](
            self.learner_config
        )

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                # try:
                #     if self.learner.model.module.prompt is not None:
                #         self.learner.model.module.prompt.process_task_count()
                # except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = (
                self.model_top_dir
                + "/models/repeat-"
                + str(self.round_id + 1)
                + "/task-"
                + self.task_names[i]
                + "/"
            )
            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc - three-level dict
            metric_table["acc"][
                self.task_names[i]
            ] = OrderedDict()  # 'acc' is a two-level dict
            # metric_table_local['acc'][self.task_names[i]] = OrderedDict() # local evaluation
            self.reset_cluster_labels = True
            accs = []
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table["acc"][val_name][self.task_names[i]] = self.task_eval(j)
                accs.append(metric_table["acc"][val_name][self.task_names[i]])
            print(f"Averaged accuracy for task {i + 1}: {np.mean(accs)}")

        # summarize metrics
        avg_metrics["acc"] = self.summarize_acc(
            avg_metrics["acc"], metric_table["acc"], metric_table_local["acc"]
        )
        avg_metrics["fr"] = self.summarize_fr(
            avg_metrics["fr"], avg_metrics["acc"]["pt"][:, :, self.round_id]
        )  # can use avg_metrics['acc']['pt-local'] for DIL

        return avg_metrics

    @torch.no_grad()
    def _compute_mean(self, model: torch.nn.Module, class_mask=None):
        model.eval()

        for cls_id in class_mask:
            self.train_dataset.load_class(cls_id)
            data_loader_cls = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers,
                pin_memory=True,
            )
            features_per_cls = []
            for i, (inputs, targets, task) in enumerate(data_loader_cls):
                # send data to gpu
                if self.learner.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                features = model(inputs, return_pre_logits=True, train=False)
                features_per_cls.append(features)
            features_per_cls = torch.cat(features_per_cls, dim=0)

            if self.ca_method == "covariance":
                self.learner.cls_mean[cls_id] = features_per_cls.mean(dim=0)
                self.learner.cls_cov[cls_id] = torch.cov(features_per_cls.T) + (
                    torch.eye(self.learner.cls_mean[cls_id].shape[-1]) * 1e-4
                ).to(inputs.device)
            elif self.ca_method == "multi-centroid":
                from sklearn.cluster import KMeans

                n_clusters = self.n_centroids  # default 10
                features_per_cls = features_per_cls.cpu().numpy()
                kmeans = KMeans(
                    n_clusters=n_clusters, n_init="auto", random_state=self.seed
                )
                kmeans.fit(features_per_cls)
                cluster_labels = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_labels == i]
                    cluster_mean = torch.tensor(
                        np.mean(cluster_data, axis=0), dtype=torch.float64
                    ).to(inputs.device)
                    cluster_var = torch.tensor(
                        np.var(cluster_data, axis=0), dtype=torch.float64
                    ).to(inputs.device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)

                self.learner.cls_mean[cls_id] = cluster_means
                self.learner.cls_cov[cls_id] = cluster_vars
            else:
                raise NotImplementedError

    def train_task_adaptive_prediction(
        self, model: torch.nn.Module, class_mask=None, task_id=-1
    ):
        model.train()
        run_epochs = self.crct_epochs
        crct_num = 0
        valid_out_dim = self.learner.valid_out_dim
        ca_lr = self.ca_lr
        weight_decay = self.ca_weight_decay
        batch_size = self.batch_size
        param_list = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "prompt" not in n
        ]
        network_params = [
            {"params": param_list, "lr": ca_lr, "weight_decay": weight_decay}
        ]

        optimizer = torch.optim.AdamW(
            network_params, lr=ca_lr / 10, weight_decay=weight_decay
        )  # ****

        criterion = torch.nn.CrossEntropyLoss()
        if self.learner.gpu:
            criterion = criterion.cuda()

        for i in range(task_id):  # only take part of the samples after random permute
            crct_num += len(class_mask[i])

        scheduler_cfg = {
            "base_value": [ca_lr / 10],
            "final_value": [1e-6],
            "optimizer": optimizer,
            "iter_step": crct_num,
            "n_epochs": run_epochs,
            "last_epoch": -1,
            "warmup_epochs": 0,
            "start_warmup_value": 0,
            "freeze_iters": 0,
        }
        scheduler = CosineSchedulerIter(**scheduler_cfg)

        for epoch in range(run_epochs):

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = int(batch_size * self.ca_batch_size_ratio)  # default 5

            metric_logger = utils_tap.MetricLogger(delimiter="  ")
            metric_logger.add_meter(
                "Lr", utils_tap.SmoothedValue(window_size=1, fmt="{value:.6f}")
            )
            metric_logger.add_meter(
                "Loss", utils_tap.SmoothedValue(window_size=1, fmt="{value:.4f}")
            )

            if self.ca_method == "covariance":
                for i in range(task_id + 1):
                    for c_id in class_mask[i]:
                        mapped_c_id = self.train_dataset.class_mapping[c_id]
                        mean = self.learner.cls_mean[c_id].detach()
                        cov = self.learner.cls_cov[c_id].detach()
                        m = MultivariateNormal(mean.float(), cov.float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)

                        sampled_label.extend([mapped_c_id] * num_sampled_pcls)
            elif self.ca_method == "multi-centroid":
                for i in range(task_id + 1):
                    for c_id in class_mask[i]:
                        mapped_c_id = self.train_dataset.class_mapping[c_id]
                        for cluster in range(len(self.learner.cls_mean[c_id])):
                            mean = self.learner.cls_mean[c_id][cluster]
                            var = self.learner.cls_cov[c_id][cluster]
                            if var.mean() == 0:
                                continue
                            m = MultivariateNormal(
                                mean.float(),
                                (
                                    torch.diag(var)
                                    + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)
                                ).float(),
                            )
                            sampled_data_single = m.sample(
                                sample_shape=(num_sampled_pcls,)
                            )
                            sampled_data.append(sampled_data_single)
                            sampled_label.extend([mapped_c_id] * num_sampled_pcls)
            else:
                raise NotImplementedError

            sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
            sampled_label = torch.tensor(sampled_label).long().to(sampled_data.device)
            # print(sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]

                try:
                    logits = model.module.forward_fc(inp)
                except:
                    logits = model.forward_fc(inp)

                logits = logits[:, :valid_out_dim]

                loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
                acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # step inside loop for Iter scheduler

                metric_logger.update(Loss=loss.item())
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters["Acc@1"].update(acc1.item(), n=inp.shape[0])
                metric_logger.meters["Acc@5"].update(acc5.item(), n=inp.shape[0])

            print("Averaged stats:", metric_logger)
