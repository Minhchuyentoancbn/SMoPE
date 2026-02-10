from torch.optim import Optimizer
import math
import numpy as np


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineSchedule(_LRScheduler):
    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos(
            (99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1))
        )

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


class _LRSchedulerIter(object):
    def __init__(self, optimizer, iter_step, n_epochs, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        last_iter = (last_epoch + 1) * iter_step - 1
        self.total_iters = iter_step * n_epochs
        schedule = np.ones((self.total_iters, len(self.optimizer.param_groups)))
        schedule = [
            schedule[:, i][:, np.newaxis] * self.base_lrs[i]
            for i in range(len(self.base_lrs))
        ]
        self.schedule = np.concatenate(schedule, axis=1)
        self.step(last_iter + 1)
        self.last_epoch = last_epoch
        self.last_iter = last_iter

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, iter):
        raise NotImplementedError

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(iter)):
            param_group["lr"] = lr


class CosineSchedulerIter(_LRSchedulerIter):
    def __init__(
        self,
        base_value,
        final_value,
        optimizer,
        iter_step,
        n_epochs,
        last_epoch=-1,
        warmup_epochs=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__(optimizer, iter_step, n_epochs, last_epoch)
        self.final_value = final_value
        warmup_iters = warmup_epochs * iter_step

        n_groups = len(self.optimizer.param_groups)
        assert (
            len(final_value) == n_groups and len(base_value) == n_groups
        ), f"Please provide {n_groups} final_value and base_value."

        freeze_schedule = np.zeros((freeze_iters)) + 1e-6
        freeze_schedule = np.repeat(
            freeze_schedule[:, np.newaxis], n_groups, axis=1
        )  # iters, n_groups

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        # warmup_schedule = np.repeat(warmup_schedule, n_groups, axis=1) # iters, n_groups

        # can base_value and final_value be list? -> each param_group of the optimizer
        iters = np.arange(self.total_iters - warmup_iters - freeze_iters)
        # schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = [
            final_value[i]
            + 0.5
            * (base_value[i] - final_value[i])
            * (1 + np.cos(np.pi * iters / len(iters)))
            for i in np.arange(n_groups)
        ]
        schedule = np.array(schedule).transpose()
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def get_lr(self, iter):
        if iter >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[iter]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    n_epochs = 50
    iter_step = 20
    model = nn.Sequential(
        nn.Linear(10, 10), nn.ReLU(True), nn.Linear(10, 4), nn.Sigmoid()
    )
    print(model)

    # create optimizers and scheduler
    # 1) one set of params
    optimizer_arg = {"params": model.parameters(), "lr": 0.01, "weight_decay": 0.0001}
    optimizer = torch.optim.Adam(**optimizer_arg)

    # 2) two sets of params
    # optimizer = torch.optim.Adam([{'params':model[0].parameters()},
    #                  {'params':model[2].parameters(),
    #                  'lr':0.1}],
    #                  lr=0.01)

    # scheduler = CosineSchedule(optimizer, K=n_epochs)
    scheduler_cfg = {
        "base_value": [0.01],  # [0.01, 0.001],
        "final_value": [0.0],  # [0.0, 0.0],
        "optimizer": optimizer,
        "iter_step": iter_step,
        "n_epochs": n_epochs,
        "last_epoch": -1,
        "warmup_epochs": 0,
        "start_warmup_value": 0,
        "freeze_iters": 0,
    }
    scheduler = CosineSchedulerIter(**scheduler_cfg)
    # lr_set = scheduler.get_lr() # only return one list of value for current epoch
    # generate lrs iteratively based on previous value?
    lr_set = []
    # for i in range(n_epochs):
    #     lr_set.extend(scheduler.get_lr())
    #     scheduler.step()
    for i in range(n_epochs * iter_step):
        lr_set.extend(scheduler.get_lr(i))
        scheduler.step()

    print(lr_set)
    fig = plt.figure(num=1, figsize=(4, 4))
    plt.plot(lr_set)
    # plt.show()
    plt.savefig("lr_schedule.png")
