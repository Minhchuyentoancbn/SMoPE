import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy


# Our method
class OnePrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, num_heads=12):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks

        self.e_p_length = int(prompt_param[0])
        self.topk = int(prompt_param[1])
        self.mu_router = float(prompt_param[2])
        self.mu_router_old = float(prompt_param[3])
        self.eps = float(prompt_param[4])
        self.e_layers = [0, 1, 2, 3, 4, 5]

        self.num_heads = num_heads
        head_dim = self.key_d // self.num_heads
        self.head_dim = head_dim
        self.num_experts = self.e_p_length // 2

        # e prompt init
        for e in self.e_layers:
            for l in range(self.num_experts):
                for h in range(self.num_heads):
                    p_k = tensor_prompt(1, head_dim)
                    setattr(self, f"e_pk_{e}_{l}_{h}", p_k)
                    p_v = tensor_prompt(1, head_dim)
                    setattr(self, f"e_pv_{e}_{l}_{h}", p_v)

                    freq = 0
                    setattr(self, f"e_freq_{e}_{l}_{h}", freq)

        self.num_samples = 0
        self.used_frequently = [
            [[False for _ in range(self.num_experts)] for _ in range(self.num_heads)]
            for _ in self.e_layers
        ]
        self.router_criterion = nn.CrossEntropyLoss()

    def process_task_count(self):
        self.task_count += 1
        self.save_old_prompts()

        for e in self.e_layers:
            for h in range(self.num_heads):
                freq_head = torch.tensor(
                    [
                        getattr(self, f"e_freq_{e}_{l}_{h}")
                        for l in range(self.num_experts)
                    ]
                ).to(torch.float32)
                freq_head_mean = freq_head.sum() / (freq_head > 0).sum().clamp(
                    min=1.0
                )  # avoid division by zero
                for l in range(self.num_experts):
                    freq = getattr(self, f"e_freq_{e}_{l}_{h}")
                    if freq >= freq_head_mean:
                        self.used_frequently[e][h][l] = True

    def router_loss(self, prompt_scores, task_id=-1, topk=-1):
        loss = 0.0

        if self.mu_router > 0 and topk > 0:
            max_loss = 0
            for e in self.e_layers:
                prompt_score, prompt_score_label_ = prompt_scores[
                    e
                ]  # (B, num_heads, 1, num_prompt)
                _, indices = torch.topk(prompt_score_label_, self.topk, dim=-1)
                mask = torch.zeros_like(prompt_score).scatter(
                    -1, indices, 1.0
                )  # (B, num_heads, 1, num_prompt)
                not_mask = 1.0 - mask
                not_mask = not_mask * self.eps
                with torch.no_grad():
                    prompt_score_max = prompt_score.max(
                        dim=-1, keepdim=True
                    ).values  # (B, num_heads, 1, 1)
                    prompt_score_min = prompt_score.min(
                        dim=-1, keepdim=True
                    ).values  # (B, num_heads, 1, 1)
                prompt_score = prompt_score + not_mask * (
                    prompt_score_max - prompt_score_min
                )  # (B, num_heads, 1, num_prompt)

                for h in range(self.num_heads):
                    indices_h = indices[:, h, 0, :]  # (B, topk)
                    for i in range(self.topk):
                        max_loss += (
                            self.router_criterion(
                                prompt_score[:, h, 0, :], indices_h[:, i]
                            )
                            * self.mu_router
                        )

            loss += max_loss

        if task_id > 0 and self.mu_router_old > 0:
            for e in self.e_layers:
                for h in range(self.num_heads):
                    current_pk_h = []
                    old_pk_h = []
                    sampled = []

                    for l in range(self.num_experts):
                        current_pk = getattr(self, f"e_pk_{e}_{l}_{h}")
                        old_pk = getattr(self, f"old_e_pk_{e}_{l}_{h}")
                        current_pk_h.append(current_pk)
                        old_pk_h.append(old_pk)
                        sampled.append(self.used_frequently[e][h][l])

                    current_pk_h = torch.cat(current_pk_h, dim=0)
                    old_pk_h = torch.cat(old_pk_h, dim=0)
                    sampled = torch.tensor(
                        sampled, device=current_pk_h.device, dtype=torch.bool
                    )
                    sampled_pk_h = old_pk_h[sampled]  # (num_used, head_dim)

                    old_logits = sampled_pk_h @ old_pk_h.t()  # (num_used, num_experts)
                    current_logits = (
                        sampled_pk_h @ current_pk_h.t()
                    )  # (num_used, num_experts)
                    _, indices = torch.topk(
                        old_logits, self.topk, dim=-1
                    )  # (num_used, topk)

                    for i in range(self.topk):
                        loss += (
                            self.router_criterion(current_logits, indices[:, i])
                            * self.mu_router_old
                        )

        return loss

    def forward(self, x_querry, l, x_block, train=False, task_id=None, noise=False):
        e_valid = False
        loss = 0.0

        if l in self.e_layers:
            e_valid = True
            B = x_block.shape[0]
            pk = []  # (num_heads, num_prompt, head_dim)
            pv = []
            eps_decay = []
            for h in range(self.num_heads):
                pk_h = []
                pv_h = []
                eps_decay_h = []
                for i in range(self.num_experts):
                    _pk_h = getattr(self, f"e_pk_{l}_{i}_{h}")
                    _pv_h = getattr(self, f"e_pv_{l}_{i}_{h}")
                    pk_h.append(_pk_h)
                    pv_h.append(_pv_h)
                    freq = getattr(self, f"e_freq_{l}_{i}_{h}")
                    if train and self.used_frequently[l][h][i]:
                        eps_decay_h.append(self.eps)
                    else:
                        if not train and freq == 0:
                            eps_decay_h.append(2.0)
                        else:
                            eps_decay_h.append(0.0)

                pk_h = torch.cat(pk_h, dim=0).unsqueeze(0)  # (1, num_prompt, head_dim)
                pv_h = torch.cat(pv_h, dim=0).unsqueeze(0)  # (1, num_prompt, head_dim)

                pk.append(pk_h)  # (num_heads, num_prompt, head_dim)
                pv.append(pv_h)  # (num_heads, num_prompt, head_dim)
                eps_decay.append(eps_decay_h)

            pk = torch.cat(pk, dim=0)  # (num_heads, num_prompt, head_dim)
            pv = torch.cat(pv, dim=0)  # (num_heads, num_prompt, head_dim)
            eps_decay = torch.tensor(
                eps_decay, device=pk.device, dtype=torch.float32
            )  # (num_heads, num_experts)
            Ek = pk.unsqueeze(0).expand(B, -1, -1, -1)
            Ev = pv.unsqueeze(0).expand(B, -1, -1, -1)

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev, eps_decay]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def print_freq(self):
        print("-" * 20)
        print(f"Num Samples: {self.num_samples}")
        for e in self.e_layers:
            for h in range(self.num_heads):
                print("-" * 10)
                print(f"Layer {e} Head {h}:")
                freq = []
                for l in range(self.num_experts):
                    freq.append(getattr(self, f"e_freq_{e}_{l}_{h}"))
                print(freq)
        print("-" * 20)

    def update_num_samples(self, num_samples):
        self.num_samples += num_samples

    def update_prompt(self, prompt_scores):
        if self.topk > 0:
            for e in self.e_layers:
                prompt_score, _ = prompt_scores[e]  # (B, num_heads, 1, num_prompt)
                for h in range(self.num_heads):
                    weight = prompt_score[:, h, 0, :]  # (B, num_prompt)
                    _, indices = torch.topk(weight, self.topk, dim=-1)
                    indices = indices.reshape(-1)

                    unique_vals, counts = torch.unique(indices, return_counts=True)

                    for u, c in zip(unique_vals.tolist(), counts.tolist()):
                        freq = getattr(self, f"e_freq_{e}_{u}_{h}")
                        setattr(self, f"e_freq_{e}_{u}_{h}", freq + c)

    def save_old_prompts(self):
        print("Saving old prompts")
        # Save old prompts
        for e in self.e_layers:
            for l in range(self.num_experts):
                for h in range(self.num_heads):
                    pv_h = getattr(self, f"e_pv_{e}_{l}_{h}")
                    pv_h = pv_h.detach().clone()
                    setattr(self, f"old_e_pv_{e}_{l}_{h}", pv_h)

                    pk_h = getattr(self, f"e_pk_{e}_{l}_{h}")
                    pk_h = pk_h.detach().clone()
                    setattr(self, f"old_e_pk_{e}_{l}_{h}", pk_h)


class VQPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.soft_t = prompt_param[-1]
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence

            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)

            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])  # 10
        self.e_p_length = int(prompt_param[1])  # 8
        self.e_layers = [0, 1, 2, 3, 4]

        # qt loss weight
        self.vq_coef = 0.4
        self.comit_coef = 0.1

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self, f"e_k_{l}")  # 10, 768
            p = getattr(self, f"e_p_{l}")  # 10, 8, 768

            n_K = nn.functional.normalize(K, dim=1)  # f, 768
            cos_sim = torch.einsum("bd,kd->bk", x_querry, n_K)
            # alpha = torch.softmax(cos_sim, dim=1)
            alpha = torch.softmax(cos_sim / self.soft_t, dim=1)

            p_a = torch.einsum("bk,kld->bld", alpha, p)

            p_a_expended = p_a.unsqueeze(1)  # (bs, 1, l, d)
            dist = torch.pow(p_a_expended - p, 2)  # # (bs, 10, l, d)
            _, idxmin = dist.sum(-1).sum(-1).min(1)
            quantized = p.index_select(0, idxmin.view(-1))

            # calculate qt loss
            e_latent_loss = F.mse_loss(p_a, quantized.detach())  # vq_loss
            q_latent_loss = F.mse_loss(quantized, p_a.detach())  # commit_loss
            P_ = p_a + (quantized - p_a).detach()

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # calculate prompt related loss here;
            loss = self.vq_coef * e_latent_loss + self.comit_coef * q_latent_loss

        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


# @inproceedings{smith2023coda,
#   title={CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning},
#   author={Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={11909--11919},
#   year={2023}
# }
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)
            setattr(self, f"e_a_{e}", a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more
        # fair in the spirit of continual learning and has little affect on performance
        #
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f"e_k_{e}")
            A = getattr(self, f"e_a_{e}")
            P = getattr(self, f"e_p_{e}")
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)
            setattr(self, f"e_a_{e}", a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):
        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()  # clone trained prompt
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print("restarting!!!")
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self, f"e_k_{l}")  # 100, 768
            A = getattr(self, f"e_a_{l}")  # 100, 768
            p = getattr(self, f"e_p_{l}")  # 100, 8, 768
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)  # start idx for 100 component
            f = int((self.task_count + 1) * pt)  # final idx for 100 component

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum("bd,kd->bkd", x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)  # f, 768
            q = nn.functional.normalize(a_querry, dim=2)  # bs, f, 768
            aq_k = torch.einsum("bkd,kd->bk", q, n_K)  # bs, f (q k match)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum(
                "bk,kld->bld", aq_k, p
            )  # bs, 8, 768 reweighted p and sum along #component

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f"g_p_{g}", p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)

    def _init_smart(self, emb_d, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])  # self.n_tasks

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self, f"e_k_{l}")  # 0 based indexing here
            p = getattr(self, f"e_p_{l}")  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum("bj,kj->bk", q, n_K)

            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]

            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape(
                    (B, -1, self.emb_d)
                )  # L2P, needs reshape top-k prompts into one longer prompt
                Ev = P_[:, :, i:, :].reshape(
                    (B, -1, self.emb_d)
                )  # CODA-P avg several pre-defined task-specific components

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f"g_p_{l}")  # 0 based indexing here
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:  # impossible for default setting; no overlap in layers
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0, 1, 2, 3, 4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class ViTZoo(nn.Module):
    def __init__(
        self,
        num_classes=10,
        pt=False,
        prompt_flag=False,
        prompt_param=None,
        pretrained=None,
    ):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.pretrained = pretrained

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(
                img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                ckpt_layer=0,
                drop_path_rate=0,  # num_classes=21843
            )
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']; del load_dict['head.bias']
            # zoo_model.load_state_dict(load_dict)

            if self.pretrained == "sup21k":
                dict_path = (
                    "pretrained/vit_base_patch16_224_augreg_in21k.bin"  # with head
                )
                load_dict = torch.load(dict_path)
                del load_dict["head.weight"]
                del load_dict["head.bias"]
                zoo_model.load_state_dict(load_dict)
                print(f"Loading {self.pretrained} from {dict_path} ...")
            elif self.pretrained == "sup1k":
                dict_path = "pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin"  # with head
                load_dict = torch.load(dict_path)
                del load_dict["head.weight"]
                del load_dict["head.bias"]
                zoo_model.load_state_dict(load_dict)
                print(f"Loading {self.pretrained} from {dict_path} ...")
            elif self.pretrained == "ibot1k":
                dict_path = "pretrained/ibot_vit_base16.pth"  # ['state_dict']
                ckpt = torch.load(dict_path, map_location="cpu")[
                    "state_dict"
                ]  # with nead
                state_dict = zoo_model.state_dict()
                not_in_k = [k for k in ckpt.keys() if k not in state_dict.keys()]
                for k in not_in_k:
                    del ckpt[k]
                state_dict.update(ckpt)
                zoo_model.load_state_dict(state_dict)
                print(f"Loading {self.pretrained} from {dict_path} ...")
            elif self.pretrained == "dino1k":
                dict_path = "pretrained/dino_vitbase16_pretrain.pth"  # without head. blocks.0.att.qkv.weight
                load_dict = torch.load(dict_path, map_location="cpu")
                zoo_model.load_state_dict(load_dict)
                print(f"Loading {self.pretrained} from {dict_path} ...")
            else:
                print("Random Initialization")

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == "l2p":
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "dual":
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "coda":
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "qt":
            self.prompt = VQPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == "smope":
            self.prompt = OnePrompt(
                768, prompt_param[0], prompt_param[1], num_heads=12
            )
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

    # pen: get penultimate features
    def forward(
        self,
        x,
        pen=False,
        train=False,
        return_pre_logits=False,
        cls_mean=None,
        return_attn=False,
        dense=False,
    ):

        reduce_query = False

        if self.prompt is not None:  # if having a prompt module
            if self.prompt_flag == "smope":
                reduce_query = True
                if dense:
                    topk = -1
                else:
                    topk = self.prompt.topk
                q = None
            else:
                reduce_query = False
                with torch.no_grad():
                    q, _, _ = self.feat(x)
                    q = q[:, 0, :]
                topk = -1

            if return_attn:
                prompt_scores = self.feat(
                    x,
                    prompt=self.prompt,
                    q=q,
                    train=train,
                    task_id=self.task_id,
                    topk=topk,
                    return_attn=True,
                    reduce_query=reduce_query,
                )
                return prompt_scores

            out, prompt_loss, pre_logits = self.feat(
                x,
                prompt=self.prompt,
                q=q,
                train=train,
                task_id=self.task_id,
                topk=topk,
                reduce_query=reduce_query,
            )
            out = out[:, 0, :]  # bs,197,768 -> bs,768 cls_token
            pre_logits = pre_logits[:, 0, :]
        else:
            out, _, pre_logits = self.feat(x, reduce_query=reduce_query)
            out = out[:, 0, :]
            pre_logits = pre_logits[:, 0, :]
        out = out.view(out.size(0), -1)
        pre_logits = pre_logits.view(pre_logits.size(0), -1)

        if return_pre_logits:
            return out

        if not pen:
            out = self.last(out)

        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

    def forward_fc(self, x):
        # x = self.feat.norm(x)
        out = self.last(x)
        return out

    @torch.no_grad()
    def _load_weights(
        self, model: VisionTransformer, checkpoint_path: str, prefix: str = ""
    ):
        """Load weights from .npz checkpoints for official Google Brain Flax implementation"""
        import numpy as np
        from timm.models.helpers import (
            build_model_with_cfg,
            resolve_pretrained_cfg,
            named_apply,
            adapt_input_conv,
            checkpoint_seq,
        )

        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)

        w = np.load(checkpoint_path)
        if not prefix and "opt/target/embedding/kernel" in w:
            prefix = "opt/target/"

        if hasattr(model.patch_embed, "backbone"):
            # hybrid
            backbone = model.patch_embed.backbone
            stem_only = not hasattr(backbone, "stem")
            stem = backbone if stem_only else backbone.stem
            stem.conv.weight.copy_(
                adapt_input_conv(
                    stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
                )
            )
            stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
            stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
            if not stem_only:
                for i, stage in enumerate(backbone.stages):
                    for j, block in enumerate(stage.blocks):
                        bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                        for r in range(3):
                            getattr(block, f"conv{r + 1}").weight.copy_(
                                _n2p(w[f"{bp}conv{r + 1}/kernel"])
                            )
                            getattr(block, f"norm{r + 1}").weight.copy_(
                                _n2p(w[f"{bp}gn{r + 1}/scale"])
                            )
                            getattr(block, f"norm{r + 1}").bias.copy_(
                                _n2p(w[f"{bp}gn{r + 1}/bias"])
                            )
                        if block.downsample is not None:
                            block.downsample.conv.weight.copy_(
                                _n2p(w[f"{bp}conv_proj/kernel"])
                            )
                            block.downsample.norm.weight.copy_(
                                _n2p(w[f"{bp}gn_proj/scale"])
                            )
                            block.downsample.norm.bias.copy_(
                                _n2p(w[f"{bp}gn_proj/bias"])
                            )
            embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
        else:
            embed_conv_w = adapt_input_conv(
                model.patch_embed.proj.weight.shape[1],
                _n2p(w[f"{prefix}embedding/kernel"]),
            )
        model.patch_embed.proj.weight.copy_(embed_conv_w)
        model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
        model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
        pos_embed_w = _n2p(
            w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False
        )
        if pos_embed_w.shape != model.pos_embed.shape:
            pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                model.pos_embed,
                getattr(model, "num_prefix_tokens", 1),
                model.patch_embed.grid_size,
            )
        model.pos_embed.copy_(pos_embed_w)
        model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
        model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
        try:
            if (
                isinstance(model.head, nn.Linear)
                and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
            ):
                model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
                model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
        except:
            print("model does not contain head.")
        # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
        # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
        for i, block in enumerate(model.blocks.children()):
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
            block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
            block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
            block.attn.qkv.weight.copy_(
                torch.cat(
                    [
                        _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                        for n in ("query", "key", "value")
                    ]
                )
            )
            block.attn.qkv.bias.copy_(
                torch.cat(
                    [
                        _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                        for n in ("query", "key", "value")
                    ]
                )
            )
            block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
            for r in range(2):
                getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                    _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
                )
                getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                    _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
                )
            block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
            block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))

    def orth_loss(self, features, cls_mean, reg=0.01):
        if cls_mean:
            # orth loss of this batch
            sample_mean = []
            for k, v in cls_mean.items():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(
                features.device, non_blocking=True
            )
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(
                sim, torch.arange(0, sim.shape[0]).long().to(features.device)
            )
            # print(loss)
            return reg * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(
                sim, torch.arange(0, sim.shape[0]).long().to(features.device)
            )
            return reg * loss
            # return 0.


def vit_pt_imnet(
    out_dim, block_division=None, prompt_flag="None", prompt_param=None, pretrained=None
):
    return ViTZoo(
        num_classes=out_dim,
        pt=True,
        prompt_flag=prompt_flag,
        prompt_param=prompt_param,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    model = ViTZoo(pt=True)
