import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from torch_scatter import scatter_add
from torch_geometric.data import Data, Batch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch_geometric.utils import unbatch, to_dense_batch
from torch.nn.utils.rnn import pad_sequence
import math
from typing import Optional
from torch_geometric.nn import LayerNorm
from utils import LossRecord, set_up_logger, set_seed, set_device, get_dir_path, save_config
import logging
from tqdm import tqdm
from functools import partial
from time import time

config = {
    "model": {
        "model_name": "GDEA",
        "output_size": 3, # displacement x,y,z
        "input_size": 13, # Boundary condition + Spatial data + Material data
        "space_dim": 3,
        "n_layers": 4,
        "n_hidden": 128,
        "n_head": 1,
        "n_inner": 512,
        "mlp_layers": 4,
        "attn_type": "linear",
        "act": "gelu",
        "ffn_dropout": 0.0,
        "attn_dropout": 0.0,
        "horiz_fourier_dim": 0,
        "deq_epsilon": 1e-2, # Convergence threshold
        "deq_history_size": 5 # How many previous rcords will Anderson Acceleartion consier
    },
    "data": {
        "dataset": "DeformedABC",
        "data_path": "Your data path",
        "sample_factor": 1,
        "train_batchsize": 8,
        "eval_batchsize": 1,
        "normalize": True,
    },
    "train": {
        "random_seed": 0,
        "cuda": True,
        "device": 0,
        "epochs": 100,
        "patience": 100,
        "eval_freq": 1,
        "saving_best": True,
        "saving_checkpoint": False,
        "checkpoint_freq": 100
    },
    "optimize": {
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4
    },
    "schedule": {
        "scheduler": "ReduceLROnPlateau",
        "patience": 100,
        "gamma": 0.5
    },    
    "log": {
        "verbose": True,
        "log": True,
        "log_dir": "./logs",
        "wandb": False,
        "wandb_project": "DeformedABC_Project"
    }
}

class DeformedABC(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.root = os.path.join(root, f"data-{mode}")
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".npz")])
        self.transform = transform
        # Cach edge connectivity
        self.edge_cache = {}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        data = np.load(path)

        # Node data
        pos = torch.tensor(data["verts"], dtype=torch.float32)  # (N,3)
        sdf = torch.tensor(data["sdf"], dtype=torch.float32).unsqueeze(1)  # (N,1)
        disp = torch.tensor(data["disp"], dtype=torch.float32)  # (N,3) / Target data
        load = torch.tensor(data["load"], dtype=torch.float32)  # (N,3)
        dirichlet = torch.tensor(data["dirichlet"], dtype=torch.float32).unsqueeze(1)  # (N,1)
        norm_vec = torch.tensor(data["norm"], dtype=torch.float32)  # (N,3)
        youngs = torch.tensor(data["youngs"], dtype=torch.float32)  # (N,1)
        poisson = torch.tensor(data["poisson"], dtype=torch.float32)  # (N,1)
        elems = torch.tensor(data["elems"], dtype=torch.long)  # (M,4)

        inputs = torch.cat([pos, sdf, norm_vec, youngs, poisson, load, dirichlet,], dim=-1)

        # Checking the edge index cache
        if idx in self.edge_cache:
            edge_index = self.edge_cache[idx]
        else:
            edge_index = self.elem_edge_index(elems)
            self.edge_cache[idx] = edge_index

        data_pyg = Data(
            inputs = inputs,
            edge_index = edge_index,
            y = disp
        )

        return data_pyg
    
    def elem_edge_index(self, elems):
        device = elems.device
        num_elements, n_nodes_per_elem = elems.shape
        num_nodes = int(elems.max().item() + 1)

        # Generate all edge pairs
        pairs = torch.combinations(
            torch.arange(n_nodes_per_elem, device=device), r=2
        )

        edges = elems[:, pairs].reshape(-1, 2)
        edges, _ = torch.sort(edges, dim=1)

        encoded = edges[:, 0] * num_nodes + edges[:, 1]
        unique_encoded = torch.unique(encoded)
        
        src = unique_encoded // num_nodes
        dst = unique_encoded % num_nodes

        # Expand to both direction
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)

        return edge_index

ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU()}

class MLP(nn.Module):

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])

    def forward(self, x):
        x = self.act(self.linear_pre(x))
        
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x # Residual connection

        x = self.linear_post(x)
        return x

class MLP_SN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP_SN, self).__init__()
        
        self.act = ACTIVATION[act] if act in ACTIVATION.keys() else nn.GELU()
        
        # Constraining Lipschitz constant
        self.target_sn = 0.9
        
        # Applying spectrial normalization
        self.linear_pre = spectral_norm(nn.Linear(n_input, n_hidden), n_power_iterations=2)
        self.linear_post = spectral_norm(nn.Linear(n_hidden, n_output), n_power_iterations=2)
        
        self.linears = nn.ModuleList([
            spectral_norm(nn.Linear(n_hidden, n_hidden), n_power_iterations=2)
            for _ in range(n_layers)
        ])

        self.gain = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        
        # Pre-linear & Activation
        x = self.linear_pre(x)
        x = self.act(x)
        x = torch.tanh(x) * self.target_sn
        
        # Hidden Layers
        for layer in self.linears:
            x = layer(x)
            x = self.act(x)
            x = torch.tanh(x) * self.target_sn

        # Post-linear
        x = self.linear_post(x)
        
        x = torch.sigmoid(x) # Constraining range to (0, 1)
        
        safe_gain = torch.sigmoid(self.gain) * self.target_sn
        return x * safe_gain

class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        return self.x[item]

class GPTConfig():
    # base GPT config, params common to all GPT versions
    def __init__(self,attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0,attn_pdrop=0.0, 
                 n_embd=128, n_head=1, n_layer=3, block_size=128, n_inner=512, act='gelu', n_inputs=1, branch_size=0,
                 boundary_size=4, spatial_size=7,material_size=2,mlp_layers=4,deq_epsilon=0.01,deq_history_size=5):
        
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = 4 * self.n_embd
        self.act = act
        self.branch_size = branch_size
        self.n_inputs = n_inputs
        self.boundary_size = boundary_size
        self.spatial_size = spatial_size
        self.material_size = material_size
        self.mlp_layers = mlp_layers
        self.deq_epsilon = deq_epsilon
        self.deq_history_size = deq_history_size

class GlobalLinearAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, # Embedding dimension must be divisible by n_head

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Projections
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.keys  = nn.Linear(config.n_embd, config.n_embd)
        self.values = nn.Linear(config.n_embd, config.n_embd)
        self.proj  = nn.Linear(config.n_embd, config.n_embd)

        # Dropout
        self.attn_drop = nn.Dropout(getattr(config, "attn_pdrop", 0.0))

    def forward(self, x, y=None, mask_x=None, mask_y=None):

        y = x if y is None else y  # self-attention fallback
        B, N, C = x.size() # B: batch
        _, M, _ = y.size()

        # Multi-head projections
        Q = self.query(x).view(B, N, self.n_head, self.head_dim).transpose(1,2)  # [B,H,N,D]
        K = self.keys(y).view(B, M, self.n_head, self.head_dim).transpose(1,2)   # [B,H,M,D] 
        V = self.values(y).view(B, M, self.n_head, self.head_dim).transpose(1,2) # [B,H,M,D]

        # Prepare masks
        mask_x = mask_x.unsqueeze(1).unsqueeze(-1).float() if mask_x is not None else torch.ones(B,1,N,1, device=x.device)
        mask_y = mask_y.unsqueeze(1).unsqueeze(-1).float() if mask_y is not None else torch.ones(B,1,M,1, device=x.device)

        # Mask K and V: padding positions get zero contribution
        K = K * mask_y
        V = V * mask_y

        # Feature-wise positive transform
        Q = F.gelu(Q) + 1.0  # [B,H,N,D]
        K = F.gelu(K) + 1.0  # [B,H,M,D]

        # Linear attention context
        KV = torch.einsum('bhmd,bhme->bhde', K, V)  # [B,H,D,D]

        K_sum = K.sum(dim=2, keepdim=True)          # [B,H,1,D]
        denom = torch.sum(Q * K_sum, dim=-1, keepdim=True)  # [B,H,N,1]
        attn_out = torch.einsum('bhnd,bhde->bhne', Q, KV) / denom  # [B,H,N,D]
        attn_out = attn_out * mask_x  # apply query mask

        # Dropout and merge heads
        attn_out = self.attn_drop(attn_out)
        out = attn_out.transpose(1,2).reshape(B, N, C)
        out = self.proj(out) # [B,N,C]

        return out

class DEQ(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x_init, norm_K_diag, sparse_values, sparse_indices, sparse_size, aggregate_ln, batch, max_iter, epsilon, history_size, info_list=None, node_info_list=None, back_info_list=None):
        
        def f(z_in):
            K_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size).coalesce()
            out = (norm_K_diag * z_in) + torch.sparse.mm(K_sparse, z_in)
            return aggregate_ln(out, batch)

        m = history_size
        lam = 1e-1
        
        z = x_init.clone()
        z_hist, f_hist = [], []
        best_z, best_res = z.clone(), float('inf')
        
        num_graphs = int(batch.max().item() + 1)
        mesh_convergence_iters = torch.full((num_graphs,), max_iter, device=z.device, dtype=torch.int)
        already_converged = torch.zeros(num_graphs, dtype=torch.bool, device=z.device)

        with torch.no_grad():
            for i in range(max_iter):
                f_z = f(z)
                res = f_z - z
                
                res_sq = (res**2).sum(dim=-1) # (N,)
                from torch_scatter import scatter_add
                mesh_res_norm_sq = scatter_add(res_sq, batch, dim=0)
                mesh_fz_norm_sq = scatter_add((f_z**2).sum(dim=-1), batch, dim=0)
                
                # Per-mesh relative error
                mesh_rel_error = torch.sqrt(mesh_res_norm_sq) / (torch.sqrt(mesh_fz_norm_sq) + 1e-9)
                
                currently_converged = mesh_rel_error < epsilon
                newly_converged = currently_converged & ~already_converged
                mesh_convergence_iters[newly_converged] = i + 1
                already_converged |= currently_converged

                # break condition
                res_norm = torch.norm(res)
                rel_error = res_norm / (torch.norm(f_z) + 1e-9)
                
                if rel_error < best_res:
                    best_res, best_z = rel_error, z.clone()

                if already_converged.all():
                    z = f_z
                    break
                
                z_hist.append(z.reshape(-1)); f_hist.append(f_z.reshape(-1))
                if len(z_hist) > m: z_hist.pop(0); f_hist.pop(0)
                
                if len(z_hist) > 1 and i > 5:
                    Z, F = torch.stack(z_hist, dim=1), torch.stack(f_hist, dim=1)
                    G = F - Z
                    dG, dF = G[:, 1:] - G[:, :-1], F[:, 1:] - F[:, :-1]
                    H = torch.matmul(dG.T, dG) + lam * torch.eye(dG.shape[1], device=z.device)
                    try:
                        gamma = torch.linalg.solve(H, torch.matmul(dG.T, G[:, -1:]))
                        z_new = (F[:, -1:] - torch.matmul(dF, gamma)).view_as(z)
                        z = 0.5 * f_z + 0.5 * z_new if torch.isfinite(z_new).all() else f_z
                    except: z = f_z
                else:
                    z = f_z

        if info_list is not None:
            info_list.append(mesh_convergence_iters.cpu().numpy().tolist())

            if node_info_list is not None:
                _, counts = torch.unique(batch, return_counts=True)
                node_info_list.append(counts.cpu().numpy().tolist())

        ctx.back_info_list = back_info_list
        ctx.save_for_backward(z, norm_K_diag, sparse_values, sparse_indices)
        ctx.sparse_size, ctx.aggregate_ln, ctx.batch, ctx.max_iter, ctx.best_res = sparse_size, aggregate_ln, batch, max_iter, best_res
        ctx.epsilon_back = epsilon
        ctx.m_back = m
        
        return z
    
    @staticmethod
    def backward(ctx, grad_output):
        z_star, norm_K_diag, sparse_values, sparse_indices = ctx.saved_tensors

        epsilon = ctx.epsilon_back
        m_back = ctx.m_back

        if ctx.best_res > 1.0:
            return None, torch.zeros_like(norm_K_diag), torch.zeros_like(sparse_values), \
                   None, None, None, None, None, None, None, None, None
        
        z_star = z_star.detach().requires_grad_(True)
        sparse_values = sparse_values.detach().requires_grad_(True)
        norm_K_diag = norm_K_diag.detach().requires_grad_(True)

        with torch.enable_grad():
            K_temp = torch.sparse_coo_tensor(sparse_indices, sparse_values, ctx.sparse_size).coalesce()
            f_star = (norm_K_diag * z_star) + torch.sparse.mm(K_temp, z_star)
            f_star = ctx.aggregate_ln(f_star, ctx.batch)

        def v_mapping(v_in):
            # Jacobian-Vector Product (JVP)
            v_jvp = torch.autograd.grad(f_star, z_star, v_in, retain_graph=True, allow_unused=True)[0]
            if v_jvp is None: v_jvp = torch.zeros_like(v_in)
            
            return grad_output + v_jvp * 0.50
        
        v = grad_output.clone()
        v_hist, fv_hist = [], []
        best_v, best_res_norm = v.clone(), float('inf')
        
        lam_back = 1e-1 
        
        num_graphs = int(ctx.batch.max().item() + 1)
        mesh_convergence_iters_bwd = torch.full((num_graphs,), ctx.max_iter, device=v.device, dtype=torch.int)
        already_converged_bwd = torch.zeros(num_graphs, dtype=torch.bool, device=v.device)
        
        for i in range(ctx.max_iter):
            fv = v_mapping(v)
            
            if not torch.isfinite(fv).all(): 
                v = best_v
                break
            
            res = fv - v 
            
            from torch_scatter import scatter_add
            res_sq = (res**2).sum(dim=-1) # (N,)
            mesh_res_norm_sq = scatter_add(res_sq, ctx.batch, dim=0)
            mesh_fv_norm_sq = scatter_add((fv**2).sum(dim=-1), ctx.batch, dim=0)
            
            # Relative error per-mesh
            mesh_rel_error = torch.sqrt(mesh_res_norm_sq) / (torch.sqrt(mesh_fv_norm_sq) + 1e-9)
            
            currently_converged = mesh_rel_error < epsilon
            newly_converged = currently_converged & ~already_converged_bwd
            mesh_convergence_iters_bwd[newly_converged] = i + 1
            already_converged_bwd |= currently_converged

            res_norm = torch.norm(res)
            rel_error = res_norm / (torch.norm(fv) + 1e-9)

            if already_converged_bwd.all():
                v = fv
                break

            if res_norm < best_res_norm:
                best_res_norm, best_v = res_norm, v.clone()

            v_hist.append(v.reshape(-1)); fv_hist.append(fv.reshape(-1))
            if len(v_hist) > m_back: v_hist.pop(0); fv_hist.pop(0)

            if len(v_hist) > 1: 
                V, FV = torch.stack(v_hist, dim=1), torch.stack(fv_hist, dim=1)
                G = FV - V 
                dG, dFV = G[:, 1:] - G[:, :-1], FV[:, 1:] - FV[:, :-1] 
                H = torch.matmul(dG.T, dG) + lam_back * torch.eye(dG.shape[1], device=v.device) 
                try: 
                    gamma = torch.linalg.solve(H, torch.matmul(dG.T, G[:, -1:]))
                    v_anderson = (FV[:, -1:] - torch.matmul(dFV - dG, gamma)).view_as(v)
                    v = 0.5 * fv + 0.5 * v_anderson if torch.isfinite(v_anderson).all() else fv
                except: 
                    v = 0.5 * v + 0.5 * fv
            else:
                v = 0.5 * v + 0.5 * fv
        
        if hasattr(ctx, 'back_info_list') and ctx.back_info_list is not None:
            ctx.back_info_list.append(mesh_convergence_iters_bwd.cpu().numpy().tolist())

        grad_K_diag = torch.autograd.grad(f_star, norm_K_diag, v, retain_graph=True, allow_unused=True)[0]
        grad_values = torch.autograd.grad(f_star, sparse_values, v, allow_unused=True)[0]
        
        return (
            None,           # 1. x_init
            grad_K_diag,    # 2. norm_K_diag
            grad_values,    # 3. sparse_values
            None,           # 4. sparse_indices
            None,           # 5. sparse_size
            None,           # 6. aggregate_ln
            None,           # 7. batch
            None,           # 8. max_iter
            None,           # 9. epsilon
            None,           # 10. history_size
            None,           # 11. info_list
            None,           # 12. node_info_list
            None            # 13. back_info_list
        )

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_width = config.n_embd
        self.act = nn.GELU

        self.deq_epsilon = config.deq_epsilon
        self.deq_history_size = config.deq_history_size
        
        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)
        self.resid_drop3 = nn.Dropout(config.resid_pdrop)

        self.selfattn = GlobalLinearAttention(config)
        
        self.ln1 = nn.LayerNorm(n_width)
        self.ln2 = nn.LayerNorm(n_width)
        self.ln3 = nn.LayerNorm(n_width)
        self.ln4 = nn.LayerNorm(n_width)

        self.mlp1 = nn.Sequential(
            nn.Linear(n_width, n_width*4),
            self.act(),
            nn.Linear(n_width*4, n_width),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(n_width, n_width*4),
            self.act(),
            nn.Linear(n_width*4, n_width),
        )

        self.edge_mlp = MLP_SN(n_width*2, n_width, 3, n_layers=config.mlp_layers, act=config.act)
        self.conv_history = [] 
        self.back_history = []
        self.node_info_list = []
        
    
    def aggregate(self, x, edge_index, batch, epsilon, history_size, max_iter=100):
        
        num_nodes = x.size(0)
        row, col = edge_index
        aggregate_ln = LayerNorm(x.size(-1), affine=False)

        # Predict edge-wise weight matrix
        edge_feat = torch.cat([x[row], x[col]], dim=-1)
        predicted_weights = self.edge_mlp(edge_feat)
        k_ii, k_jj, k_ij = predicted_weights[:, 0], predicted_weights[:, 1], predicted_weights[:, 2]

        # Matrix assembly
        K_diag_raw = scatter_add(k_ii, row, dim=0, dim_size=num_nodes) + scatter_add(k_jj, col, dim=0, dim_size=num_nodes)
        row_sum_off_diag = scatter_add(k_ij, row, dim=0, dim_size=num_nodes)
        total_row_sum = K_diag_raw + row_sum_off_diag + 1e-6

        # Noramlize rows
        shrink_factor = 0.9
        norm_K_diag = (K_diag_raw / total_row_sum * shrink_factor).unsqueeze(-1)
        norm_k_ij = (k_ij / total_row_sum[row]) * shrink_factor

        sparse_size = (num_nodes, num_nodes)

        x_star = DEQ.apply(
            x, norm_K_diag, norm_k_ij, edge_index, sparse_size, 
            aggregate_ln, batch, max_iter, epsilon, history_size, self.conv_history, self.node_info_list, self.back_history
        )
        
        return x_star

    def forward(self, x, batch, edge_index):

        x = x + self.resid_drop1(self.aggregate(self.ln1(x), edge_index, batch, epsilon=self.deq_epsilon, history_size=self.deq_history_size))
        x = x + self. mlp1(self.ln2(x))

        x_padded, mask_bool = to_dense_batch(x, batch=batch)
        mask = mask_bool.float() 
        
        res_input = x_padded.clone() 
        
        attn_input = self.ln3(x_padded)
        attn_output = self.selfattn(x=attn_input, y=attn_input, mask_x=mask, mask_y=mask)
        
        x_padded = res_input + self.resid_drop2(attn_output)
        
        x_padded = x_padded + self.resid_drop3(self.mlp2(self.ln4(x_padded)))
        
        x = x_padded[mask_bool]

        return x

class GDEA(nn.Module): # Graph Neural Operator Transformer
    def __init__(self,
                 input_size = 13,
                 space_dim=3,
                 output_size=3,
                 n_layers=2,
                 n_hidden=64,
                 n_head=1,
                 n_inner = 4,
                 mlp_layers=2,
                 attn_type='linear',
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 deq_epsilon = 1e-2,
                 deq_history_size = 5
                 ):
        super(GDEA, self).__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        self.input_size = input_size
        self.output_size = output_size
        self.space_dim = space_dim
        
        # configure values for transformer structures.
        self.gpt_config = GPTConfig(attn_type=attn_type,
                                    embd_pdrop=ffn_dropout, 
                                    resid_pdrop=ffn_dropout, 
                                    attn_pdrop=attn_dropout,
                                    n_embd=n_hidden, #256
                                    n_head=n_head, 
                                    n_layer=n_layers,
                                    block_size=n_hidden, # context window (token)
                                    act=act, 
                                    branch_size=0, # 0 here because we are using self-attention.
                                    n_inputs=1, # 1 here because we don't separate attentions.
                                    n_inner=n_inner,
                                    mlp_layers = mlp_layers,
                                    deq_epsilon = deq_epsilon,
                                    deq_history_size = deq_history_size)
        
        self.project_in = MLP(self.input_size, n_hidden, n_hidden, n_layers=mlp_layers, act=act)
        self.project_out = MLP(n_hidden, n_hidden, self.output_size, n_layers=mlp_layers, act=act)

        self.Blocks = nn.Sequential(*[Block(self.gpt_config) for _ in range(self.gpt_config.n_layer)])
    
    def forward(self, g):

        x = g.inputs
        x = self.project_in(x)

        for block in self.Blocks:
            x = block(x, g.batch, g.edge_index)

        out = self.project_out(x)

        return out

class BaseTrainer:
    def __init__(self, model_name, device, epochs, eval_freq=5, patience=-1,
                 verbose=False, wandb_log=False, logger=False, saving_best=True, 
                 saving_checkpoint=False, checkpoint_freq=100, saving_path=None):
        self.model_name = model_name
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.patience = patience
        self.wandb = wandb_log
        self.verbose = verbose
        self.saving_best = saving_best
        self.saving_checkpoint = saving_checkpoint
        self.checkpoint_freq = checkpoint_freq
        self.saving_path = saving_path
        if verbose:
            self.logger = logging.info if logger else print
    
    def get_initializer(self, name):
        if name is None: return None
        if name == 'xavier_normal': init_ = partial(torch.nn.init.xavier_normal_)
        elif name == 'kaiming_uniform': init_ = partial(torch.nn.init.kaiming_uniform_)
        elif name == 'kaiming_normal': init_ = partial(torch.nn.init.kaiming_normal_)
        return init_
    
    def build_optimizer(self, model, args, **kwargs):
        opt_args = args['optimize']
        if opt_args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt_args['lr'], weight_decay=opt_args['weight_decay'])
        elif opt_args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt_args['lr'], momentum=opt_args.get('momentum', 0.9), weight_decay=opt_args['weight_decay'])
        elif opt_args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt_args['lr'], weight_decay=opt_args['weight_decay'])
        else: raise NotImplementedError(f"Optimizer {opt_args['optimizer']} not implemented")
        return optimizer

    def build_scheduler(self, optimizer, args, **kwargs):
        sched_args = args['schedule']
        if sched_args['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched_args['milestones'], gamma=sched_args['gamma'])
        elif sched_args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=sched_args['lr'], div_factor=sched_args['div_factor'], 
                                                           final_div_factor=sched_args['final_div_factor'], pct_start=sched_args['pct_start'], 
                                                           steps_per_epoch=sched_args['steps_per_epoch'], epochs=args['train']['epochs'])
        elif sched_args['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_args['step_size'], gamma=sched_args['gamma'])
        elif sched_args['scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sched_args['gamma'], patience=sched_args['patience'])
        else: raise NotImplementedError(f"Scheduler {sched_args['scheduler']} not implemented")
        return scheduler
    
    def build_model(self, **kwargs): raise NotImplementedError
    
    def process(self, model, train_loader, valid_loader, test_loader, optimizer, criterion, regularizer=None, scheduler=None, **kwargs):
        all_losses = []
        val_losses = []
        all_train_iters = []
        all_val_iters = []
        all_train_back_iters = [] # Backward (Train)

        # List to collect node size
        all_train_node_sizes = []
        all_val_node_sizes = []
        
        # Defining saving path
        non_log_path = 'Your non log path'
        os.makedirs(non_log_path, exist_ok=True) # Generate path if not exist
        save_filename = os.path.join(non_log_path, "mdl-GDEA-losses.npz")

        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
            self.logger("Test dataset size: {}".format(len(test_loader.dataset)))

        best_epoch = 0
        best_metrics = None
        counter = 0
        
        with tqdm(total=self.epochs) as bar:
            for epoch in range(self.epochs):

                global CURRENT_EPOCH
                CURRENT_EPOCH = epoch

                torch.cuda.empty_cache()
                
                # Train
                train_loss_record, train_f_iters, train_b_iters, train_nodes = self.train(model=model, train_loader=train_loader, optimizer=optimizer,
                                                        criterion=criterion, scheduler=scheduler, regularizer=regularizer,
                                                        accumulation_steps=8, **kwargs)
                
                all_losses.append(train_loss_record.to_dict()["train_loss"])
                all_train_iters.append(train_f_iters)
                all_train_back_iters.append(train_b_iters)
                all_train_node_sizes.append(train_nodes)

                if self.verbose:
                    self.logger("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
                
                # Validation
                if (epoch + 1) % self.eval_freq == 0:
                    valid_loss_record, val_iters, val_nodes = self.evaluate(model, valid_loader, criterion, split="valid", **kwargs)
                    val_losses.append(valid_loss_record.to_dict()["valid_loss"])
                    all_val_iters.append(val_iters)
                    all_val_node_sizes.append(val_nodes)

                    if self.verbose: self.logger("Epoch {} | {}".format(epoch, valid_loss_record))
                    valid_metrics = valid_loss_record.to_dict()
                    
                    # Save in .npz
                    np.savez(
                        save_filename,
                        all_losses=np.array(all_losses),
                        val_losses=np.array(val_losses),
                        
                        all_train_iters=np.array(all_train_iters, dtype=object),
                        all_val_iters=np.array(all_val_iters, dtype=object),

                        all_train_back_iters=np.array(all_train_back_iters, dtype=object),
                        
                        all_train_node_sizes=np.array(all_train_node_sizes, dtype=object),
                        all_val_node_sizes=np.array(all_val_node_sizes, dtype=object)
                    )
                    
                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): 
                            scheduler.step(valid_metrics["valid_loss"])
                        else: 
                            scheduler.step()

                    if not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']:
                        counter = 0
                        best_epoch = epoch
                        best_metrics = valid_metrics
                        torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                        model.cuda()
                        if self.verbose: self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                    elif self.patience != -1:
                        counter += 1
                        if counter >= self.patience:
                            if self.verbose: self.logger("Early stop at epoch {}".format(epoch))
                            break
                
                bar.update(1)

        self.logger("Optimization Finished!")
        torch.save(model.state_dict(), os.path.join(non_log_path, "mdl-GDEA.pth"))

        if not best_metrics: torch.save(model.state_dict(), os.path.join(non_log_path, "mdl-GDEA.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        valid_loss_record, _, _ = self.evaluate(model, valid_loader, criterion, split="valid", **kwargs)
        self.logger("Valid metrics: {}".format(valid_loss_record))
        test_loss_record, _, _ = self.evaluate(model, test_loader, criterion, split="test", **kwargs)
        self.logger("Test metrics: {}".format(test_loss_record))

    def train(self, model, train_loader, optimizer, criterion, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss"])
        
        epoch_fwd_iters = []
        epoch_bwd_iters = []
        epoch_node_sizes = []
        
        model.cuda(); model.train()
        for (x, y) in train_loader:

            current_fwd = []
            current_bwd = []
            current_nodes = []
            self._set_history_list(model, current_fwd, current_bwd, current_nodes) # 이 함수가 중요!

            x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x).reshape(y.shape)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            loss_record.update({"train_loss": loss.sum().item()}, n=y_pred.shape[0])
            
            epoch_fwd_iters.extend(current_fwd)
            epoch_bwd_iters.extend(current_bwd)
            epoch_node_sizes.extend(current_nodes)

        return loss_record, epoch_fwd_iters, epoch_bwd_iters, epoch_node_sizes
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split)])
        
        epoch_fwd_iters = []
        epoch_node_sizes = []
        
        model.eval()
        with torch.no_grad():
            for (x, y) in eval_loader:
                current_fwd = []
                current_nodes = []
                self._set_history_list(model, current_fwd, None, current_nodes)

                x, y = x.to('cuda'), y.to('cuda')
                y_pred = model(x).reshape(y.shape)
                loss = criterion(y_pred, y)
                
                loss_record.update({"{}_loss".format(split): loss.sum().item()}, n=y_pred.shape[0])
                
                epoch_fwd_iters.extend(current_fwd)
                epoch_node_sizes.extend(current_nodes)
                
        return loss_record, epoch_fwd_iters, epoch_node_sizes

class GraphBaseTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(
            model_name=args['model']['model_name'], device=args['train']['device'], epochs=args['train']['epochs'],
            eval_freq=args['train']['eval_freq'], patience=args['train']['patience'], verbose=args['log']['verbose'],
            wandb_log=args['log']['wandb'], logger=args['log']['log'], saving_best=args['train']['saving_best'],
            saving_checkpoint=args['train']['saving_checkpoint'], checkpoint_freq=args['train']['checkpoint_freq'],
            saving_path=args['saving_path']
        )
    
    def _set_history_list(self, model, fwd_list, bwd_list, node_list):

        for module in model.modules():

            if module.__class__.__name__ == 'Block':
                module.conv_history = fwd_list
                module.back_history = bwd_list
                module.node_info_list = node_list

    def train(self, model, train_loader, optimizer, criterion, accumulation_steps=8, scheduler=None, loss_list=None, **kwargs):
        
        loss_record = LossRecord(["train_loss"])
        epoch_fwd_iters = []
        epoch_bwd_iters = [] 
        epoch_node_sizes = [] 
        
        model.cuda()
        model.train()
        optimizer.zero_grad()

        with tqdm(train_loader, desc="Training", leave=False) as tbar:
            for step, graph in enumerate(tbar):
                graph = graph.to("cuda")
                
                current_fwd_history = []
                current_bwd_history = []
                current_nodes = []

                self._set_history_list(model, current_fwd_history, current_bwd_history, current_nodes)
                
                # Forward pass: Do fixed point iteration of DEQ
                y_pred = model(graph).reshape(graph.y.shape)
                train_loss, _ = self.loss(y_pred, graph.y, criterion, batch=graph.batch, graph=graph)
                
                # Backward Pass: Do DEQ's backward solver
                (train_loss / accumulation_steps).backward()

                # Gradient Accumulation
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                for item in current_fwd_history:
                    if isinstance(item, list):
                        epoch_fwd_iters.extend(item)

                    else:
                        epoch_fwd_iters.append(item)
                
                for item in current_bwd_history:
                    if isinstance(item, list):
                        epoch_bwd_iters.extend(item)
                    else:
                        epoch_bwd_iters.append(item)

                # Update records
                loss_record.update({"train_loss": train_loss.item()}, n=graph.y.shape[0])
                tbar.set_postfix(recorded=len(epoch_fwd_iters))
                
        return loss_record, epoch_fwd_iters, epoch_bwd_iters, epoch_node_sizes
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        
        loss_record = LossRecord([f"{split}_loss"])
        epoch_fwd_iters = []
        epoch_node_sizes = []
        model.eval()
        
        with torch.no_grad():
            for graph in eval_loader:
                graph = graph.to('cuda')
                
                current_batch_history = []
                current_nodes = []
                
                self._set_history_list(model, current_batch_history, None, current_nodes)

                y_pred = model(graph).reshape(graph.y.shape)
                
                for item in current_batch_history:
                    if isinstance(item, list):
                        epoch_fwd_iters.extend(item)
                        epoch_node_sizes.extend(current_nodes[0] if isinstance(current_nodes[0], list) else current_nodes)
                    else:
                        epoch_fwd_iters.append(item)
                        epoch_node_sizes.extend(current_nodes)
                
                eval_loss, _ = self.loss(y_pred, graph.y, criterion, batch=graph.batch, graph=graph)
                loss_record.update({f"{split}_loss": eval_loss.item()}, n=graph.y.shape[0])
                
        return loss_record, epoch_fwd_iters, epoch_node_sizes

    def loss(self, y_pred, y, criterion, batch=None, loss_list=None, **kwargs):
        loss_dict = {}
        if batch is None: norm = criterion(y_pred, y)
        else:
            num_graphs = batch.max().item() + 1
            norm = torch.zeros(1, device=y.device)
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.sum() == 0: continue
                norm += criterion(y_pred[mask].unsqueeze(0), y[mask].unsqueeze(0))
        return norm, loss_dict

# Add build_model function to GraphBaseTrainer
class GDEATrainer(GraphBaseTrainer):
    def __init__(self, args):
        
        super().__init__(args)
        
    def build_model(self, args, **kwargs):
        model = GDEA(
            input_size = args['model']['input_size'],
            space_dim=args['model']['space_dim'],
            output_size=args['model']['output_size'],
            n_layers=args['model']['n_layers'],
            n_hidden=args['model']['n_hidden'],
            n_head=args['model']['n_head'],
            n_inner=args['model']['n_inner'],
            mlp_layers=args['model']['mlp_layers'],
            attn_type=args['model']['attn_type'],
            act=args['model']['act'],
            ffn_dropout=args['model']['ffn_dropout'],
            attn_dropout=args['model']['attn_dropout'],
            horiz_fourier_dim=args['model']['horiz_fourier_dim'],
            deq_epsilon=args['model']['deq_epsilon'],
            deq_history_size=args['model']['deq_history_size'],
        )
        return model

TRAINER_DICT = {
    'GDEA': GDEATrainer,
    'gdea': GDEATrainer,
}

def is_valid_sample(data):

    try:
        for key, value in data:
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    return False
            elif isinstance(value, np.ndarray):
                if not np.isfinite(value).all():
                    return False
        return True
    except Exception as e:
        print(f"[Error] sample 검사 중 오류: {e}")
        return False

def filter_dataset(dataset, desc="Filtering dataset"):

    valid_indices = []
    for idx in tqdm(range(len(dataset)), desc=desc):
        try:
            sample = dataset[idx]
            if is_valid_sample(sample):
                valid_indices.append(idx)
        except Exception as e:
            print(f"[Skip] idx={idx}, error={e}")
            continue

    print(f" {len(valid_indices)} / {len(dataset)} samples valid")
    return torch.utils.data.Subset(dataset, valid_indices)

def DeformedABC_procedure(args):
    # Checking model
    if args['model']['model_name'] not in TRAINER_DICT.keys():
        raise NotImplementedError(f"Model {args['model']['model_name']} not implemented")

    # Setting logger
    logger = logging.info if args['log']['log'] else print

    # Load data
    logger(f"Loading {args['data']['dataset']} dataset")
    start = time()

    train_dataset = DeformedABC(
        root=args['data']['data_path'],
        mode="train",
        transform=None
    )

    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=args['data']['train_batchsize'],
        shuffle=True
    )

    test_dataset = DeformedABC(
        root=args['data']['data_path'],
        mode="test",
        transform=None
    )

    test_loader = GeoDataLoader(
        test_dataset,
        batch_size=args['data']['eval_batchsize'],
        shuffle=False
    )

    logger(f"Loading data costs {time() - start:.2f}s")

    # Build model
    logger("Building model")
    start = time()
    trainer = TRAINER_DICT[args['model']['model_name']](args)
    model = trainer.build_model(args).to(args['train']['device'])
    optimizer = trainer.build_optimizer(model, args)
    scheduler = trainer.build_scheduler(optimizer, args)
    criterion = torch.nn.MSELoss()

    logger(f"Model: {model}")
    logger(f"Criterion: {criterion}")
    logger(f"Optimizer: {optimizer}")
    logger(f"Scheduler: {scheduler}")
    logger(f"Building model costs {time() - start:.2f}s")

    trainer.process(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        valid_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        loss_list=['disp']
    )

def main():
    torch.cuda.empty_cache()
    # Load configuration defined in the top of this code
    args = config.copy()  # config는 이미 위에서 정의된 dict
    
    # Set up logging and directories
    if args['log']['log'] is True:
        saving_path, saving_name = set_up_logger(
            args["model"]["model_name"],
            args["data"]["dataset"],
            args["log"]["log_dir"]
        )
    elif args["train"].get('save_best', False) or args["train"].get('save_check_points', False):
        saving_path, saving_name = get_dir_path(
            args["model"]["model_name"],
            args["data"]["dataset"],
            args["log"]["log_dir"]
        )
    else:
        saving_path, saving_name = None, None

    args['saving_path'] = saving_path
    args['saving_name'] = saving_name

    save_config(args, saving_path)
    
    # Set device and random seeds
    set_device(args["train"]["cuda"], args["train"]["device"])
    set_seed(args["train"]["random_seed"])

    # Run dataset-specific procedure
    if args["data"]["dataset"] == "DeformedABC":
        DeformedABC_procedure(args)
    else:
        raise NotImplementedError(f"Dataset {args['data']['dataset']} is not implemented")

if __name__ == "__main__":
    main()
