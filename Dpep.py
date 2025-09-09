import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.device_mesh import init_device_mesh
import os


class MoELayer(nn.Module):
    """MoE layer with Expert Parallel - each device handles subset of experts"""
    def __init__(self, hidden_size, num_experts, expert_size, expert_parallel_group=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.expert_parallel_group = expert_parallel_group
        
        # Get expert parallel info
        if expert_parallel_group is not None:
            self.ep_rank = dist.get_rank(expert_parallel_group)
            self.ep_size = dist.get_world_size(expert_parallel_group)
        else:
            self.ep_rank = 0
            self.ep_size = 1
            
        # Calculate which experts this device handles
        experts_per_device = num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * experts_per_device
        self.expert_end_idx = (self.ep_rank + 1) * experts_per_device
        self.local_num_experts = experts_per_device
        self.total_num_experts = num_experts
        
        # Router/gating network (replicated on all devices)
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Only create experts for this device's shard
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_size),
                nn.ReLU(),
                nn.Linear(expert_size, hidden_size)
            ) for _ in range(self.local_num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Gate computation (same on all devices)
        gate_logits = self.gate(x_flat)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        expert_indices = torch.argmax(gate_probs, dim=-1)
        
        # Initialize output
        local_output = torch.zeros_like(x_flat)
        
        # Process only local experts
        for local_expert_idx in range(self.local_num_experts):
            global_expert_idx = self.expert_start_idx + local_expert_idx
            expert_mask = (expert_indices == global_expert_idx)
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[local_expert_idx](expert_input)
                local_output[expert_mask] = expert_output
        
        # All-reduce across expert parallel group to combine outputs
        if self.expert_parallel_group is not None:
            dist.all_reduce(local_output, group=self.expert_parallel_group)
        
        return local_output.view(batch_size, seq_len, hidden_size)


class AttentionLayer(nn.Module):
    """Simple attention layer for demonstration"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Simple attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MoE"""
    def __init__(self, hidden_size, num_heads, num_experts, expert_size, expert_parallel_group=None):
        super().__init__()
        self.attention = AttentionLayer(hidden_size, num_heads)
        self.moe = MoELayer(hidden_size, num_experts, expert_size, expert_parallel_group)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out
        
        # MoE with residual connection  
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x


class MoETransformer(nn.Module):
    """Simple MoE Transformer model"""
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_experts, expert_size, expert_parallel_group=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_experts, expert_size, expert_parallel_group)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.lm_head(x)


def setup_hsdp2_with_dp_ep():
    """
    Setup HSDP2 with:
    - DP (Data Parallel) for attention layers
    - EP (Expert Parallel) for MoE layers  
    - ZeRO stage 1 for gradient sharing within replica groups
    """
    
    # Get distributed training info
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Initialize distributed training
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Setup device mesh for hybrid parallelism
    # Assume we have 8 GPUs: 2 replica groups of 4 GPUs each
    replica_group_size = 4  # ZeRO replicas within each group
    expert_parallel_size = 2  # Expert parallel across groups
    
    assert world_size == replica_group_size * expert_parallel_size, \
        f"World size {world_size} must equal replica_group_size * expert_parallel_size = {replica_group_size * expert_parallel_size}"
    
    # Create 2D device mesh: [replica_group, expert_parallel]
    device_mesh = init_device_mesh(
        "cuda", 
        (replica_group_size, expert_parallel_size),
        mesh_dim_names=("replica", "expert")
    )
    
    # Get process groups
    replica_group = device_mesh.get_group("replica")
    expert_parallel_group = device_mesh.get_group("expert")
    
    print(f"Rank {rank}: replica_group={replica_group}, expert_parallel_group={expert_parallel_group}")
    
    # Model configuration
    model_config = {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "num_experts": 8,
        "expert_size": 2048,
        "expert_parallel_group": expert_parallel_group
    }
    
    # Create model
    model = MoETransformer(**model_config).to(device)
    
    # Setup FSDP with ZeRO stage 1 for attention layers (DP within replica groups)
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.NO_SHARD,  # No parameter sharding - ZeRO stage 1
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        "process_group": replica_group,  # Use replica group for FSDP
        "use_orig_params": True
    }
    
    # Apply FSDP to attention layers for gradient synchronization within replica groups
    for layer in model.layers:
        layer.attention = FSDP(layer.attention, **fsdp_config)
    
    # Apply FSDP to embedding and output layers
    model.embedding = FSDP(model.embedding, **fsdp_config)
    model.lm_head = FSDP(model.lm_head, **fsdp_config)
    
    # Wrap the entire model
    model = FSDP(model, **fsdp_config)
    
    return model, device_mesh, replica_group, expert_parallel_group


def train_step(model, batch, optimizer):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def main():
    """Main training loop"""
    
    # Setup model and distributed training
    model, device_mesh, replica_group, expert_parallel_group = setup_hsdp2_with_dp_ep()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy data for demonstration
    batch_size = 4
    seq_length = 512
    vocab_size = 32000
    
    device = next(model.parameters()).device
    rank = dist.get_rank()
    
    # Training loop
    for step in range(100):
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        }
        
        # Training step
        loss = train_step(model, batch, optimizer)
        
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


# Launch command:
# torchrun --nproc_per_node=8 --nnodes=1 hsdp2_example.py
#
# For multi-node:
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 hsdp2_example.py
