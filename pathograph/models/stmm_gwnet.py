"""
ST-MM-GNN Layer A: GraphWaveNet-style model for pathogen status prediction.

This module implements the core architecture combining:
- Node feature construction from trade/climate multimodal inputs
- Dilated temporal convolution with causal padding
- Static graph diffusion convolution
- Output head for next-month pathogen status logits
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGraphConv(nn.Module):
    """Graph diffusion convolution with static supports."""
    
    def __init__(self, in_channels: int, out_channels: int, supports: List[torch.Tensor], diffusion_K: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.supports = supports  # List of (N, N) adjacency matrices
        self.diffusion_K = diffusion_K
        
        # Linear layer for each (support, power) combination
        # +1 for identity (k=0)
        num_matrices = len(supports) * (diffusion_K + 1)
        self.mlp = nn.Linear(in_channels * num_matrices, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, N, T)
        Returns:
            out: (B, C_out, N, T)
        """
        B, C, N, T = x.shape
        
        # Collect diffusion features
        diffusion_features = []
        
        for support in self.supports:
            support = support.to(x.device)  # Ensure same device
            
            # k=0: identity
            diffusion_features.append(x)
            
            # k=1..K: powers of support
            x_k = x
            for k in range(1, self.diffusion_K + 1):
                # Apply support via einsum: (N,N) @ (B,C,N,T) -> (B,C,N,T)
                x_k = torch.einsum('nm,bcmt->bcnt', support, x_k)
                diffusion_features.append(x_k)
        
        # Concatenate all features: (B, C_in * num_matrices, N, T)
        x_concat = torch.cat(diffusion_features, dim=1)
        
        # Permute to (B, N, T, C_in * num_matrices) for linear layer
        x_concat = x_concat.permute(0, 2, 3, 1)
        
        # Apply MLP: (B, N, T, C_out)
        out = self.mlp(x_concat)
        
        # Permute back to (B, C_out, N, T)
        out = out.permute(0, 3, 1, 2)
        
        return out


class GatedTCNBlock(nn.Module):
    """Gated temporal convolution block with causal padding and skip connection."""
    
    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Causal padding amount: (kernel_size - 1) * dilation
        self.causal_padding = (kernel_size - 1) * dilation
        
        # Filter and gate convolutions
        self.filter_conv = nn.Conv2d(
            residual_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        
        self.gate_conv = nn.Conv2d(
            residual_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        
        # Skip and residual projections
        self.skip_conv = nn.Conv2d(dilation_channels, skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv2d(dilation_channels, residual_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, N, T)
        Returns:
            residual_out: (B, C, N, T)
            skip_out: (B, skip_channels, N, T)
        """
        # Apply causal padding on the time dimension (last dim)
        x_padded = F.pad(x, (self.causal_padding, 0), mode='constant', value=0)
        
        # Gated activation: tanh(filter) * sigmoid(gate)
        filter_out = torch.tanh(self.filter_conv(x_padded))
        gate_out = torch.sigmoid(self.gate_conv(x_padded))
        z = filter_out * gate_out  # CORRECTED: multiplication, not addition
        
        z = self.dropout(z)
        
        # Skip connection output
        skip_out = self.skip_conv(z)
        
        # Residual connection output
        residual_out = self.residual_conv(z)
        residual_out = residual_out + x  # Residual connection
        
        return residual_out, skip_out


class STMMGraphWaveNet(nn.Module):
    """
    ST-MM-GNN Layer A: GraphWaveNet/MTGNN-style model for pathogen status prediction.
    
    Architecture:
    1. Node feature construction from trade/climate inputs
    2. Dilated TCN blocks with graph diffusion
    3. Output head for pathogen logits
    """
    
    def __init__(
        self,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        dilations: List[int] = None,
        diffusion_K: int = 2,
        dropout: float = 0.1,
        num_pathogens: int = 8,
        num_nodes: int = 194,
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        
        self.residual_channels = residual_channels
        self.num_pathogens = num_pathogens
        self.num_nodes = num_nodes
        
        # Node feature dimensions:
        # base_trade: 4 (outflow_exports, outflow_imports, inflow_exports, inflow_imports)
        # risk_trade: 32 (risk_outflow_flat, risk_inflow_flat)
        # climate_anoms: 10
        # Total: 46
        node_feature_dim = 46
        
        # Input projection
        self.input_proj = nn.Conv2d(node_feature_dim, residual_channels, kernel_size=1)
        
        # Static graph supports (will be set in forward pass)
        self.supports: Optional[List[torch.Tensor]] = None
        
        # TCN + Graph convolution layers
        self.tcn_blocks = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        
        for dilation in dilations:
            self.tcn_blocks.append(
                GatedTCNBlock(
                    residual_channels=residual_channels,
                    dilation_channels=dilation_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            # Placeholder: will instantiate graph conv in forward when supports are available
            self.graph_convs.append(None)
        
        # Output layers
        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, kernel_size=1)
        self.end_conv2 = nn.Conv2d(end_channels, num_pathogens, kernel_size=1)
        
    def _build_node_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Construct per-node features from edge tensors (trade) and node tensors (climate).
        
        Args:
            batch: Dict with keys:
                - base_trade: (B, L, N, N, 2)
                - risk_trade: (B, L, N, N, 8, 2)
                - climate_anoms: (B, L, N, 10)
        
        Returns:
            node_features: (B, L, N, 46)
        """
        base_trade = batch['base_trade']  # (B, L, N, N, 2)
        risk_trade = batch['risk_trade']  # (B, L, N, N, 8, 2)
        climate_anoms = batch['climate_anoms']  # (B, L, N, 10)
        
        # Base trade aggregation
        # outflow = sum over dst j
        base_outflow = base_trade.sum(dim=3)  # (B, L, N, 2)
        # inflow = sum over src j
        base_inflow = base_trade.sum(dim=2)  # (B, L, N, 2)
        
        # Concatenate: [outflow_exports, outflow_imports, inflow_exports, inflow_imports]
        base_features = torch.cat([
            base_outflow[..., 0:1],  # outflow exports (FOB)
            base_outflow[..., 1:2],  # outflow imports (FOB)
            base_inflow[..., 0:1],   # inflow exports (FOB)
            base_inflow[..., 1:2],   # inflow imports (FOB)
        ], dim=-1)  # (B, L, N, 4)
        
        # Risk trade aggregation
        # risk_outflow = sum over dst j
        risk_outflow = risk_trade.sum(dim=3)  # (B, L, N, 8, 2)
        # risk_inflow = sum over src j
        risk_inflow = risk_trade.sum(dim=2)  # (B, L, N, 8, 2)
        
        # Flatten (K, C) -> K*C
        B, L, N, K, C = risk_outflow.shape
        risk_outflow_flat = risk_outflow.reshape(B, L, N, K * C)  # (B, L, N, 16)
        risk_inflow_flat = risk_inflow.reshape(B, L, N, K * C)    # (B, L, N, 16)
        
        risk_features = torch.cat([risk_outflow_flat, risk_inflow_flat], dim=-1)  # (B, L, N, 32)
        
        # Climate features: use anomalies directly
        climate_features = climate_anoms  # (B, L, N, 10)
        
        # Concatenate all node features
        node_features = torch.cat([
            base_features,    # 4
            risk_features,    # 32
            climate_features, # 10
        ], dim=-1)  # (B, L, N, 46)
        
        return node_features
    
    def _build_supports(self, adjacency: torch.Tensor) -> List[torch.Tensor]:
        """
        Build static graph supports from adjacency matrix.
        
        Args:
            adjacency: (N, N) uint8 or float adjacency matrix
        
        Returns:
            List of (N, N) normalized support matrices
        """
        # Convert to float
        A = adjacency.float()
        
        # Add self-loops to handle isolated nodes (prevents NaN in normalization)
        N = A.shape[0]
        A = A + torch.eye(N, device=A.device, dtype=A.dtype)
        
        # Row-normalize A -> A_rw
        row_sum = A.sum(dim=1, keepdim=True)
        # Now all rows have at least 1 (from self-loop), so no need for clamp
        A_rw = A / row_sum
        
        # Row-normalize A.T -> A_rw_T (compute separately, don't just transpose)
        A_T = A.T
        row_sum_T = A_T.sum(dim=1, keepdim=True)
        A_rw_T = A_T / row_sum_T
        
        return [A_rw, A_rw_T]
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            batch: Dict with keys matching datamodule output:
                - base_trade, risk_trade, climate_anoms, adjacency_border, etc.
        
        Returns:
            logits: (B, N, P) pathogen status logits
        """
        # Build node features: (B, L, N, F) where F=46
        node_features = self._build_node_features(batch)
        
        # Sanitize NaN inputs (replace with 0) - handles corrupt climate data
        if torch.isnan(node_features).any():
            node_features = torch.nan_to_num(node_features, nan=0.0)
        
        # Transpose to (B, F, N, L) for conv2d
        x = node_features.permute(0, 3, 2, 1)  # (B, 46, N, L)
        
        # Input projection
        x = self.input_proj(x)  # (B, residual_channels, N, L)
        
        # Build static supports if not already done
        if self.supports is None:
            adjacency = batch['adjacency_border']  # (N, N)
            self.supports = self._build_supports(adjacency)
            
            # Instantiate graph convs now that we have supports
            for i in range(len(self.tcn_blocks)):
                self.graph_convs[i] = DiffusionGraphConv(
                    in_channels=self.residual_channels,
                    out_channels=self.residual_channels,
                    supports=self.supports,
                    diffusion_K=2,
                ).to(x.device)
        
        # Accumulate skip connections
        skip_sum = 0
        
        # Pass through TCN + Graph conv layers
        for tcn_block, graph_conv in zip(self.tcn_blocks, self.graph_convs):
            x, skip = tcn_block(x)
            x = graph_conv(x)
            skip_sum = skip_sum + skip
        
        # Take final timestep
        x = skip_sum[..., -1]  # (B, skip_channels, N)
        
        # Output head
        x = F.relu(x)
        x = self.end_conv1(x.unsqueeze(-1))  # Add time dim for conv2d
        x = F.relu(x)
        x = self.end_conv2(x)  # (B, num_pathogens, N, 1)
        
        # Remove time dim and permute to (B, N, P)
        logits = x.squeeze(-1).permute(0, 2, 1)  # (B, N, P)
        
        return logits
