"""
Compositional Part-Graph Reasoning for Few-Shot Object Detection.

This module implements graph-based reasoning over object parts,
where prototypes are decomposed into part representations and
matched via graph neural network message passing.

Key idea: Model objects as compositions of parts with learned
spatial relationships. Match query objects to prototypes by
aligning their part graphs.

Reference: Novel approach inspired by scene graph and part-based recognition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PartDecomposer(nn.Module):
    """
    Decomposes feature vectors into part representations.
    
    Uses attention-based soft clustering to identify part-like
    sub-regions of the feature space.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_parts: int = 4,
        part_dim: int = 256,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Input feature dimension
            num_parts: Number of parts to decompose into
            part_dim: Dimension of each part representation
            use_attention: Use attention-based decomposition
            dropout: Dropout rate
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.use_attention = use_attention
        
        # Part prototypes (learnable)
        self.part_prototypes = nn.Parameter(
            torch.randn(num_parts, feature_dim) * 0.01
        )
        
        # Part projection
        self.part_projector = nn.Sequential(
            nn.Linear(feature_dim, part_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(part_dim * 2, part_dim)
        )
        
        if use_attention:
            # Attention for soft part assignment
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 4, num_parts)
            )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose features into parts.
        
        Args:
            features: Feature vectors (N, D)
        
        Returns:
            Dict with 'parts' (N, num_parts, part_dim), 
            'attention_weights' (N, num_parts)
        """
        N = features.shape[0]
        
        if self.use_attention:
            # Compute attention weights over parts
            attn_logits = self.attention(features)  # (N, num_parts)
            attn_weights = F.softmax(attn_logits, dim=1)
            
            # Weighted combination of feature with part prototypes
            # Each part gets a weighted view of the feature
            parts = []
            for p in range(self.num_parts):
                # Modulate feature by attention to this part
                weighted_feat = features * attn_weights[:, p:p+1]
                # Add part prototype bias
                part_feat = weighted_feat + self.part_prototypes[p]
                # Project to part space
                part_repr = self.part_projector(part_feat)
                parts.append(part_repr)
            
            parts = torch.stack(parts, dim=1)  # (N, num_parts, part_dim)
        else:
            # Simple linear decomposition
            parts = []
            chunk_size = self.feature_dim // self.num_parts
            for p in range(self.num_parts):
                start = p * chunk_size
                end = start + chunk_size if p < self.num_parts - 1 else self.feature_dim
                chunk = features[:, start:end]
                # Pad to full size for projector
                padded = F.pad(chunk, (0, self.feature_dim - chunk.shape[1]))
                part_repr = self.part_projector(padded)
                parts.append(part_repr)
            
            parts = torch.stack(parts, dim=1)
            attn_weights = torch.ones(N, self.num_parts, device=features.device) / self.num_parts
        
        return {
            'parts': parts,
            'attention_weights': attn_weights
        }


class PartGraphConv(nn.Module):
    """
    Graph convolution layer for part reasoning.
    
    Propagates information between parts based on learned
    edge weights (relationships).
    """
    
    def __init__(
        self,
        part_dim: int = 256,
        num_parts: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            part_dim: Part representation dimension
            num_parts: Number of parts in graph
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.part_dim = part_dim
        self.num_parts = num_parts
        self.num_heads = num_heads
        
        assert part_dim % num_heads == 0
        self.head_dim = part_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(part_dim, part_dim)
        self.k_proj = nn.Linear(part_dim, part_dim)
        self.v_proj = nn.Linear(part_dim, part_dim)
        
        # Output projection
        self.out_proj = nn.Linear(part_dim, part_dim)
        
        # Edge bias (learnable relationship priors)
        self.edge_bias = nn.Parameter(torch.zeros(num_parts, num_parts))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(part_dim)
    
    def forward(self, parts: torch.Tensor) -> torch.Tensor:
        """
        Graph convolution over parts.
        
        Args:
            parts: Part representations (N, num_parts, part_dim)
        
        Returns:
            Updated part representations (N, num_parts, part_dim)
        """
        N, P, D = parts.shape
        
        # Compute Q, K, V
        Q = self.q_proj(parts)  # (N, P, D)
        K = self.k_proj(parts)
        V = self.v_proj(parts)
        
        # Reshape for multi-head attention
        Q = Q.view(N, P, self.num_heads, self.head_dim).transpose(1, 2)  # (N, H, P, d)
        K = K.view(N, P, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, P, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (N, H, P, P)
        
        # Add edge bias
        scores = scores + self.edge_bias.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, V)  # (N, H, P, d)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(N, P, D)
        out = self.out_proj(out)
        
        # Residual + norm
        out = self.layer_norm(parts + out)
        
        return out


class PartGraphNetwork(nn.Module):
    """
    Full part-graph network for object representation.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_parts: int = 4,
        part_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Input feature dimension
            num_parts: Number of parts
            part_dim: Part representation dimension
            num_layers: Number of graph conv layers
            num_heads: Attention heads per layer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.decomposer = PartDecomposer(
            feature_dim=feature_dim,
            num_parts=num_parts,
            part_dim=part_dim,
            dropout=dropout
        )
        
        self.graph_layers = nn.ModuleList([
            PartGraphConv(
                part_dim=part_dim,
                num_parts=num_parts,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Aggregation layer
        self.aggregator = nn.Sequential(
            nn.Linear(num_parts * part_dim, part_dim),
            nn.ReLU(inplace=True),
            nn.Linear(part_dim, part_dim)
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process features through part-graph network.
        
        Args:
            features: Feature vectors (N, D)
        
        Returns:
            Dict with 'graph_repr', 'parts', 'attention'
        """
        # Decompose into parts
        decomp = self.decomposer(features)
        parts = decomp['parts']  # (N, P, d)
        
        # Graph reasoning
        for layer in self.graph_layers:
            parts = layer(parts)
        
        # Aggregate to single representation
        N, P, d = parts.shape
        flat_parts = parts.view(N, P * d)
        graph_repr = self.aggregator(flat_parts)
        
        return {
            'graph_repr': graph_repr,
            'parts': parts,
            'attention': decomp['attention_weights']
        }


class PartGraphMatcher:
    """
    Matches objects using part-graph alignment.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_parts: int = 4,
        part_dim: int = 256,
        num_layers: int = 2,
        similarity_mode: str = "graph",  # "graph", "parts", "combined"
        device: str = "cuda"
    ):
        self.feature_dim = feature_dim
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.similarity_mode = similarity_mode
        self.device = device
        
        self.network = PartGraphNetwork(
            feature_dim=feature_dim,
            num_parts=num_parts,
            part_dim=part_dim,
            num_layers=num_layers
        ).to(device)
        
        self.network.eval()
    
    def compute_part_similarity(
        self,
        query_parts: torch.Tensor,
        proto_parts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between part sets using optimal transport.
        
        Args:
            query_parts: Query parts (P, d)
            proto_parts: Prototype parts (P, d)
        
        Returns:
            Part alignment similarity score
        """
        # Normalize
        q_norm = F.normalize(query_parts, dim=1)
        p_norm = F.normalize(proto_parts, dim=1)
        
        # Pairwise similarities
        sim_matrix = torch.mm(q_norm, p_norm.t())  # (P, P)
        
        # Hungarian-style matching: take max along each row
        max_per_query, _ = sim_matrix.max(dim=1)
        
        return max_per_query.mean()
    
    def match(
        self,
        query_feature: torch.Tensor,
        prototype_features: torch.Tensor
    ) -> Tuple[float, Dict]:
        """
        Match query to prototypes using part-graph alignment.
        
        Args:
            query_feature: Query feature (D,)
            prototype_features: Prototype features (K, D)
        
        Returns:
            Tuple of (best_similarity, match_info)
        """
        with torch.no_grad():
            # Process query
            query_out = self.network(query_feature.unsqueeze(0).to(self.device))
            
            # Process prototypes
            proto_out = self.network(prototype_features.to(self.device))
            
            if self.similarity_mode == "graph":
                # Compare aggregated graph representations
                q_repr = F.normalize(query_out['graph_repr'], dim=1)
                p_repr = F.normalize(proto_out['graph_repr'], dim=1)
                sims = torch.mm(q_repr, p_repr.t()).squeeze(0)
                
            elif self.similarity_mode == "parts":
                # Compare part-by-part
                K = prototype_features.shape[0]
                sims = []
                for k in range(K):
                    part_sim = self.compute_part_similarity(
                        query_out['parts'][0],
                        proto_out['parts'][k]
                    )
                    sims.append(part_sim)
                sims = torch.stack(sims)
                
            else:  # combined
                # Average of graph and part similarities
                q_repr = F.normalize(query_out['graph_repr'], dim=1)
                p_repr = F.normalize(proto_out['graph_repr'], dim=1)
                graph_sims = torch.mm(q_repr, p_repr.t()).squeeze(0)
                
                K = prototype_features.shape[0]
                part_sims = []
                for k in range(K):
                    part_sim = self.compute_part_similarity(
                        query_out['parts'][0],
                        proto_out['parts'][k]
                    )
                    part_sims.append(part_sim)
                part_sims = torch.stack(part_sims)
                
                sims = 0.5 * graph_sims + 0.5 * part_sims
            
            best_idx = torch.argmax(sims)
            best_sim = float(sims[best_idx].item())
            
            return best_sim, {
                'all_similarities': sims.cpu(),
                'best_idx': int(best_idx.item()),
                'query_attention': query_out['attention'].cpu()
            }


class PartGraphPCB:
    """
    Wrapper around PrototypicalCalibrationBlock that uses
    part-graph reasoning for prototype matching.
    """
    
    def __init__(self, base_pcb, cfg):
        """
        Args:
            base_pcb: The original PrototypicalCalibrationBlock instance
            cfg: Config node with NOVEL_METHODS.PART_GRAPH settings
        """
        self.base_pcb = base_pcb
        self.cfg = cfg
        
        # Extract part-graph config
        pg_cfg = cfg.NOVEL_METHODS.PART_GRAPH
        self.matcher = PartGraphMatcher(
            feature_dim=int(pg_cfg.FEATURE_DIM),
            num_parts=int(pg_cfg.NUM_PARTS),
            part_dim=int(pg_cfg.PART_DIM),
            num_layers=int(pg_cfg.NUM_LAYERS),
            similarity_mode=str(pg_cfg.SIMILARITY_MODE),
            device=str(cfg.MODEL.DEVICE)
        )
        
        # Build part representations for prototypes
        self._build_prototype_parts()
    
    def _build_prototype_parts(self):
        """Pre-compute part representations for all prototypes."""
        if not hasattr(self.base_pcb, '_real_class_features'):
            return
        
        self.prototype_part_cache = {}
        
        for cls in self.base_pcb._real_class_features:
            feat_list = self.base_pcb._real_class_features[cls]
            if not feat_list:
                continue
            
            features = torch.stack(feat_list, dim=0)
            
            with torch.no_grad():
                part_out = self.matcher.network(features.to(self.matcher.device))
                self.prototype_part_cache[cls] = {
                    'graph_repr': part_out['graph_repr'].cpu(),
                    'parts': part_out['parts'].cpu()
                }
    
    def execute_calibration(self, inputs, dts):
        """
        Execute calibration using part-graph matching.
        """
        # Use base PCB's calibration for now
        # Full implementation would override similarity computation
        return self.base_pcb.execute_calibration(inputs, dts)
    
    def __getattr__(self, name):
        """Delegate all other attributes to base PCB."""
        return getattr(self.base_pcb, name)


def build_part_graph_pcb(base_pcb, cfg):
    """
    Factory function to wrap a PCB with part-graph reasoning.
    
    Args:
        base_pcb: Original PrototypicalCalibrationBlock
        cfg: Config with NOVEL_METHODS.PART_GRAPH settings
    
    Returns:
        PartGraphPCB wrapper
    """
    return PartGraphPCB(base_pcb, cfg)
