from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from .trade_zarr import open_trade_zarr


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TradeSplit:
    t_min: int
    t_max: int


@dataclass(frozen=True)
class TradeDatasetConfig:
    base_zarr_path: str
    risk_zarr_path: str
    lookback: int = 24
    horizon: int = 1
    split: SplitName = "train"
    split_train: TradeSplit = TradeSplit(0, 815)
    split_val: TradeSplit = TradeSplit(816, 851)
    split_test: TradeSplit = TradeSplit(852, 907)
    apply_log1p: bool = True
    standardize: bool = True
    scaler_json_path: Optional[str] = None
    return_mode: Literal["separate", "concat"] = "separate"
    
    # Target config
    return_targets: bool = False
    target_kind: Literal["base", "risk", "both", "status"] = "base"
    label_mode: Literal["status", "incident"] = "status"
    include_target_masks: bool = True
    # Future-proofing: implicit "target_transform" is assumed "same_as_inputs" for now.
    
    # Pathogen status target (only used when target_kind="status")
    pathogen_zarr_path: Optional[str] = None
    
    # Climate multimodal inputs
    climate_zarr_path: Optional[str] = None
    climate_array_key: str = "climate"
    climate_anoms_zarr_path: Optional[str] = None
    climate_anoms_array_key: str = "anomaly"  # or "anomalies" depending on preprocessing
    
    # Meta matrices (distance, adjacency)
    meta_distance_path: Optional[str] = None
    meta_adjacency_path: Optional[str] = None
    
    # Valid-index filtering (gated by return_targets)
    require_target_observed: bool = False
    min_target_observed: int = 1
    require_target_observed_kind: Optional[Literal["base", "risk", "both", "status"]] = None
    valid_t_cache_dir: Optional[str] = None
    debug_guard: bool = False


def _month_of_year_from_t(t: int) -> int:
    # t=0 -> Jan (1)
    return (t % 12) + 1


def _sin_cos_month(m: int) -> Tuple[float, float]:
    # m in 1..12
    ang = 2.0 * np.pi * (m - 1) / 12.0
    return float(np.sin(ang)), float(np.cos(ang))


class TradeDatasetZarr:
    """Zarr-backed dataset for trade tensors.

    Returns windows ending at time t, predicting t+horizon.

    Notes:
      - Missingness is represented via mask arrays (uint8). Values for unobserved cells should be treated as 0.
      - is_estimated indicates CIF-derived import estimates (and derived risk estimates).
    """

    def __init__(self, cfg: TradeDatasetConfig):
        self.cfg = cfg
        self.h = open_trade_zarr(cfg.base_zarr_path, cfg.risk_zarr_path)
        
        # Load pathogen status if target_kind is "status"
        self.pathogen_h = None
        if cfg.target_kind == "status":
            if not cfg.pathogen_zarr_path:
                raise ValueError('target_kind="status" requires pathogen_zarr_path')
            from .pathogen_zarr import open_pathogen_zarr
            self.pathogen_h = open_pathogen_zarr(cfg.pathogen_zarr_path)
            # Validate time alignment
            if self.pathogen_h.T != self.h.T:
                raise ValueError(
                    f"Pathogen T={self.pathogen_h.T} does not match trade T={self.h.T}"
                )
        
        # Load climate tensors (optional for multimodal)
        self.climate_h = None
        self.climate_anoms_h = None
        if cfg.climate_zarr_path:
            from .climate_zarr import open_climate_zarr
            self.climate_h = open_climate_zarr(cfg.climate_zarr_path, cfg.climate_array_key)
            if self.climate_h.T != self.h.T:
                raise ValueError(
                    f"Climate T={self.climate_h.T} does not match trade T={self.h.T}"
                )
        
        if cfg.climate_anoms_zarr_path:
            from .climate_zarr import open_climate_zarr
            self.climate_anoms_h = open_climate_zarr(cfg.climate_anoms_zarr_path, cfg.climate_anoms_array_key)
            if self.climate_anoms_h.T != self.h.T:
                raise ValueError(
                    f"Climate anomalies T={self.climate_anoms_h.T} does not match trade T={self.h.T}"
                )
        
        # Load meta matrices (optional for multimodal)
        self.distance_km = None
        self.adjacency_border = None
        if cfg.meta_distance_path and cfg.meta_adjacency_path:
            from .climate_zarr import load_meta_matrices
            self.distance_km, self.adjacency_border = load_meta_matrices(
                cfg.meta_distance_path,
                cfg.meta_adjacency_path,
                expected_N=self.h.N
            )

        # choose split
        split_map = {
            "train": cfg.split_train,
            "val": cfg.split_val,
            "test": cfg.split_test,
        }
        sp = split_map[cfg.split]

        # We need t such that [t-lookback+1 .. t] is valid AND t+horizon is valid within split bounds.
        self.t_start = max(sp.t_min + (cfg.lookback - 1), 0)
        self.t_end = min(sp.t_max - cfg.horizon, self.h.T - 1 - cfg.horizon)
        if self.t_end < self.t_start:
            raise ValueError(f"Split window too small for lookback/horizon. start={self.t_start}, end={self.t_end}")

        self._scaler = None
        if cfg.standardize:
            if not cfg.scaler_json_path:
                raise ValueError("standardize=True requires scaler_json_path")
            with open(cfg.scaler_json_path, "r", encoding="utf-8") as f:
                self._scaler = json.load(f)
        
        # Valid-index filtering (only when return_targets AND require_target_observed)
        self._valid_t: Optional[np.ndarray] = None
        self._filtering_enabled = cfg.return_targets and cfg.require_target_observed
        
        if self._filtering_enabled:
            self._valid_t = self._build_valid_t()
            if len(self._valid_t) == 0:
                raise ValueError(
                    f"No valid time indices found after filtering. "
                    f"Split {cfg.split}, t_start={self.t_start}, t_end={self.t_end}, "
                    f"require_kind={cfg.require_target_observed_kind or cfg.target_kind}, "
                    f"min={cfg.min_target_observed}"
                )
            # Log filtering status
            total_range = self.t_end - self.t_start + 1
            print(f"[TradeDataset {cfg.split}] Filtering enabled: "
                  f"valid_t={len(self._valid_t)}/{total_range}, "
                  f"first_t={int(self._valid_t[0])}, last_t={int(self._valid_t[-1])}")
        else:
            total_range = self.t_end - self.t_start + 1
            print(f"[TradeDataset {cfg.split}] Filtering DISABLED: "
                  f"using all {total_range} indices in range")
    
    def _build_valid_t(self) -> np.ndarray:
        """Build array of valid time indices where target masks meet threshold."""
        import hashlib
        from pathlib import Path
        
        cfg = self.cfg
        require_kind = cfg.require_target_observed_kind or cfg.target_kind
        min_obs = cfg.min_target_observed
        H = cfg.horizon
        
        # Cache key components
        cache_key_parts = [
            cfg.split,
            str(self.t_start),
            str(self.t_end),
            str(cfg.lookback),
            str(H),
            require_kind,
            str(min_obs),
            cfg.base_zarr_path,
            cfg.risk_zarr_path,
        ]
        cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()[:12]
        
        # Try loading from cache
        if cfg.valid_t_cache_dir:
            cache_dir = Path(cfg.valid_t_cache_dir)
            cache_path = cache_dir / f"valid_t_{cache_key}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Scan and filter
        valid_t_list = []
        for t in range(self.t_start, self.t_end + 1):
            t_y = t + H
            
            # Check base target mask
            base_ok = True
            if require_kind in ("base", "both"):
                base_m = self.h.base_mask[t_y]
                base_count = int(np.count_nonzero(base_m != 0))
                base_ok = base_count >= min_obs
            
            # Check risk target mask
            risk_ok = True
            if require_kind in ("risk", "both"):
                risk_m = self.h.risk_mask[t_y]
                risk_count = int(np.count_nonzero(risk_m != 0))
                risk_ok = risk_count >= min_obs
            
            # Check status target mask
            status_ok = True
            if require_kind == "status":
                if self.pathogen_h is None:
                    raise ValueError("pathogen_h not loaded but require_kind is status")
                status_m = self.pathogen_h.status_mask[t_y]
                if cfg.label_mode == "incident":
                    prev_m = self.pathogen_h.status_mask[t]
                    status_m = status_m & prev_m
                status_count = int(np.count_nonzero(status_m != 0))
                status_ok = status_count >= min_obs
            
            if base_ok and risk_ok and status_ok:
                valid_t_list.append(t)
        
        valid_t = np.array(valid_t_list, dtype=np.int32)
        
        # Save to cache if configured
        if cfg.valid_t_cache_dir:
            cache_dir = Path(cfg.valid_t_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"valid_t_{cache_key}.npy"
            np.save(cache_path, valid_t)
        
        return valid_t

    def __len__(self) -> int:
        if self._filtering_enabled:
            return len(self._valid_t)
        return int(self.t_end - self.t_start + 1)
    
    @property
    def valid_t_len(self) -> int:
        """Number of valid time indices after filtering (0 if disabled)."""
        return len(self._valid_t) if self._valid_t is not None else 0
    
    @property
    def first_valid_t(self) -> Optional[int]:
        """First valid time index (None if no filtering or empty)."""
        if self._valid_t is not None and len(self._valid_t) > 0:
            return int(self._valid_t[0])
        return None
    
    @property
    def last_valid_t(self) -> Optional[int]:
        """Last valid time index (None if no filtering or empty)."""
        if self._valid_t is not None and len(self._valid_t) > 0:
            return int(self._valid_t[-1])
        return None

    def _apply_transforms(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # x is float32; mean/std are 1D per-feature
        if self.cfg.apply_log1p:
            x = np.log1p(np.maximum(x, 0.0))
        if self.cfg.standardize:
            # broadcast mean/std to x's last dim
            x = (x - mean) / (std + 1e-8)
        return x

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if self._filtering_enabled:
            t = int(self._valid_t[idx])
        else:
            t = self.t_start + idx
        L = self.cfg.lookback
        H = self.cfg.horizon

        t0 = t - (L - 1)
        t1 = t + 1  # slice end exclusive
        t_y = t + H
        
        # Safety check for t_y (should be covered by t_end logic but good simply to assert)
        if t_y >= self.h.T:
             raise IndexError(f"Target index t_y={t_y} out of bounds (T={self.h.T})")

        # Load base window
        base_trade = self.h.base_trade[t0:t1, :, :, :]            # (L,N,N,2)
        
        # Base mask is categorical codes (0=valid, 1=null, 2=missing, ...).
        # AUDIT FIX: Real data shows Code 1 contains 99% of trade (2.7T vs 34B). 
        # So Code 1 (and non-zero) is Observed. Code 0 is Missing/Null.
        base_mask_codes = self.h.base_mask[t0:t1, :, :].astype(np.uint8) # (L,N,N)
        base_mask = (base_mask_codes != 0).astype(np.uint8)
        
        base_est = self.h.base_is_estimated[t0:t1, :, :].astype(np.uint8)

        # Load risk window
        risk_trade = self.h.risk_trade[t0:t1, :, :, :, :]         # (L,N,N,K,2)
        # Risk mask is "observed_mask" (1=observed, 0=missing) already.
        risk_mask = self.h.risk_mask[t0:t1, :, :, :].astype(np.uint8) # (L,N,N,K)
        risk_est = self.h.risk_is_estimated[t0:t1, :, :, :].astype(np.uint8)

        # Target time feature (month-of-year) for t_y
        m = _month_of_year_from_t(t_y)
        sin_m, cos_m = _sin_cos_month(m)
        time_feat = np.array([sin_m, cos_m], dtype=np.float32)

        # Prepare Scaler
        base_mean = None
        base_std = None
        risk_mean = None
        risk_std = None
        
        if self.cfg.standardize:
            sc = self._scaler
            base_mean = np.array(sc["base"]["mean"], dtype=np.float32)  # (2,)
            base_std = np.array(sc["base"]["std"], dtype=np.float32)    # (2,)
            risk_mean = np.array(sc["risk"]["mean"], dtype=np.float32)  # (K*2,)
            risk_std = np.array(sc["risk"]["std"], dtype=np.float32)

        # Apply Transforms to Inputs
        if self.cfg.standardize:
            base_trade = self._apply_transforms(base_trade, base_mean, base_std)
            risk_flat = risk_trade.reshape((L, self.h.N, self.h.N, self.h.K * 2))
            risk_flat = self._apply_transforms(risk_flat, risk_mean, risk_std)
            risk_trade = risk_flat.reshape((L, self.h.N, self.h.N, self.h.K, 2))
        else:
            if self.cfg.apply_log1p:
                base_trade = np.log1p(np.maximum(base_trade, 0.0))
                risk_trade = np.log1p(np.maximum(risk_trade, 0.0))
        
        # Prepare targets if requested
        targets = {}
        if self.cfg.return_targets:
            # We treat targets exactly like inputs: same transforms.
            # Base Target
            if self.cfg.target_kind in ("base", "both"):
                y_base = self.h.base_trade[t_y, :, :, :]    # (N,N,2)
                y_base_m_codes = self.h.base_mask[t_y, :, :].astype(np.uint8) # (N,N)
                y_base_m = (y_base_m_codes != 0).astype(np.uint8)
                
                y_base_e = self.h.base_is_estimated[t_y, :, :].astype(np.uint8)
                
                if self.cfg.standardize:
                    y_base = self._apply_transforms(y_base, base_mean, base_std)
                elif self.cfg.apply_log1p:
                    y_base = np.log1p(np.maximum(y_base, 0.0))
                
                targets["y_base"] = y_base.astype(np.float32)
                if self.cfg.include_target_masks:
                    targets["y_base_mask"] = y_base_m
                    targets["y_base_is_estimated"] = y_base_e

            # Risk Target
            if self.cfg.target_kind in ("risk", "both"):
                y_risk = self.h.risk_trade[t_y, :, :, :, :] # (N,N,K,2)
                y_risk_m = self.h.risk_mask[t_y, :, :, :].astype(np.uint8) # (N,N,K)
                y_risk_e = self.h.risk_is_estimated[t_y, :, :, :].astype(np.uint8)
                
                if self.cfg.standardize:
                    # Flatten K*2
                    y_risk_flat = y_risk.reshape((self.h.N, self.h.N, self.h.K * 2))
                    y_risk_flat = self._apply_transforms(y_risk_flat, risk_mean, risk_std)
                    y_risk = y_risk_flat.reshape((self.h.N, self.h.N, self.h.K, 2))
                elif self.cfg.apply_log1p:
                    y_risk = np.log1p(np.maximum(y_risk, 0.0))
                
                targets["y_risk"] = y_risk.astype(np.float32)
                if self.cfg.include_target_masks:
                    targets["y_risk_mask"] = y_risk_m
                    targets["y_risk_is_estimated"] = y_risk_e
            
            # Status Target (Pathogen)
            if self.cfg.target_kind == "status":
                if self.pathogen_h is None:
                    raise ValueError("pathogen_h not loaded but target_kind is status")
                
                # Load pathogen status at t_y: (N, P)
                y_status = self.pathogen_h.status[t_y, :, :]  # (N, P) = (194, 8)
                y_status_m = self.pathogen_h.status_mask[t_y, :, :].astype(np.uint8)  # (N, P)
                
                # Pathogen status is categorical (0/1 or codes), no log/standardization
                # Convert to float32 for model
                targets["y_next"] = y_status.astype(np.float32)
                if self.cfg.include_target_masks:
                    targets["y_mask"] = y_status_m
                
                # Incident mode specific targets
                if self.cfg.label_mode == "incident":
                    prev = self.pathogen_h.status[t, :, :] 
                    future = y_status
                    y_incident = ((future == 1) & (prev == 0)).astype(np.float32)
                    
                    prev_m = self.pathogen_h.status_mask[t, :, :].astype(np.uint8)
                    mask_incident = y_status_m & prev_m
                    
                    targets["y_incident"] = y_incident
                    if self.cfg.include_target_masks:
                        targets["y_incident_mask"] = mask_incident

        if self.cfg.return_mode == "separate":
            ret = {
                "t": np.int32(t),
                "t_y": np.int32(t_y),
                "time_feat": time_feat,                 # (2,)
                "base_trade": base_trade.astype(np.float32),
                "base_mask": base_mask,
                "base_is_estimated": base_est,
                "risk_trade": risk_trade.astype(np.float32),
                "risk_mask": risk_mask,
                "risk_is_estimated": risk_est,
            }
            
            # Add climate inputs if loaded
            if self.climate_h is not None:
                climate_window = self.climate_h.climate[t0:t1, :, :]  # (L, N, F)
                ret["climate"] = climate_window.astype(np.float32)
            
            if self.climate_anoms_h is not None:
                anoms_window = self.climate_anoms_h.climate[t0:t1, :, :]  # (L, N, F)
                ret["climate_anoms"] = anoms_window.astype(np.float32)
            
            # Add meta matrices if loaded (not time-dependent)
            if self.distance_km is not None:
                ret["distance_km"] = self.distance_km
            
            if self.adjacency_border is not None:
                ret["adjacency_border"] = self.adjacency_border
            
            if self.cfg.return_targets:
                ret.update(targets)
                
            for k, v in ret.items():
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
                    # Runtime explicit guards requested by user
                    if self.cfg.debug_guard:
                        if not np.isfinite(v).all():
                            raise ValueError(f"FATAL: Non-finite values detected in batch key {k} at t={t}")
                        vmax = np.abs(v).max()
                        if vmax > 1e4:
                            raise ValueError(f"FATAL: Scale explosion detected in batch key {k}. "
                                             f"Max abs value {vmax} > 1e4 at t={t}. "
                                             f"Config: standardize={self.cfg.standardize}, apply_log1p={self.cfg.apply_log1p}")
                                         
                    ret[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    
            return ret

        # concat mode: flatten risk (K,2)->(K*2) and concatenate features along last dim
        L_, N = L, self.h.N
        risk_val = risk_trade.reshape((L_, N, N, self.h.K * 2))
        risk_m = risk_mask.reshape((L_, N, N, self.h.K * 2))
        risk_e = risk_est.reshape((L_, N, N, self.h.K * 2))

        base_val = base_trade
        base_m = base_mask
        base_e = base_est

        feat = np.concatenate([
            base_val,
            risk_val,
            base_m.astype(np.float32),
            risk_m.astype(np.float32),
            base_e.astype(np.float32),
            risk_e.astype(np.float32),
        ], axis=-1).astype(np.float32)

        ret = {
            "t": np.int32(t),
            "t_y": np.int32(t_y),
            "time_feat": time_feat,
            "edge_feat": feat,   # (L,N,N, 2 + 16 + 2 + 16 + 2 + 16 = 54)
        }
        # In concat mode, we likely still return targets as separate keys because "predicting output" 
        # is usually distinct from "graph input".
        if self.cfg.return_targets:
            # We won't concat targets into edge_feat (that would be weird for future prediction), 
            # so we just add them as separate tensors.
            ret.update(targets)
            
        for k, v in ret.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
                # Runtime explicit guards requested by user
                if self.cfg.debug_guard:
                    if not np.isfinite(v).all():
                        raise ValueError(f"FATAL: Non-finite values detected in batch key {k} at t={t}")
                    vmax = np.abs(v).max()
                    if vmax > 1e4:
                        raise ValueError(f"FATAL: Scale explosion detected in batch key {k}. "
                                         f"Max abs value {vmax} > 1e4 at t={t}. "
                                         f"Config: standardize={self.cfg.standardize}, apply_log1p={self.cfg.apply_log1p}")
                
                ret[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            
        return ret
