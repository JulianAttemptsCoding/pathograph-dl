param(
  [string]$DATASET = "gs://pathograph-057a2273fe-data/datasets/stepA/v1",
  [switch]$RunPytestGate
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
  New-Item -ItemType Directory -Force -Path $p | Out-Null
}

function Require-File([string]$p) {
  if (!(Test-Path $p)) { throw "MISSING_EXPECTED_FILE: $p" }
}

function Require-Zarr([string]$p) {
  if (!(Test-Path (Join-Path $p "zarr.json"))) { throw "MISSING_ZARR_JSON: $p" }
}

Write-Host "=== Stage StepA v1 from GCS ==="
Write-Host "DATASET = $DATASET"
Write-Host "PWD     = $PWD"

# ---------------------------
# Trade tensors (Zarr)
# ---------------------------
$SRC_FOB  = "$DATASET/trade/imf_imts_step1/trade_fob_tensor.zarr"
$DST_FOB  = "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
Ensure-Dir (Split-Path $DST_FOB)
gcloud storage rsync -r $SRC_FOB $DST_FOB
Require-Zarr $DST_FOB
Require-Zarr (Join-Path $DST_FOB "trade")

$SRC_RISK = "$DATASET/trade/faostat_step2/trade_risk_tensor.zarr"
$DST_RISK = "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
Ensure-Dir (Split-Path $DST_RISK)
gcloud storage rsync -r $SRC_RISK $DST_RISK
Require-Zarr $DST_RISK

# Trade scaler JSON
$SRC_SCALER = "$DATASET/trade/trade_step3_scaler.json"
$DST_SCALER = "data/processed/trade/trade_step3_scaler.json"
Ensure-Dir (Split-Path $DST_SCALER)
gcloud storage cp $SRC_SCALER $DST_SCALER
Require-File $DST_SCALER

# Trade manifests (Step7 verification)
$SRC_IMF_MAN  = "$DATASET/trade/imf_imts_step1/manifest.json"
$DST_IMF_MAN  = "data/processed/trade/imf_imts_step1/manifest.json"
Ensure-Dir (Split-Path $DST_IMF_MAN)
gcloud storage cp $SRC_IMF_MAN $DST_IMF_MAN
Require-File $DST_IMF_MAN

$SRC_RISK_MAN = "$DATASET/trade/faostat_step2/preprocessing_manifest.json"
$DST_RISK_MAN = "data/processed/trade/faostat_step2/preprocessing_manifest.json"
Ensure-Dir (Split-Path $DST_RISK_MAN)
gcloud storage cp $SRC_RISK_MAN $DST_RISK_MAN
Require-File $DST_RISK_MAN

# ---------------------------
# Pathogen status tensor (Zarr)
# ---------------------------
$SRC_PATHO = "$DATASET/pathogen/status_tensor.zarr"
$DST_PATHO = "data/processed/pathogen/status_tensor.zarr"
Ensure-Dir (Split-Path $DST_PATHO)
gcloud storage rsync -r $SRC_PATHO $DST_PATHO
Require-Zarr $DST_PATHO

# ---------------------------
# Climate tensor + anomalies (Zarr)
# ---------------------------
$SRC_CLIM = "$DATASET/climate/climate_tensor.zarr"
$DST_CLIM = "data/processed/climate/climate_tensor.zarr"
Ensure-Dir (Split-Path $DST_CLIM)
gcloud storage rsync -r $SRC_CLIM $DST_CLIM
Require-Zarr $DST_CLIM

$SRC_ANOMS = "$DATASET/climate/climate_step4/climate_anomalies.zarr"
$DST_ANOMS = "data/processed/climate/climate_step4/climate_anomalies.zarr"
Ensure-Dir (Split-Path $DST_ANOMS)
gcloud storage rsync -r $SRC_ANOMS $DST_ANOMS
Require-Zarr $DST_ANOMS

# ---------------------------
# Meta matrices (.npy)
# ---------------------------
Ensure-Dir "data/processed/meta"

$SRC_DIST   = "$DATASET/meta/distance_km.npy"
$DST_DIST   = "data/processed/meta/distance_km.npy"
gcloud storage cp $SRC_DIST $DST_DIST
Require-File $DST_DIST

$SRC_BORDER = "$DATASET/meta/adjacency_border.npy"
$DST_BORDER = "data/processed/meta/adjacency_border.npy"
gcloud storage cp $SRC_BORDER $DST_BORDER
Require-File $DST_BORDER

$SRC_TIME   = "$DATASET/meta/time_index_master.npy"
$DST_TIME   = "data/processed/meta/time_index_master.npy"
gcloud storage cp $SRC_TIME $DST_TIME
Require-File $DST_TIME

Write-Host "=== STAGING COMPLETE ✅ ==="
Write-Host "Trade FOB:      $DST_FOB"
Write-Host "Trade RISK:     $DST_RISK"
Write-Host "Pathogen:       $DST_PATHO"
Write-Host "Climate:        $DST_CLIM"
Write-Host "Climate anoms:  $DST_ANOMS"
Write-Host "Meta:           $DST_DIST, $DST_BORDER, $DST_TIME"
Write-Host ""

if ($RunPytestGate) {
  Write-Host "=== PYTEST GATE (ALL) ==="
  python -m pytest -q
}
