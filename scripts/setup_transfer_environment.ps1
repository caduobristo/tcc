param(
    [string]$BasePython = "C:\TCC\experiments\fxnet_reproduction\.conda\envs\gfx-classifier\python.exe",
    [switch]$Standalone
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot ".venv-transfer"
if ($Standalone -and (Test-Path -LiteralPath (Join-Path $environmentPath "pyvenv.cfg"))) {
    if ((Get-Content -LiteralPath (Join-Path $environmentPath "pyvenv.cfg")) -match "include-system-site-packages = true") {
        throw "The existing environment is an overlay. Preserve it; use a fresh checkout for standalone setup."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $environmentPath "Scripts\python.exe"))) {
    if ($Standalone) {
        & $BasePython -m venv $environmentPath
    } else {
        # Local overlay: preserve the base environment, reuse its working CUDA stack.
        & $BasePython -m venv --system-site-packages $environmentPath
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the transfer-learning environment" }
}
$transferPython = Join-Path $environmentPath "Scripts\python.exe"
if ($Standalone) {
    & $transferPython -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA installation failed" }
}
& $transferPython -m pip install -r (Join-Path $projectRoot "requirements-transfer.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $transferPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency validation failed" }
& $transferPython -m ipykernel install --user --name tcc-transfer --display-name "TCC Transfer Learning (PyTorch CUDA)"
if ($LASTEXITCODE -ne 0) { throw "Kernel installation failed" }
Write-Output "Environment ready. No training started."
