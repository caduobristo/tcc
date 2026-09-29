$CondaExe = 'C:\ProgramData\Miniconda3\Scripts\conda.exe'
$ProjectRoot = $PSScriptRoot
$WorkspaceRoot = Split-Path $ProjectRoot -Parent
$EnvPrefix = Join-Path $WorkspaceRoot '.conda\envs\gfx-classifier'
$CacheRoot = Join-Path $WorkspaceRoot '.cache'
$TempDir = Join-Path $CacheRoot 'temp'
$MplDir = Join-Path $CacheRoot 'matplotlib'
$NumbaDir = Join-Path $CacheRoot 'numba'
$JupyterRoot = Join-Path $CacheRoot 'jupyter'
$JupyterConfigDir = Join-Path $JupyterRoot 'config'
$JupyterDataDir = Join-Path $JupyterRoot 'data'
$JupyterRuntimeDir = Join-Path $JupyterRoot 'runtime'
$IPythonDir = Join-Path $JupyterRoot 'ipython'

if ($args.Count -eq 0) {
    Write-Error 'Usage: run_in_gfx_env.ps1 <command> [args...]'
    exit 1
}

if (!(Test-Path $CondaExe)) {
    Write-Error "Conda executable not found: $CondaExe"
    exit 1
}

if (!(Test-Path $EnvPrefix)) {
    Write-Error "Conda environment not found: $EnvPrefix"
    exit 1
}

@($TempDir, $MplDir, $NumbaDir, $JupyterConfigDir, $JupyterDataDir, $JupyterRuntimeDir, $IPythonDir) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

$env:TMP = $TempDir
$env:TEMP = $TempDir
$env:MPLCONFIGDIR = $MplDir
$env:NUMBA_CACHE_DIR = $NumbaDir
$env:JUPYTER_CONFIG_DIR = $JupyterConfigDir
$env:JUPYTER_DATA_DIR = $JupyterDataDir
$env:JUPYTER_RUNTIME_DIR = $JupyterRuntimeDir
$env:IPYTHONDIR = $IPythonDir
$env:HOME = $WorkspaceRoot

& $CondaExe run --no-capture-output --prefix $EnvPrefix @args
exit $LASTEXITCODE
