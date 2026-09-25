# 手动遥操作采集的一键启动脚本（在 Windows PowerShell 里运行）
#   powershell -ExecutionPolicy Bypass -File .\run_manual.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_manual.ps1 -Episodes 30 -Speed 0.08
param(
    [int]$Episodes = 20,
    [double]$Speed = 0.12
)

chcp 65001 > $null                                   # 控制台用 UTF-8，中文才不乱码
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -Path $PSScriptRoot

$py = 'C:\Users\sgz\anaconda3\envs\lerobot_mujoco_vla_tutorial\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }         # 找不到就退回 PATH 里的 python

Write-Host "启动手动采集：目标 $Episodes 段，速度 $Speed m/s" -ForegroundColor Cyan
Write-Host "MuJoCo 窗口出来以后，先用鼠标点一下那个窗口拿到键盘焦点，再按键。" -ForegroundColor Yellow

& $py -u manual_collect.py --episodes $Episodes --speed $Speed 2>&1 |
    Tee-Object -FilePath 'manual_session.log'        # 同时也留一份日志，方便事后排查
