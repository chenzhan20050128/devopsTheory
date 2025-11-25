# 完整训练命令脚本 (PowerShell)
# 使用默认配置运行训练

# 切换到项目根目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Join-Path $scriptPath "..\.."
Set-Location $projectRoot

# 运行训练
python -m openstack_textcnn.src2.train

# 如果需要自定义参数，可以使用以下命令：
# python -m openstack_textcnn.src2.train --device cuda --epochs 20 --batch-size 32 --lr 0.0005 --output-dir "openstack_textcnn/artifacts_forecast_v2"

Write-Host "训练完成！按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

