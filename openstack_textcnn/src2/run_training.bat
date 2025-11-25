@echo off
REM 完整训练命令脚本
REM 使用默认配置运行训练

cd /d "%~dp0\..\.."
python -m openstack_textcnn.src2.train

pause

