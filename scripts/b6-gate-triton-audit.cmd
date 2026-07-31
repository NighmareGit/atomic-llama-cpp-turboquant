@echo off
setlocal
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
if exist C:/projects/atomic-llama-cpp-turboquant/Path-B-Event-Support-Pipeline-Plus/build-cuda-b-bin/portable/rpc-server.exe (echo REPO_PORTABLE=1) else (echo REPO_PORTABLE=0)
if exist C:/backup/pathb-portable/rpc-server.exe (echo BACKUP_PORTABLE=1) else (echo BACKUP_PORTABLE=0)
cd /d C:/projects/atomic-llama-cpp-turboquant/Path-B-Event-Support-Pipeline-Plus
git rev-parse --short HEAD
echo TRITON_AUDIT_OK