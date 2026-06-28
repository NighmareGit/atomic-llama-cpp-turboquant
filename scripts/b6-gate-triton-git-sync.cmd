@echo off
setlocal
cd /d C:\projects\atomic-llama-cpp-turboquant\Path-B-Event-Support-Pipeline-Plus
git stash push -u -m b6-gate-triton-sync 2>nul
git remote get-url gitea 2>nul
if errorlevel 1 git remote add gitea http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git
if not errorlevel 1 git remote set-url gitea http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git
git fetch gitea --prune
git checkout Path-B-Event-Support-Pipeline-Plus 2>nul
if errorlevel 1 git checkout -b Path-B-Event-Support-Pipeline-Plus gitea/Path-B-Event-Support-Pipeline-Plus
git reset --hard gitea/Path-B-Event-Support-Pipeline-Plus
git rev-parse --short HEAD
echo TRITON_GIT_SYNC_OK