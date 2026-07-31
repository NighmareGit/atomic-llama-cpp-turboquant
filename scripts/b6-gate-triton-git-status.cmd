@echo off
setlocal
cd /d C:/projects/atomic-llama-cpp-turboquant/Path-B-Event-Support-Pipeline-Plus
hostname
git rev-parse --short HEAD
git status -sb
echo TRITON_GIT_STATUS_OK