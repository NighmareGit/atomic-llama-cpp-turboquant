@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant
cmake --build build-cuda-b-bin --config Release -j 8 -t rpc-server
if errorlevel 1 exit /b 1
set PORTABLE=build-cuda-b-bin\portable
set BINDIR=build-cuda-b-bin\bin
copy /Y "%BINDIR%\rpc-server.exe" "%PORTABLE%\rpc-server.exe"
copy /Y "%BINDIR%\ggml-cuda.dll" "%PORTABLE%\ggml-cuda.dll"
findstr CMAKE_CUDA_ARCHITECTURES build-cuda-b-bin\CMakeCache.txt
echo RPC_REBUILD_OK