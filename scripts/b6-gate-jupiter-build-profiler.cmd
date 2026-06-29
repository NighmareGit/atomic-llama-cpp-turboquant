@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d D:\projects\atomic-llama-cpp-5070ti\atomic-llama-cpp-turboquant
cmake --build build-cuda-b-bin --config Release -j 8 -t llama-pipeline-profiler
if errorlevel 1 exit /b 1
set PORTABLE=build-cuda-b-bin\portable
set BINDIR=build-cuda-b-bin\bin
copy /Y "%BINDIR%\llama-pipeline-profiler.exe" "%PORTABLE%\llama-pipeline-profiler.exe"
copy /Y "%BINDIR%\llama-pipeline-profiler-impl.dll" "%PORTABLE%\llama-pipeline-profiler-impl.dll"
copy /Y "%BINDIR%\ggml-rpc.dll" "%PORTABLE%\ggml-rpc.dll"
echo PROFILER_BUILD_OK