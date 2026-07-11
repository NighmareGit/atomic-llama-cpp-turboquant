# AI Agent Patch Guide: Windows CUDA Dynamic Build & Sampler Assertion Fix

This document outlines the root causes of the Windows DLL linking errors and the runtime sampler assertion crashes when compiling with dynamic load support (`GGML_BACKEND_DL=ON` and `GGML_RPC=ON`), and provides step-by-step instructions to apply the patch.

---

## 1. Problem Statements & Root Causes

### 1.1. Unresolved Symbols in `ggml-base.dll` (Build Time)
The fork-specific RPC optimizations in [ggml-backend.cpp](file:///D:/projects/atomic-llama-cpp-5070ti/atomic-llama-cpp-turboquant/ggml/src/ggml-backend.cpp) directly call functions declared in [ggml-rpc.h](file:///D:/projects/atomic-llama-cpp-5070ti/atomic-llama-cpp-turboquant/ggml/include/ggml-rpc.h) (e.g., [ggml_backend_is_rpc](file:///D:/projects/atomic-llama-cpp-5070ti/atomic-llama-cpp-turboquant/ggml/include/ggml-rpc.h#L23) and [ggml_backend_rpc_event_defer_barrier](file:///D:/projects/atomic-llama-cpp-5070ti/atomic-llama-cpp-turboquant/ggml/include/ggml-rpc.h#L58)).

- When compiling with `GGML_BACKEND_DL=ON`, `ggml-rpc` is built as a separate `MODULE` target (plugin DLL) while `ggml-base` is compiled as a shared library (`ggml-base.dll`).
- On Linux, the linker tolerates unresolved symbols in shared libraries at link time.
- On Windows, MSVC does not permit unresolved symbols in a DLL. Thus, `ggml-base.dll` fails to link with 10 unresolved external symbols referencing `ggml_backend_rpc_*` functions.

### 1.2. Runtime Sampler Crash (`Assertion failed: found`)
When running the built executables, the sampler occasionally crashes with:
```
Assertion failed: found, file ...\src\llama-sampler.cpp, line 1094
```
This is because:
1. The default build configuration under the `Ninja Multi-Config` generator defaults to `Debug` since `CMAKE_BUILD_TYPE` and `CMAKE_CONFIGURATION_TYPES` are not specified at configure time.
2. The original `build.ps1` script overrode `-DCMAKE_CXX_FLAGS_RELEASE="/FS"` and `-DCMAKE_C_FLAGS_RELEASE="/FS"`. This completely wiped out the default CMake compiler flags (such as `/O2` and `/DNDEBUG`).
3. Due to the lack of `/DNDEBUG`, standard C++ assertions (`assert(...)`) remained active in the compiled binaries, causing the server to abort immediately instead of invoking the sampler's safe fallback logic.
4. Without `/O2` optimizations, model execution was slow and prone to subnormal precision issues that trigger the sampler boundaries.

---

## 2. Solution Architecture

The solution patches both the link-time DLL resolution and the configuration compile flags at the script level:

1. **`fix-rpc.cmake`**: A CMake script containing a guarded macro override for `add_library`. When CMake defines the `ggml-base` library target, the macro intercepts the call, compiles `ggml-rpc.cpp` and `transport.cpp` directly into `ggml-base`, and links `ws2_32` (Windows Sockets).
2. **`build.ps1`**: The build script is modified to:
   - Inject `fix-rpc.cmake` via `-DCMAKE_PROJECT_INCLUDE`.
   - Force Release configurations via `-DCMAKE_CONFIGURATION_TYPES=Release` and `-DCMAKE_BUILD_TYPE=Release`.
   - Restore compiler-specific optimization and assertion-disabling flags (`/FS /MD /O2 /Ob2 /DNDEBUG`) escaped properly using PowerShell backticks.

---

## 3. Step-by-Step Instructions to Patch

For an AI agent applying this patch to a fresh repository checkout:

### Step 3.1: Create the CMake Injection File
Write the following contents to `scripts/cuda-windows/fix-rpc.cmake`:

```cmake
# Fix unresolved RPC symbols in ggml-base when GGML_BACKEND_DL=ON on Windows
if (NOT DEFINED FIX_RPC_INCLUDED)
    set(FIX_RPC_INCLUDED TRUE)
    macro(add_library name)
        _add_library(${name} ${ARGN})
        if ("${name}" STREQUAL "ggml-base")
            message(STATUS "[fix-rpc.cmake] Injecting RPC sources into ggml-base target")
            target_sources(ggml-base PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/ggml-rpc/ggml-rpc.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/ggml-rpc/transport.cpp"
            )
            target_link_libraries(ggml-base PRIVATE ws2_32)
        endif()
    endmacro()
endif()
```

### Step 3.2: Modify `build.ps1`
Locate the compiler arguments setup in `scripts/cuda-windows/build.ps1` and apply the following modifications:

1. Calculate the path to `fix-rpc.cmake`:
   ```powershell
   $fixRpcCmake = (Join-Path $CollateralRoot "fix-rpc.cmake").Replace('\', '/')
   ```
2. Update the `$cmakeArgs` array to configure for a true optimized Release build with the injection script:
   ```powershell
   $cmakeArgs = @(
       "-S", ".",
       "-B", "build-cuda-b-bin",
       "-G", "Ninja Multi-Config",
       "-DCMAKE_CONFIGURATION_TYPES=Release",
       "-DCMAKE_BUILD_TYPE=Release",
       "-DCMAKE_CUDA_COMPILER=`"$nvcc`"",
       "-DCMAKE_PROJECT_INCLUDE=`"$fixRpcCmake`"",
       "-DGGML_CUDA=ON",
       "-DGGML_RPC=ON",
       "-DGGML_NATIVE=OFF",
       "-DGGML_BACKEND_DL=ON",
       "-DGGML_CPU_ALL_VARIANTS=ON",
       "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
       "-DGGML_SCHED_MAX_COPIES=4",
       "-DGGML_CUDA_CUB_3DOT2=ON",
       "-DLLAMA_BUILD_SERVER=ON",
       "-DLLAMA_BUILD_TOOLS=ON",
       "-DLLAMA_BUILD_TESTS=OFF",
       "-DLLAMA_BUILD_EXAMPLES=OFF",
       "-DLLAMA_CURL=OFF",
       "-DLLAMA_OPENSSL=OFF",
       "-DCMAKE_CXX_FLAGS_RELEASE=`"/FS /MD /O2 /Ob2 /DNDEBUG`"",
       "-DCMAKE_C_FLAGS_RELEASE=`"/FS /MD /O2 /Ob2 /DNDEBUG`""
   )
   ```

### Step 3.3: Run and Verify the Build
Run the build script with a clean configuration to compile and assemble:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cuda-windows\build.ps1 -Clean
```

Verify that:
1. The compilation finishes successfully with `BUILD_OK`.
2. The compiler command flags printed in `build-cuda-b-bin/compile_commands.json` contain `/FS /MD /O2 /Ob2 /DNDEBUG` instead of debug parameters (`/RTC1`, `-MDd`).
3. Running the server with custom models and parameters completes completions successfully without triggering assertion crashes.
