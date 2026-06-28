set(TARGET_IMPL llama-pipeline-profiler-impl)

add_library(${TARGET_IMPL}
    llama-pipeline-profiler.cpp
    pipeline-jsonl.cpp
    pipeline-trace-parse.cpp
    pipeline-diagnose.cpp
    pipeline-trace-sample.cpp
    pipeline-regression.cpp
    pipeline-rpc-validate.cpp
    pipeline-gpu-telemetry.cpp
)
set_target_properties(${TARGET_IMPL} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET_IMPL} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_IMPL} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})
if (GGML_RPC AND NOT GGML_BACKEND_DL)
    target_link_libraries(${TARGET_IMPL} PUBLIC ggml-rpc)
endif()
target_compile_features(${TARGET_IMPL} PUBLIC cxx_std_17)

set(TARGET llama-pipeline-profiler)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE ${TARGET_IMPL})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()