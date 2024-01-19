cmake_minimum_required(VERSION 3.13)

set(CMAKE_SYSTEM_NAME Generic)

project (
    open_htj2k
    LANGUAGES CXX
)

set(CMAKE_LIBRARY_OUTput_DIRECTORY ${CMAKE_BINARY_DIR}/html)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html)

add_subdirectory(".." open_htj2k EXCLUDE_FROM_ALL)
add_executable(libopen_htj2k "src/wrapper.cpp")
target_compile_options(libopen_htj2k PRIVATE -DOPENHTJ2K_ENABLE_ARM_NEON -msimd128)
set_target_properties(libopen_htj2k PROPERTIES SUFFIX ".js" LINK_FLAGS "-03 -s WASM=1 -s EXPORT_ES6=1 -s MODULARIZE=1 -s ENVIRONMENT=web -s EXPORTED_FUNCTIONS=[_free,_malloc] -s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,writeArrayToMemory,getValue] -s NO_EXIT_RUNTIME=1 -s ALLOW_MEMORY_GROWTH=1 -sASSERTIONS -sNO_DISABLE_EXCEPTION_CATCHING")
target_link_libraries(libopen_htj2k PRIVATE open_htj2k)