cmake_minimum_required(VERSION 3.13)

project(open_htj2k LANGUAGES CXX DESCRIPTION "Open source implementation of 15444-1 and 15444-15")
option(BUILD_SHARED_LIBS "Shared Libraries" ON)
enable_language(CXX)

#### Compiler settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "${CMAKE_HOST_SYSTEM_PROCESSOR}")

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^[xX]86_64$")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /EHsc /D \"_CRT_SECURE_NO_WARNINGS\"")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /D \"_CRT_SECURE_NO_WARNINGS\"")
	endif()
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} /Od /DDEBUG /ZI")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /Ox")
	set(CMAKE_CXX_FLAGS_RelWithDebInfo "${CMAKE_CXX_FLAGS} /O2 /ZI")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
	if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^[xX]86_64$")
		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Wformat -march=native -mtune=native")
	else()
		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Wformat")
	endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -Wformat -fsanitize=address,undefined -fsanitize-recover=address -fno-omit-frame-pointer")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -flto -fopt-info-vec")
	set(CMAKE_CXX_FLAGS_RelWithDebInfo "${CMAKE_CXX_FLAGS} -O2 -g -fno-omit-frame-pointer -pipe")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -Wformat -fsanitize=address,undefined -fsanitize-recover=address -fno-omit-frame-pointer")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -flto -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize")
	set(CMAKE_CXX_FLAGS_RelWithDebInfo "${CMAKE_CXX_FLAGS} -O2 -g -fno-omit-frame-pointer -pipe")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	message(STATUS "Intel compiler")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -p")
	set(CMAKE_CXX_FLAGS_RelWithDebInfo "${CMAKE_CXX_FLAGS} -O2 -g -parallel")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -parallel -qopenmp -ipo -liomp5")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

#### Source files settings "libopen_htj2k" shared library target
add_library(open_htj2k ${SOURCES})
target_include_directories(open_htj2k PUBLIC source/core/common source/core/codestream source/core/coding source/core/transform source/core/interface)
set_target_properties(open_htj2k PROPERTIES OUTPUT_NAME $<IF:$<CONFIG:Debug>,open_htj2k_d,open_htj2k_R>)

#find_package(OpenMP)
#if(OpenMP_FOUND)
# 	message(STATUS "OpenMP is found.")
#	target_link_libraries(open_htj2k OpenMP::OpenMP_CXX)
#endif()
add_subdirectory(source/core/codestream)
add_subdirectory(source/core/coding)
add_subdirectory(source/core/transform)
add_subdirectory(source/core/interface)
#find_package(OpenCV)
#if (OpenCV_FOUND)
#	message(STATUS "OpenCV is found.")
#	target_compile_definitions(open_htj2k PUBLIC VIEWER)
#	target_include_directories(open_htj2k PUBLIC ${OpenCV_INCLUDE_DIRS})
#	target_link_libraries(open_htj2k ${OpenCV_LIBS})
#endif()

#### decoder target
add_executable(open_htj2k_dec)
target_include_directories(open_htj2k_dec PUBLIC source/core/interface)
add_subdirectory(source/apps/decoder)
set_target_properties(open_htj2k_dec PROPERTIES OUTPUT_NAME $<IF:$<CONFIG:Debug>,open_htj2k_dec_dbg,open_htj2k_dec>)
target_link_libraries(open_htj2k_dec open_htj2k)

#### encoder target
add_executable(open_htj2k_enc)
target_include_directories(open_htj2k_enc PUBLIC source/core/interface)
add_subdirectory(source/apps/encoder)
set_target_properties(open_htj2k_enc PROPERTIES OUTPUT_NAME $<IF:$<CONFIG:Debug>,open_htj2k_enc_dbg,open_htj2k_enc>)
target_link_libraries(open_htj2k_enc open_htj2k)

#### imgcmp target
add_executable(imgcmp)
add_subdirectory(source/apps/imgcmp)
target_include_directories(imgcmp PUBLIC source/apps/imgcmp/)

#### Conformance testing by Ctest
enable_testing()
set(CONFORMANCE_DATA_DIR "${CMAKE_SOURCE_DIR}/conformance_data")
### HT decoding tests
## PROFILE 0
add_test(NAME dec_p0_ht_01_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k -o ht_p0_01_b11.pgx)
add_test(NAME dec_p0_ht_02_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_02_b11.j2k -o ht_p0_02_b11.pgx)
add_test(NAME dec_p0_ht_03_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_03_b11.j2k -o ht_p0_03_b11.pgx)
add_test(NAME dec_p0_ht_03_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_03_b14.j2k -o ht_p0_03_b14.pgx)
add_test(NAME dec_p0_ht_04_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b11.j2k -o ht_p0_04_b11.pgx)
add_test(NAME dec_p0_ht_04_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_04_b12.j2k -o ht_p0_04_b12.pgx)
add_test(NAME dec_p0_ht_05_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b11.j2k -o ht_p0_05_b11.pgx)
add_test(NAME dec_p0_ht_05_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_05_b12.j2k -o ht_p0_05_b12.pgx)
add_test(NAME dec_p0_ht_06_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b11.j2k -o ht_p0_06_b11.pgx)
add_test(NAME dec_p0_ht_06_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b15.j2k -o ht_p0_06_b15.pgx)
add_test(NAME dec_p0_ht_06_18 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_06_b18.j2k -o ht_p0_06_b18.pgx)
add_test(NAME dec_p0_hm_06_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_hm_06_b11.j2k -o hm_p0_06_b11.pgx)
add_test(NAME dec_p0_hm_06_18 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_hm_06_b18.j2k -o hm_p0_06_b18.pgx)
add_test(NAME dec_p0_ht_07_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_07_b11.j2k -o ht_p0_07_b11.pgx)
add_test(NAME dec_p0_ht_07_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_07_b15.j2k -o ht_p0_07_b15.pgx)
add_test(NAME dec_p0_ht_07_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_07_b16.j2k -o ht_p0_07_b16.pgx)
add_test(NAME dec_p0_ht_08_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b11.j2k -o ht_p0_08_b11.pgx -reduce 1)
add_test(NAME dec_p0_ht_08_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b15.j2k -o ht_p0_08_b15.pgx -reduce 1)
add_test(NAME dec_p0_ht_08_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_08_b16.j2k -o ht_p0_08_b16.pgx -reduce 1)
add_test(NAME dec_p0_ht_09_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_09_b11.j2k -o ht_p0_09_b11.pgx)
add_test(NAME dec_p0_ht_10_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_10_b11.j2k -o ht_p0_10_b11.pgx)
add_test(NAME dec_p0_ht_11_10 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_11_b10.j2k -o ht_p0_11_b10.pgx)
add_test(NAME dec_p0_ht_12_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_12_b11.j2k -o ht_p0_12_b11.pgx)
add_test(NAME dec_p0_ht_14_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_14_b11.j2k -o ht_p0_14_b11.pgx)
add_test(NAME dec_p0_ht_15_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_15_b11.j2k -o ht_p0_15_b11.pgx)
add_test(NAME dec_p0_ht_15_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_15_b14.j2k -o ht_p0_15_b14.pgx)
add_test(NAME dec_p0_hm_15_8 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_hm_15_b8.j2k -o hm_p0_15_b8.pgx)
add_test(NAME dec_p0_ht_16_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds0_ht_16_b11.j2k -o ht_p0_16_b11.pgx)
## PROFILE 1
add_test(NAME dec_p1_ht_01_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b11.j2k -o ht_p1_01_b11.pgx)
add_test(NAME dec_p1_ht_01_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_01_b12.j2k -o ht_p1_01_b12.pgx)
add_test(NAME dec_p1_ht_02_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b11.j2k -o ht_p1_02_b11.pgx)
add_test(NAME dec_p1_ht_02_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_02_b12.j2k -o ht_p1_02_b12.pgx)
add_test(NAME dec_p1_ht_03_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b11.j2k -o ht_p1_03_b11.pgx)
add_test(NAME dec_p1_ht_03_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_03_b12.j2k -o ht_p1_03_b12.pgx)
add_test(NAME dec_p1_ht_04_9 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_04_b9.j2k -o ht_p1_04_b9.pgx)
add_test(NAME dec_p1_ht_05_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_05_b11.j2k -o ht_p1_05_b11.pgx)
add_test(NAME dec_p1_ht_06_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_06_b11.j2k -o ht_p1_06_b11.pgx)
add_test(NAME dec_p1_ht_07_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ds1_ht_07_b11.j2k -o ht_p1_07_b11.pgx)
### Conformance tests for HT
## PROFILE 0
# 1
add_test(NAME comp_p0_ht_01_11 COMMAND imgcmp ht_p0_01_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_01-0.pgx 0 0)
# 2
add_test(NAME comp_p0_ht_02_11 COMMAND imgcmp ht_p0_02_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_02-0.pgx 1 0.001)
# 3
add_test(NAME comp_p0_ht_03_11 COMMAND imgcmp ht_p0_03_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 17 0.15)
add_test(NAME comp_p0_ht_03_14 COMMAND imgcmp ht_p0_03_b14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 0 0)
# 4
add_test(NAME comp_p0_ht_04_11r COMMAND imgcmp ht_p0_04_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 7 0.876)
add_test(NAME comp_p0_ht_04_11g COMMAND imgcmp ht_p0_04_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 6 0.726)
add_test(NAME comp_p0_ht_04_11b COMMAND imgcmp ht_p0_04_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 8 1.170)
add_test(NAME comp_p0_ht_04_12r COMMAND imgcmp ht_p0_04_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 5 0.776)
add_test(NAME comp_p0_ht_04_12g COMMAND imgcmp ht_p0_04_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 4 0.626)
add_test(NAME comp_p0_ht_04_12b COMMAND imgcmp ht_p0_04_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 6 1.070)
# 5
add_test(NAME comp_p0_ht_05_11a COMMAND imgcmp ht_p0_05_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.302)
add_test(NAME comp_p0_ht_05_11b COMMAND imgcmp ht_p0_05_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 3 0.308)
add_test(NAME comp_p0_ht_05_11c COMMAND imgcmp ht_p0_05_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 3 0.270)
add_test(NAME comp_p0_ht_05_11d COMMAND imgcmp ht_p0_05_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 2 0.001)
add_test(NAME comp_p0_ht_05_12a COMMAND imgcmp ht_p0_05_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.302)
add_test(NAME comp_p0_ht_05_12b COMMAND imgcmp ht_p0_05_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 2 0.307)
add_test(NAME comp_p0_ht_05_12c COMMAND imgcmp ht_p0_05_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 2 0.269)
add_test(NAME comp_p0_ht_05_12d COMMAND imgcmp ht_p0_05_b12_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 0 0)
# 6
add_test(NAME comp_p0_ht_06_11a COMMAND imgcmp ht_p0_06_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 3135 86287)
add_test(NAME comp_p0_ht_06_11b COMMAND imgcmp ht_p0_06_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_ht_06_11c COMMAND imgcmp ht_p0_06_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_ht_06_11d COMMAND imgcmp ht_p0_06_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 200 2000)
add_test(NAME comp_p0_ht_06_15a COMMAND imgcmp ht_p0_06_b15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME comp_p0_ht_06_15b COMMAND imgcmp ht_p0_06_b15_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_ht_06_15c COMMAND imgcmp ht_p0_06_b15_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_ht_06_15d COMMAND imgcmp ht_p0_06_b15_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 12 10.0)
add_test(NAME comp_p0_ht_06_18a COMMAND imgcmp ht_p0_06_b18_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME comp_p0_ht_06_18b COMMAND imgcmp ht_p0_06_b18_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_ht_06_18c COMMAND imgcmp ht_p0_06_b18_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_ht_06_18d COMMAND imgcmp ht_p0_06_b18_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
add_test(NAME comp_p0_hm_06_11a COMMAND imgcmp hm_p0_06_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME comp_p0_hm_06_11b COMMAND imgcmp hm_p0_06_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_hm_06_11c COMMAND imgcmp hm_p0_06_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_hm_06_11d COMMAND imgcmp hm_p0_06_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 200 2000)
add_test(NAME comp_p0_hm_06_18a COMMAND imgcmp hm_p0_06_b18_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME comp_p0_hm_06_18b COMMAND imgcmp hm_p0_06_b18_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_hm_06_18c COMMAND imgcmp hm_p0_06_b18_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_hm_06_18d COMMAND imgcmp hm_p0_06_b18_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
# 7
add_test(NAME comp_p0_ht_07_11r COMMAND imgcmp ht_p0_07_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 40 25.0)
add_test(NAME comp_p0_ht_07_11g COMMAND imgcmp ht_p0_07_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 40 25.0)
add_test(NAME comp_p0_ht_07_11b COMMAND imgcmp ht_p0_07_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 50 25.0)
add_test(NAME comp_p0_ht_07_15r COMMAND imgcmp ht_p0_07_b15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 2 0.075)
add_test(NAME comp_p0_ht_07_15g COMMAND imgcmp ht_p0_07_b15_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 2 0.05)
add_test(NAME comp_p0_ht_07_15b COMMAND imgcmp ht_p0_07_b15_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 2 0.075)
add_test(NAME comp_p0_ht_07_16r COMMAND imgcmp ht_p0_07_b16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 0 0)
add_test(NAME comp_p0_ht_07_16g COMMAND imgcmp ht_p0_07_b16_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 0 0)
add_test(NAME comp_p0_ht_07_16b COMMAND imgcmp ht_p0_07_b16_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 0 0)
# 8
add_test(NAME comp_p0_ht_08_11r COMMAND imgcmp ht_p0_08_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 40 45.0)
add_test(NAME comp_p0_ht_08_11g COMMAND imgcmp ht_p0_08_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 40 30.0)
add_test(NAME comp_p0_ht_08_11b COMMAND imgcmp ht_p0_08_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 40 45.0)
add_test(NAME comp_p0_ht_08_15r COMMAND imgcmp ht_p0_08_b15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME comp_p0_ht_08_15g COMMAND imgcmp ht_p0_08_b15_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 1 0.001)
add_test(NAME comp_p0_ht_08_15b COMMAND imgcmp ht_p0_08_b15_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
add_test(NAME comp_p0_ht_08_16r COMMAND imgcmp ht_p0_08_b16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME comp_p0_ht_08_16g COMMAND imgcmp ht_p0_08_b16_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 0 0)
add_test(NAME comp_p0_ht_08_16b COMMAND imgcmp ht_p0_08_b16_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
# 9
add_test(NAME comp_p0_ht_09_11 COMMAND imgcmp ht_p0_09_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_09-0.pgx 0 0)
# 10
add_test(NAME comp_p0_ht_10_11r COMMAND imgcmp ht_p0_10_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-0.pgx 0 0)
add_test(NAME comp_p0_ht_10_11g COMMAND imgcmp ht_p0_10_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-1.pgx 0 0)
add_test(NAME comp_p0_ht_10_11b COMMAND imgcmp ht_p0_10_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-2.pgx 0 0)
# 11
add_test(NAME comp_p0_ht_11_10 COMMAND imgcmp ht_p0_11_b10_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_11-0.pgx 0 0)
# 12
add_test(NAME comp_p0_ht_12_11 COMMAND imgcmp ht_p0_12_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_12-0.pgx 0 0)
# 14
add_test(NAME comp_p0_ht_14_11r COMMAND imgcmp ht_p0_14_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-0.pgx 0 0)
add_test(NAME comp_p0_ht_14_11g COMMAND imgcmp ht_p0_14_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-1.pgx 0 0)
add_test(NAME comp_p0_ht_14_11b COMMAND imgcmp ht_p0_14_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-2.pgx 0 0)
# 15
add_test(NAME comp_p0_ht_15_11 COMMAND imgcmp ht_p0_15_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 17 0.15)
add_test(NAME comp_p0_ht_15_14 COMMAND imgcmp ht_p0_15_b14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
add_test(NAME comp_p0_hm_15_8 COMMAND imgcmp hm_p0_15_b8_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
# 16
add_test(NAME comp_p0_ht_16_11 COMMAND imgcmp ht_p0_16_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_16-0.pgx 0 0)
## PROFILE 1
# 1
add_test(NAME comp_p1_ht_01_11 COMMAND imgcmp ht_p1_01_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 1 0.001)
add_test(NAME comp_p1_ht_01_12 COMMAND imgcmp ht_p1_01_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 0 0)
# 2
add_test(NAME comp_p1_ht_02_11r COMMAND imgcmp ht_p1_02_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 7 0.865)
add_test(NAME comp_p1_ht_02_11g COMMAND imgcmp ht_p1_02_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 6 0.716)
add_test(NAME comp_p1_ht_02_11b COMMAND imgcmp ht_p1_02_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 8 1.151)
add_test(NAME comp_p1_ht_02_12r COMMAND imgcmp ht_p1_02_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 5 0.765)
add_test(NAME comp_p1_ht_02_12g COMMAND imgcmp ht_p1_02_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 4 0.616)
add_test(NAME comp_p1_ht_02_12b COMMAND imgcmp ht_p1_02_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 6 1.051)
# 3
add_test(NAME comp_p1_ht_03_11a COMMAND imgcmp ht_p1_03_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.300)
add_test(NAME comp_p1_ht_03_11b COMMAND imgcmp ht_p1_03_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 3 0.24)
add_test(NAME comp_p1_ht_03_11c COMMAND imgcmp ht_p1_03_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 3 0.25)
add_test(NAME comp_p1_ht_03_11d COMMAND imgcmp ht_p1_03_b11_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 2 0.001)
add_test(NAME comp_p1_ht_03_12a COMMAND imgcmp ht_p1_03_b12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.300)
add_test(NAME comp_p1_ht_03_12b COMMAND imgcmp ht_p1_03_b12_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 2 0.210)
set_tests_properties(comp_p1_ht_03_12b  PROPERTIES WILL_FAIL true)
add_test(NAME comp_p1_ht_03_12c COMMAND imgcmp ht_p1_03_b12_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 1 0.200)
set_tests_properties(comp_p1_ht_03_12c  PROPERTIES WILL_FAIL true)
add_test(NAME comp_p1_ht_03_12d COMMAND imgcmp ht_p1_03_b12_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 0 0)
# 4
add_test(NAME comp_p1_ht_04_9 COMMAND imgcmp ht_p1_04_b9_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_04-0.pgx 624 3080)
# 5
add_test(NAME comp_p1_ht_05_11r COMMAND imgcmp ht_p1_05_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-0.pgx 40 8.458)
add_test(NAME comp_p1_ht_05_11g COMMAND imgcmp ht_p1_05_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-1.pgx 40 9.716)
add_test(NAME comp_p1_ht_05_11b COMMAND imgcmp ht_p1_05_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-2.pgx 40 10.154)
# 6
add_test(NAME comp_p1_ht_06_11r COMMAND imgcmp ht_p1_06_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-0.pgx 2 0.600)
add_test(NAME comp_p1_ht_06_11g COMMAND imgcmp ht_p1_06_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-1.pgx 2 0.600)
add_test(NAME comp_p1_ht_06_11b COMMAND imgcmp ht_p1_06_b11_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-2.pgx 2 0.600)
# 7
add_test(NAME comp_p1_ht_07_11a COMMAND imgcmp ht_p1_07_b11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-0.pgx 0 0)
add_test(NAME comp_p1_ht_07_11b COMMAND imgcmp ht_p1_07_b11_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-1.pgx 0 0)
### Part 1 decoding tests
## PROFILE 0
add_test(NAME  dec_p0_01 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_01.j2k -o p0_01.pgx)
add_test(NAME  dec_p0_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_02.j2k -o p0_02.pgx)
add_test(NAME  dec_p0_03 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_03.j2k -o p0_03.pgx)
add_test(NAME  dec_p0_04 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_04.j2k -o p0_04.pgx)
add_test(NAME  dec_p0_05 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_05.j2k -o p0_05.pgx)
add_test(NAME  dec_p0_06 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_06.j2k -o p0_06.pgx)
add_test(NAME  dec_p0_07 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_07.j2k -o p0_07.pgx)
add_test(NAME  dec_p0_08 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_08.j2k -o p0_08.pgx -reduce 1)
add_test(NAME  dec_p0_09 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_09.j2k -o p0_09.pgx)
add_test(NAME  dec_p0_10 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_10.j2k -o p0_10.pgx)
add_test(NAME  dec_p0_11 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_11.j2k -o p0_11.pgx)
add_test(NAME  dec_p0_12 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_12.j2k -o p0_12.pgx)
add_test(NAME  dec_p0_14 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_14.j2k -o p0_14.pgx)
add_test(NAME  dec_p0_15 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_15.j2k -o p0_15.pgx)
add_test(NAME  dec_p0_16 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p0_16.j2k -o p0_16.pgx)
## PROFILE 1
add_test(NAME  dec_p1_01 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_01.j2k -o p1_01.pgx)
add_test(NAME  dec_p1_02 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_02.j2k -o p1_02.pgx)
add_test(NAME  dec_p1_03 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_03.j2k -o p1_03.pgx)
add_test(NAME  dec_p1_04 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_04.j2k -o p1_04.pgx)
add_test(NAME  dec_p1_05 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_05.j2k -o p1_05.pgx)
add_test(NAME  dec_p1_06 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_06.j2k -o p1_06.pgx)
add_test(NAME  dec_p1_07 COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/p1_07.j2k -o p1_07.pgx)
### Conformance test for Part 1
## PROFILE 0
# 1
add_test(NAME comp_p0_01 COMMAND imgcmp p0_01_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_01-0.pgx 0 0)
# 2
add_test(NAME comp_p0_02 COMMAND imgcmp p0_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_02-0.pgx 0 0)
# 3
add_test(NAME comp_p0_03 COMMAND imgcmp p0_03_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_03-0.pgx 0 0)
# 4
add_test(NAME comp_p0_04_r COMMAND imgcmp p0_04_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-0.pgx 5 0.776)
add_test(NAME comp_p0_04_g COMMAND imgcmp p0_04_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-1.pgx 4 0.626)
add_test(NAME comp_p0_04_b COMMAND imgcmp p0_04_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_04-2.pgx 6 1.070)
# 5
add_test(NAME comp_p0_05_a COMMAND imgcmp p0_05_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-0.pgx 2 0.302)
add_test(NAME comp_p0_05_b COMMAND imgcmp p0_05_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-1.pgx 2 0.307)
add_test(NAME comp_p0_05_c COMMAND imgcmp p0_05_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-2.pgx 2 0.269)
add_test(NAME comp_p0_05_d COMMAND imgcmp p0_05_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_05-3.pgx 0 0)
# 6
add_test(NAME comp_p0_06_a COMMAND imgcmp p0_06_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-0.pgx 635 11287)
add_test(NAME comp_p0_06_b COMMAND imgcmp p0_06_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-1.pgx 403 6124)
add_test(NAME comp_p0_06_c COMMAND imgcmp p0_06_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-2.pgx 378 3968)
add_test(NAME comp_p0_06_d COMMAND imgcmp p0_06_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_06-3.pgx 0 0)
# 7
add_test(NAME comp_p0_07_r COMMAND imgcmp p0_07_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-0.pgx 0 0)
add_test(NAME comp_p0_07_g COMMAND imgcmp p0_07_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-1.pgx 0 0)
add_test(NAME comp_p0_07_b COMMAND imgcmp p0_07_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_07-2.pgx 0 0)
# 8
add_test(NAME comp_p0_08_r COMMAND imgcmp p0_08_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-0.pgx 0 0)
add_test(NAME comp_p0_08_g COMMAND imgcmp p0_08_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-1.pgx 0 0)
add_test(NAME comp_p0_08_b COMMAND imgcmp p0_08_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_08-2.pgx 0 0)
# 9
add_test(NAME comp_p0_09 COMMAND imgcmp p0_09_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_09-0.pgx 0 0)
# 10
add_test(NAME comp_p0_10_r COMMAND imgcmp p0_10_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-0.pgx 0 0)
add_test(NAME comp_p0_10_g COMMAND imgcmp p0_10_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-1.pgx 0 0)
add_test(NAME comp_p0_10_b COMMAND imgcmp p0_10_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_10-2.pgx 0 0)
# 11
add_test(NAME comp_p0_11 COMMAND imgcmp p0_11_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_11-0.pgx 0 0)
# 12
add_test(NAME comp_p0_12 COMMAND imgcmp p0_12_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_12-0.pgx 0 0)
# 14
add_test(NAME comp_p0_14_r COMMAND imgcmp p0_14_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-0.pgx 0 0)
add_test(NAME comp_p0_14_g COMMAND imgcmp p0_14_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-1.pgx 0 0)
add_test(NAME comp_p0_14_b COMMAND imgcmp p0_14_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_14-2.pgx 0 0)
# 15
add_test(NAME comp_p0_15 COMMAND imgcmp p0_15_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_15-0.pgx 0 0)
# 16
add_test(NAME comp_p0_16 COMMAND imgcmp p0_16_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p0_16-0.pgx 0 0)
## PROFILE 1
# 1
add_test(NAME comp_p1_01 COMMAND imgcmp p1_01_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_01-0.pgx 0 0)
# 2
add_test(NAME comp_p1_02_r COMMAND imgcmp p1_02_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-0.pgx 5 0.765)
add_test(NAME comp_p1_02_g COMMAND imgcmp p1_02_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-1.pgx 4 0.616)
add_test(NAME comp_p1_02_b COMMAND imgcmp p1_02_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_02-2.pgx 6 1.051)
# 3
add_test(NAME comp_p1_03_a COMMAND imgcmp p1_03_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-0.pgx 2 0.300)
add_test(NAME comp_p1_03_b COMMAND imgcmp p1_03_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-1.pgx 2 0.210)
set_tests_properties(comp_p1_03_b PROPERTIES WILL_FAIL true)
add_test(NAME comp_p1_03_c COMMAND imgcmp p1_03_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-2.pgx 1 0.200)
set_tests_properties(comp_p1_03_c PROPERTIES WILL_FAIL true)
add_test(NAME comp_p1_03_d COMMAND imgcmp p1_03_03.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_03-3.pgx 0 0)
# 4
add_test(NAME comp_p1_04 COMMAND imgcmp p1_04_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_04-0.pgx 624 3080)
# 5
add_test(NAME comp_p1_05_r COMMAND imgcmp p1_05_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-0.pgx 40 8.458)
add_test(NAME comp_p1_05_g COMMAND imgcmp p1_05_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-1.pgx 40 9.716)
add_test(NAME comp_p1_05_b COMMAND imgcmp p1_05_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_05-2.pgx 40 10.154)
# 6
add_test(NAME comp_p1_06_r COMMAND imgcmp p1_06_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-0.pgx 2 0.600)
add_test(NAME comp_p1_06_g COMMAND imgcmp p1_06_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-1.pgx 2 0.600)
add_test(NAME comp_p1_06_b COMMAND imgcmp p1_06_02.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_06-2.pgx 2 0.600)
# 7
add_test(NAME comp_p1_07_a COMMAND imgcmp p1_07_00.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-0.pgx 0 0)
add_test(NAME comp_p1_07_b COMMAND imgcmp p1_07_01.pgx ${CONFORMANCE_DATA_DIR}/references/c1p1_07-1.pgx 0 0)