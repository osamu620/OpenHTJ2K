[![CMake](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/cmake.yml)
[![CodeQL](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/osamu620/OpenHTJ2K/actions/workflows/codeql-analysis.yml)
# OpenHTJ2K
OpenHTJ2K is an open source implementation of ITU-T Rec.814 | ISO/IEC 15444-15 (a.k.a. JPEG 2000 Part 15, High-Throughput JPEG 2000; HTJ2K)

# What OpenHTJ2K provides
OpenHTJ2K provides a shared liberary and sample applications having the following functionalities:
- Decoding of ITU-T Rec.800 | ISO/IEC 15444-1 (JPEG 2000 Part 1) or ITU-T Rec.814 | ISO/IEC 15444-15 (JPEG 2000 Part 15.) compliant codestreams
  - fully compliant with conformance testing defined in ITU-T Rec.803 | ISO 15444-4.
- Encoding an image into a codestream/JPH file which is compliant with HTJ2K
  - currently supports only HTJ2K. The optional markers like COC, POC, etc. are not implemented.
  - **Quality control for lossy compression with ***Qfactor*** feature** 

# Requirements
cmake (version 3.14 or later) and C++14 compliant compiler.

# Building
Type the following command. `./` is a root of cloned repository and `${BUILD_DIR}` is a build directory (for example, `../build` or `./build` and so on)

- You can also specify `-DCMAKE_BUILD_TYPE=Debug` or `-DCMAKE_BUILD_TYPE=RelWithDebInfo` to build with debug information.
- You can also specify `-G "Xcode"` to create a project for Xcode.
- You can also specify `-G "Visual Studio 16 2019"` to create a project for Visual Studio 2019.
(see https://cmake.org/cmake/help/v3.14/manual/cmake-generators.7.html#id12)

```
cd ./
cmake -G "Unix Makefiles" -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release
cd  ${BUILD_DIR}
make
```

Then the executables should be found in `${BUILD_DIR}/bin` directory.

# Usage
## Encoder
Only Part 15 compliant encoding is supported.
```bash
./open_htj2k_enc -i inputimage(in PNM/PGX format) or comma separated input images for each color component(in PGM/PGX format) -o output-codestream [options...]
```
### options
- `Stiles=Size`
  - Size of tile "{height, width}"
- `Sorigin=Size`
- `Stile_origin=Size`
- `Clevels=Int`
  - Valid range for number of DWT levels is from 0 to 32 (Default is 5)
- `Creversible=Bool`
  - `yes` for lossless mode, `no` for lossy mode
- `Cblk=Size`
  - Code-block size
- `Cprecincts=Size`
  - Precinct size
- `Cycc=Bool`
  - `yes` to use RGB->YCbCr
- `Corder`
  - Progression order: LRCP, RLCP, RPCL, PCRL, CPRL
- `Cuse_sop=Bool`
- `Cuse_eph=Bool`
- `Qstep=Float`
  - 0.0 < base step size <= 2.0
- `Qguard=Int`
  - 0 to 7 for the number of guard bits 
- `Qderived=Bool`
  - `yes` switches the quantyzation style to **derived** (Default is `no`)
- `Qfactor=Int`
  - 0 to 100 for the quality of the lossyly compressed image
- `-jph_color_space`
  - Color space of input components: RGB, YCC
	- if inputs are represented in YCbCr, use YCC

## Decoder
The both Part 1 and Part 15 compliant decoding are supported.
```bash
./open_htj2k_dec -i codestream -o outputimage [-reduce n]
```
To see a help, use `-h` option.

## Supported file types
### Encoder
- input image formats: .pgm, .ppm, .pgx
- output codestreams: .j2k, .j2c, .jphc (Part 15 codestream), .jph (Part 15 file format)
### Decoder
- input codestreams : .j2k, .j2c, .jphc
- output image formats: .raw, .ppm, .pgm, .pgx
