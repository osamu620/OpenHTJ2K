# `open_htj2k_transcode` — transcoder CLI reference (experimental)

Converts a JPEG 2000 Part 1 (EBCOT/MQ) codestream or JP2 file into an HTJ2K
(Part 15) codestream or JPH file, and an HTJ2K codestream back into a Part 1
codestream, **without decoding the image**. The code-blocks are decoded to their
quantisation indices and the same indices are coded again with the other block
coder. There is no inverse or forward wavelet transform, no re-quantisation and
no colour transform, so the transcoding is fast, adds no loss, and a decoder
reconstructs from the output the same image it reconstructs from the input
(exceptions below).

The direction is that of the input: an HTJ2K input (a CAP marker segment for
Part 15, or a code-block style with the HT bit) becomes Part 1, anything else
becomes HTJ2K. See [HTJ2K to Part 1](#htj2k-to-part-1) for the second
direction.

Full runtime help: `open_htj2k_transcode -h`.

## Synopsis

```bash
open_htj2k_transcode -i <input> -o <output> [options...]
```

- `-i` input: a raw codestream (`.j2c` / `.j2k` / `.jhc`) or a JP2 or JPH file.
  The kind is found from the file's signature, not its extension.
- `-o` output: for a Part 1 input, a raw codestream (`.j2c` / `.jhc`) or a JPH
  file (`.jph`); a `.jp2` output is refused, since an HT codestream does not
  belong in a JP2 file. For an HTJ2K input, a raw Part 1 codestream (`.j2c` /
  `.j2k`); `.jph` and `.jp2` are refused.

## Options

- `-jph_color_space RGB|YCC` (Part 1 input)
  - The colour space a `.jph` file declares when the input is a raw codestream
    with three or four components: `RGB` is sRGB (the default), `YCC` is sYCC,
    for components that are Y, Cb, Cr (the same option as in `open_htj2k_enc`).
    It only sets the label; no sample is changed.
  - Refused for a codestream output, for a JP2 input (its own colour box is
    kept) and for images that do not have three or four components.
- `-partial refine|drop|keep` (Part 1 input)
  - What to do with a code-block that the Part 1 encoder cut part-way through a
    bit-plane (a lossy encode at a rate or with a limited pass count).
    `refine` (default) codes the block at its last complete cleanup plane and
    then the partial plane with HT SigProp and MagRef passes, which is exact.
    `drop` codes it at the last complete plane and drops the partial plane
    (smallest output, lossy). `keep` codes every decoded bit (the
    reconstruction is off by half a step, and the output is much larger).
    Blocks cut after a cleanup pass are exact in all modes.
- `-layers merge|keep`
  - Part 1 input: `merge` (default) writes one quality layer. `keep` keeps the
    layers of the input with placeholder passes, so that the output can be
    transcoded back into a Part 1 codestream with the same layers. See
    [Layers](#layers).
  - HTJ2K input: `keep` (default) keeps the layers of the input, the
    placeholder passes becoming the Part 1 passes they stand for; `merge` writes
    one quality layer. See [HTJ2K to Part 1](#htj2k-to-part-1).
- `-Cmodes 0|<switches joined by '|'>` (HTJ2K input)
  - The Part 1 code-block style switches of the output: `BYPASS`, `RESET`,
    `RESTART`, `CAUSAL`, `ERTERM`, `SEGMARK` (for example
    `-Cmodes "BYPASS|ERTERM"`), or `0` for none. By default those of the input's
    code-block style: CAUSAL for HT code-blocks, all six for the Part 1
    code-blocks of a mixed codestream.
- `-num_threads N`
  - Worker threads for the code-block loop. `0` (default) uses all hardware
    threads, `1` none. The output is identical for every `N`.
- `-copy_input`
  - The library copies the input before it reads it, which takes memory the
    size of the input more. The output is the same. By default the input is
    read where it is.
- `-h` — help.

## Examples

```bash
# A Part 1 file to an HTJ2K codestream
open_htj2k_transcode -i in.j2k -o out.j2c

# A JP2 file to a JPH file; the colour, resolution and XML boxes are kept
open_htj2k_transcode -i in.jp2 -o out.jph

# A raw Part 1 codestream of Y, Cb, Cr components to a JPH file that says so
open_htj2k_transcode -i in.j2c -o out.jph -jph_color_space YCC

# Keep the layers, and go back to Part 1 with the original code-block switches
open_htj2k_transcode -i in.j2k -o ht.j2c -layers keep
open_htj2k_transcode -i ht.j2c -o back.j2k -Cmodes "RESTART|ERTERM"

# An HTJ2K file (codestream or JPH) to a Part 1 codestream with one layer
open_htj2k_transcode -i in.jph -o out.j2k -layers merge
```

The tool prints the sizes and the time, and for a Part 1 input how many
code-blocks were coded exactly or cut.

## What the output is (Part 1 to HTJ2K)

| | Input (Part 1) | Output (HTJ2K) |
|---|---|---|
| Block coder | EBCOT/MQ | HT (`Cmodes` HT, with the input's CAUSAL switch, in COD and every COC) |
| Quality layers | as coded | one; with `-layers keep` those of the input (see [Layers](#layers)) |
| Rsiz | as coded (a Part 1 profile, for example) | 0x4000: a CAP marker segment follows (bit 15 is kept) |
| CAP marker segment | none | added (HTREV when every component is reversible, else HTIRV, which includes mixed filters; RGN when an RGN marker segment is kept; HETEROGENEOUS when a tile-part header has COD, COC, QCD, QCC or RGN) |
| SIZ, QCD, QCC, RGN, COM, CRG, tiling, precincts, progression order, SOP/EPH | | kept |
| COD/COC/QCD/QCC/RGN/COM in tile-part headers | | kept |
| PLT | | written again for the output's packets |
| TLM | | written again (for more than one tile) |
| POC | read in its order | not written; packets are in the COD's progression order |
| PPM | packet headers read from it | not written; packet headers are inline |
| CPF | | written: the input's Rsiz plus one (the profile of the corresponding Part 1 codestream), for a transcode back |
| PLM | | not written |

The output decodes to the same image as the input in every case except these:

- A code-block cut part-way through a bit-plane with `-partial drop` or `keep`
  (see above).
- A *lossy* stream with a region of interest (RGN). The HT decoder places the
  reconstruction point differently for those coefficients, so the result is
  within about 3 levels of 4095 of the Part 1 decode on the conformance
  stream tested (p0_06), not identical. Reversible streams with RGN are exact.

## Size and speed

The output is of the same order as the input: from 63% to 150% of it on the
conformance and generated test streams, from 111% to 126% on the 4K frames
below. The HT block coder is less compact than the MQ coder it replaces, and
the difference shows most on small images and many small precincts. The gain
is in decoding: on one thread the HTJ2K output of those frames decodes 3 to 26
times as fast as the Part 1 input, the lossless ones most.

Measured on an AMD Ryzen 9 9950X (16 cores, 32 hardware threads; Linux, GCC 14,
Release build with AVX2), median of 5 runs. The inputs are two frames coded by
`open_htj2k_enc` with `Cmodes=0`, lossless (`Creversible=yes`) and lossy
(`Qfactor=90`): a 3840×2160 12-bit RGB frame (the image of
`conformance_data/4KuF_H0L1_arri_1bpp.j2c`) and a 4096×2160 16-bit RGB frame
of Elephants Dream. The transcode time is the one `open_htj2k_transcode`
prints (without reading and writing the files); the decode time is the one
`open_htj2k_dec -iter 4` prints, per frame (the image is written once).

| Part 1 input | Input | Output | Transcode | Decode, Part 1 | Decode, HTJ2K |
|---|---:|---:|---:|---:|---:|
| 3840×2160 12-bit, lossless | 6.51 MB | 8.22 MB (126%) | 763 / 261 / 63 ms | 722 / 259 / 70 ms | 66 / 33 / 33 ms |
| 3840×2160 12-bit, `Qfactor=90` | 1.01 MB | 1.14 MB (113%) | 134 / 48 / 16 ms | 131 / 56 / 30 ms | 31 / 20 / 22 ms |
| 4096×2160 16-bit, lossless | 18.4 MB | 20.5 MB (111%) | 1952 / 665 / 118 ms | 1916 / 666 / 136 ms | 73 / 35 / 32 ms |
| 4096×2160 16-bit, `Qfactor=90` | 0.68 MB | 0.76 MB (112%) | 100 / 36 / 13 ms | 100 / 44 / 27 ms | 29 / 20 / 17 ms |

Times are for `-num_threads` 1 / 2 / 32. A transcode takes about as long as a
Part 1 decode of the input: decoding the MQ-coded blocks dominates both. It
scales with the number of code-blocks and threads; the calling thread works
beside the N threads of the pool, so `-num_threads 2` is about three times as
fast as 1. Peak memory (the whole process, 32 threads) was 45 MB for the
lossless 3840×2160 frame and 74 MB for the 4096×2160 one.

## Layers

A Part 1 block is refined layer by layer: each layer adds coding passes. An HT
block is one HT set (a cleanup segment, with an optional refinement segment),
which is decoded whole.

`-layers merge` (the default) writes one quality layer.

`-layers keep` keeps the layers of the input, the way the WG1 guideline on
placeholder passes (*Guideline on Placeholder Passes and Multiple HT Sets in
HTJ2K codestreams*, WG1 N101237, section 6) recommends for a reversible
transcoder. Each packet adds to each code-block the number of coding passes it
added in the input:

- The passes before the block's last Part 1 cleanup pass become *placeholder
  passes* (ISO/IEC 15444-15, B.3). The packet headers count them, but they carry
  no bytes.
- The HT cleanup pass goes in the packet that held that last cleanup pass, and
  the HT SigProp and MagRef passes go in the packets of the Part 1 passes they
  stand for. When those two are in different layers, the refinement segment is
  split between them, and a decode without the MagRef layer reads the SigProp
  pass alone.
- In the first packet that includes a block, a multiple of three placeholder
  passes is replaced by as many more missing MSBs, which leaves at most two
  placeholder passes there, as the guideline recommends. When that would leave
  the packet with no pass of the block, the block keeps the number of missing
  MSBs of the input instead, which the guideline also allows.

The layer boundaries of the input are thus in the packet headers, and the
output can be transcoded back into a Part 1 codestream with its layers. There is
still one HT set per code-block (SINGLEHT). The output decodes to the same image
as with `merge`. It is 0.5% to 4% larger on the streams tested, for the packet
headers of the extra layers (10% with 64×64 tiles). A decode of only the first
layers is not the gradual refinement of the Part 1 file: a code-block appears
only from the layer that holds its last cleanup pass.

`keep` needs `-partial refine` (the default), and refuses a lossy stream with a
region of interest, whose blocks are coded below their last pass. A block whose
passes decode to zero has nothing to code and is left out, as with `merge`.

OpenJPEG (2.5.0 and 2.5.4 tried) does not read placeholder passes: it rejects
`keep` output that has them, as it rejects the guideline's own example files.
The guideline asks every HTJ2K decoder to handle placeholder passes (section 5);
until OpenJPEG does, use `merge` for files that OpenJPEG must read.

## HTJ2K to Part 1

For an HTJ2K input the tool writes a Part 1 codestream. Every code-block is
decoded to its quantisation indices, an HT block with its cleanup, SigProp and
MagRef passes and a Part 1 block of a mixed codestream with its own passes, and
coded again with the Part 1 block coder down to the same coding pass. A
decoder reconstructs the same image from the output as from the input.

- **Missing MSBs.** A Part 1 block's passes start at the cleanup pass of its
  top bit-plane, so the number of missing MSBs is that of the block's indices.
  An HT block may signal more (an HT encoder may turn placeholder passes into
  missing MSBs) or fewer; the passes are numbered from the indices' top
  bit-plane.
- **Layers** (`-layers keep`, the default). Each packet adds to each block the
  passes it added in the input. Placeholder passes become the Part 1 passes
  they stand for. The passes the input does not count (the bit-planes an HT
  encoder signalled as missing MSBs above the data) go into the block's first
  packet, and passes in all-zero bit-planes above the data are dropped. A cut
  inside a codeword segment keeps the shortest prefix of the coded data that
  decodes the passes before it; the last cut is terminated. Every layer
  therefore decodes, in a Part 1 decoder, to what the input's layers decode to.
  An HTJ2K file made from a Part 1 file with `-layers keep` goes back to the
  layers of that file: every code-block of the 22 Part 1 conformance streams
  that `keep` accepts comes back with the same passes in each layer and the
  same missing MSBs, and codestreams of `open_htj2k_enc` come back byte for byte
  when the transcode back is given their code-block switches with `-Cmodes`.
  `-layers merge` writes one layer.
- **Code-block switches.** The output takes those of the input's code-block
  style (bits 0-5), or those of `-Cmodes`. An HT code-block uses only CAUSAL,
  which is kept: it changes the samples a SigProp pass visits, so a layer cut
  after a SigProp pass decodes differently without it. The Part 1 switches an
  earlier Part 1 to HTJ2K transcode dropped can be given back with `-Cmodes`;
  they change the bytes, not the decoded image.
- **Header.** Rsiz comes from the input's CPF marker segment (its value less
  one: the profile of the corresponding Part 1 codestream), or is the input's
  without bit 14. The CAP and CPF marker segments are not written. The other
  marker segments are handled as in the other direction (POC and PPM are not
  written, PLT and TLM are written again).

The output is smaller than the input: 61-96% of it on the HTJ2K conformance
streams, 92% for the median one (MQ coding is more compact than HT). The
frames of [Size and speed](#size-and-speed), coded with HT by
`open_htj2k_enc`, become byte for byte the Part 1 codestreams the encoder makes
with `Cmodes=0` (same machine and timing; peak memory 36 MB and 72 MB for the
lossless ones):

| HTJ2K input | Input | Output | Transcode, 1 / 2 / 32 threads |
|---|---:|---:|---:|
| 3840×2160 12-bit, lossless | 8.22 MB | 6.51 MB (79%) | 867 / 294 / 62 ms |
| 3840×2160 12-bit, `Qfactor=90` | 1.14 MB | 1.01 MB (88%) | 150 / 53 / 13 ms |
| 4096×2160 16-bit, lossless | 20.5 MB | 18.4 MB (90%) | 2033 / 691 / 112 ms |
| 4096×2160 16-bit, `Qfactor=90` | 0.76 MB | 0.68 MB (89%) | 113 / 40 / 14 ms |

All 40 HTJ2K conformance streams (`ds0_ht_*`, `ds1_ht_*` and the mixed
`ds0_hm_*`) transcode. 35 decode to the image of the input, pixel for pixel.
The other five have a region of interest whose code-blocks are coded above
their last bit-plane (lossy, or bounded by the magnitude bound B). This
library's HT and Part 1 decoders reconstruct those coefficients differently
from the same indices: by one level on ds0_ht_03_b11 and ds0_ht_15_b11, and by
up to 590 levels of 4095 in one component of ds0_ht_06, where both decodes are
within the conformance tolerance of p0_06, the Part 1 stream it was made from.

Refused, with a message: an HTJ2K input with more than one HT set per
code-block (MULTIHT), Part 2 DFS or ATK marker segments, a CAP marker segment
with capabilities besides Part 15, and a code-block that would take more than
65535 bytes with the Part 1 block coder.

## Memory and limits

The input and the output are held whole in memory. The tiles are transcoded one
after the other and each is freed before the next, so a tiled image needs about
the input plus the output plus one tile. An image of one tile needs three to four
times the input.

| 6144×6144 RGB, lossless, 43 MB codestream | Peak memory |
|---|---|
| 36 tiles of 1024×1024 | 96 MB |
| one tile | 169 MB |

The input must

- have no Part 2 DFS or ATK marker segments (refused),
- be less than 4 GB (the codestream reader keeps 32-bit positions; refused with
  a message),
- describe tiles whose state stays within the decoder's memory budget: the
  larger of 256 MiB, 512 bytes per codestream byte and 2 bytes per image sample.
  The state of a tile is that of the input tile (its precincts, and the
  code-blocks its packets have data for) and of the output tile, which holds
  every code-block of the tile with two bytes per quality layer. A header that
  declares more (tens of thousands of layers over many small code-blocks, say)
  is refused as not supported.

A tile that has no tile-part in the input is left out of the output, as the
input leaves it out. A codestream with no tile-part at all is refused.

## Interoperability

The output was decoded in full with OpenJPEG 2.5.4 and compared with the Part 1
decode: 43 conformance, generated and JP2 streams are pixel-identical. OpenJPEG's
HT decoder does not support RGN, so the three RGN conformance streams it
cannot read (p0_03, p0_06, p0_15) fail there, not in this library. OpenJPEG
does not read the placeholder passes of `-layers keep` either (see
[Layers](#layers)).

OpenJPEG 2.4 decodes the Part 1 output of every HTJ2K test input (the 40
conformance streams and the round trips). A Part 1 conformance stream taken to
HTJ2K with `-layers keep` and back decodes in OpenJPEG as the original does, on
all 22 that `keep` accepts.

## Library use

```cpp
#include "transcoder.hpp"

std::vector<uint8_t> in = /* a Part 1 codestream or JP2 file, plus 16 readable bytes after it */;
std::vector<uint8_t> out;

open_htj2k::transcode_options opt;
opt.input_padded = true;                       // in.data() + in.size() .. +16 are readable
opt.num_threads  = 0;                          // all hardware threads
opt.partial_plane = open_htj2k::transcode_partial_plane::refine;
opt.layers       = open_htj2k::transcode_layers::merge;  // keep: the input's layers
opt.jph_output   = false;                      // true: a JPH file
open_htj2k::transcode_stats stats;
opt.stats        = &stats;

open_htj2k::transcode_part1_to_htj2k(in.data(), in.size() - 16, out, opt);  // throws std::exception
```

Without `input_padded` the transcoder copies the input first. The
function throws, with a message, on input it does not transcode.

The other direction:

```cpp
open_htj2k::transcode_to_part1_options opt1;
opt1.input_padded     = true;
opt1.layers           = open_htj2k::transcode_layers::keep;  // default; merge: one layer
opt1.code_block_style = -1;  // the input's switches; or e.g. 0x01 | 0x10 (BYPASS | ERTERM)
if (open_htj2k::transcode_input_is_htj2k(in.data(), in.size() - 16))
  open_htj2k::transcode_htj2k_to_part1(in.data(), in.size() - 16, out, opt1);
```

## Tests

`ctest -L transcoder` runs the transcoder tests (decode-compare round trips on
the Part 1 and HTJ2K conformance streams and on streams made with
`open_htj2k_enc`, in both layer modes and both directions; Part 1 to HTJ2K and
back, byte for byte for codestreams of the encoder; marker and colour-space
checks, refused inputs, a fuzzer-found stream). They are built with
the rest of the tests.
