# `open_htj2k_transcode` — transcoder CLI reference (experimental)

Converts a JPEG 2000 Part 1 (EBCOT/MQ) codestream or JP2 file into an HTJ2K
(Part 15) codestream or JPH file, **without decoding the image**. The Part 1
code-blocks are decoded to their quantisation indices and the same indices are
coded again with the HT block coder. There is no inverse or forward wavelet
transform, no re-quantisation and no colour transform, so the transcoding is
fast, adds no loss, and a decoder reconstructs from the output the same image
it reconstructs from the input (exceptions below).

Full runtime help: `open_htj2k_transcode -h`.

## Synopsis

```bash
open_htj2k_transcode -i <input> -o <output> [options...]
```

- `-i` input: a raw codestream (`.j2c` / `.j2k`) or a JP2 file (`.jp2`). The
  kind is found from the file's signature, not its extension.
- `-o` output: a raw codestream (`.j2c` / `.jhc`) or a JPH file (`.jph`). A
  `.jp2` output is refused: an HT codestream does not belong in a JP2 file.

## Options

- `-jph_color_space RGB|YCC`
  - The colour space a `.jph` file declares when the input is a raw codestream
    with three or four components: `RGB` is sRGB (the default), `YCC` is sYCC,
    for components that are Y, Cb, Cr (the same option as in `open_htj2k_enc`).
    It only sets the label; no sample is changed.
  - Refused for a codestream output, for a JP2 input (its own colour box is
    kept) and for images that do not have three or four components.
- `-partial refine|drop|keep`
  - What to do with a code-block that the Part 1 encoder cut part-way through a
    bit-plane (a lossy encode at a rate or with a limited pass count).
    `refine` (default) codes the block at its last complete cleanup plane and
    then the partial plane with HT SigProp and MagRef passes, which is exact.
    `drop` codes it at the last complete plane and drops the partial plane
    (smallest output, lossy). `keep` codes every decoded bit (the
    reconstruction is off by half a step, and the output is much larger).
    Blocks cut after a cleanup pass are exact in all modes.
- `-layers merge|keep`
  - `merge` (default) writes one quality layer. `keep` keeps the layers of the
    input with placeholder passes, so that the output can be transcoded back
    into a Part 1 codestream with the same layers. See [Layers](#layers).
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
./build/bin/open_htj2k_transcode -i in.j2k -o out.j2c

# A JP2 file to a JPH file; the colour, resolution and XML boxes are kept
./build/bin/open_htj2k_transcode -i in.jp2 -o out.jph

# A raw Part 1 codestream of Y, Cb, Cr components to a JPH file that says so
./build/bin/open_htj2k_transcode -i in.j2c -o out.jph -jph_color_space YCC
```

The tool prints the sizes, how many code-blocks were coded exactly or cut, and
the time.

## What the output is

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

The output is of the same order as the input. On the conformance and generated
test streams it is from 63% to 150% of the input; for ordinary images
(lossless and lossy photographs) 100% to 110%, 107% for a 3840×2160 lossless
frame. The HT block coder is less compact than the MQ coder it replaces, and the
difference shows most on small images and many small precincts. The gain is in decoding: that 4K frame decodes
in 0.12 s from the HTJ2K file against 1.45 s from the Part 1 original (2
threads, one test machine).

Transcoding it took 1.4 s with 2 threads. Time scales with the number of
code-blocks and threads.

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

- use the Part 1 block coder (an HT input is refused),
- have no Part 2 DFS or ATK marker segments (refused),
- be less than 4 GB (the codestream reader keeps 32-bit positions; refused with
  a message).

A tile that has no tile-part in the input is left out of the output, as the
input leaves it out. A codestream with no tile-part at all is refused.

## Interoperability

The output was decoded in full with OpenJPEG 2.5.4 and compared with the Part 1
decode: 43 conformance, generated and JP2 streams are pixel-identical. OpenJPEG's
HT decoder does not support RGN, so the three RGN conformance streams it
cannot read (p0_03, p0_06, p0_15) fail there, not in this library. OpenJPEG
does not read the placeholder passes of `-layers keep` either (see
[Layers](#layers)). OpenJPEG is the only third-party decoder that has been
tried.

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

## Tests

`ctest -L transcoder` runs the transcoder tests (decode-compare round trips on
the conformance streams and on streams made with `open_htj2k_enc`, in both
layer modes, marker and colour-space checks, refused inputs, a fuzzer-found
stream). They are built with
the rest of the tests.
