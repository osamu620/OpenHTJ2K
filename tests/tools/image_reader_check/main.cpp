// image_reader_check: open_htj2k::image on PGM, PPM and PGX files that are valid, cut short
// or inconsistent.  The files are written to the working directory.
//
// A valid file must give its samples; a file the reader cannot use must be refused with an
// exception, without a hang, a read or a write out of bounds, or memory sized by a header
// alone:
//   - a binary PGM whose first samples are 10, 13 and 32 (white space as characters);
//   - text PGM and PPM files, with comments;
//   - a list of PGM files of different sizes, the second one a text file;
//   - headers that end early (the reader looped for ever);
//   - a header declaring more samples than the file holds;
//   - a header declaring no columns or no rows (every row of 0 x 2000000000 was gone through);
//   - a PPM file in a list (the planes of the list were written past their end);
//   - a maximum value or a bit depth out of range.
// Prints PASS and exits 0, or prints what failed and exits 1.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>
#include "encoder.hpp"

namespace {
int failures = 0;

void fail(const std::string &what) {
  std::printf("FAIL %s\n", what.c_str());
  ++failures;
}

std::string write_file(const std::string &name, const std::string &bytes) {
  FILE *f = std::fopen(name.c_str(), "wb");
  if (f == nullptr || std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
    fail("cannot write " + name);
  }
  if (f != nullptr) std::fclose(f);
  return name;
}

// The samples of every component, in order.
bool read_all(const std::vector<std::string> &names, std::vector<std::vector<int32_t>> &planes,
              std::string &error) {
  planes.clear();
  try {
    open_htj2k::image img(names);
    for (uint16_t c = 0; c < img.get_num_components(); ++c) {
      const size_t n   = static_cast<size_t>(img.get_component_width(c)) * img.get_component_height(c);
      const int32_t *p = img.get_buf(c);
      planes.emplace_back(p, p + n);
    }
  } catch (std::exception &e) {
    error = e.what();
    return false;
  }
  return true;
}

void expect_planes(const char *what, const std::vector<std::string> &names,
                   const std::vector<std::vector<int32_t>> &want) {
  std::vector<std::vector<int32_t>> got;
  std::string error;
  if (!read_all(names, got, error)) {
    fail(std::string(what) + ": refused (" + error + ")");
  } else if (got != want) {
    fail(std::string(what) + ": other samples than the file holds");
  }
}

void expect_refused(const char *what, const std::vector<std::string> &names) {
  std::vector<std::vector<int32_t>> got;
  std::string error;
  if (read_all(names, got, error)) fail(std::string(what) + ": read, but must be refused");
}
}  // namespace

int main() {
  const std::string d = "image_reader_check_";
  // Valid files.
  expect_planes("binary PGM starting with white-space values",
                {write_file(d + "ws.pgm", std::string("P5\n3 2\n255\n") + "\x0a\x0d\x20" + "ABC")},
                {{10, 13, 32, 65, 66, 67}});
  expect_planes("binary PGM, 16 bits",
                {write_file(d + "g16.pgm", std::string("P5 2 1 4095\n") + "\x01\x02\x0f\xff")},
                {{0x0102, 0x0fff}});
  expect_planes("binary PPM", {write_file(d + "c.ppm", "P6\n# a comment\n2 1\n255\nabcdef")},
                {{'a', 'd'}, {'b', 'e'}, {'c', 'f'}});
  expect_planes("text PGM",
                {write_file(d + "t.pgm", "P2\n# a comment\n3 2\n255\n1 2 3\n# another\n40 50 60\n")},
                {{1, 2, 3, 40, 50, 60}});
  expect_planes("text PPM", {write_file(d + "t.ppm", "P3 2 1 255 1 2 3 4 5 6")}, {{1, 4}, {2, 5}, {3, 6}});
  expect_planes("PGX, signed 12 bits, little-endian",
                {write_file(d + "s12.pgx", std::string("PG LM - 12 2 1\n") + "\xff\xff\x34\x02")},
                {{-1, 0x0234}});
  // A list: the second file is larger than the first, and a text file.
  expect_planes(
      "list of a binary and a text PGM",
      {write_file(d + "l0.pgm", "P5 1 1 255\nZ"), write_file(d + "l1.pgm", "P2 2 2 255 7 8 9 10")},
      {{'Z'}, {7, 8, 9, 10}});

  // Headers that end early.
  const char *cut[] = {"P5", "P5\n12", "P5\n12 ", "P5\n4 4\n",   "P6 2",
                       "PG", "PG ML",  "PG ML +", "PG ML + 8 2", "P2 3"};
  for (const char *h : cut) expect_refused(h, {write_file(d + "cut.pgx", h)});
  // More samples declared than the file holds.
  expect_refused("too few samples (PGM)", {write_file(d + "few.pgm", "P5\n100000 100000\n255\nAB")});
  expect_refused("too few samples (PGX)", {write_file(d + "few.pgx", "PG ML + 16 60000 60000\nABCD")});
  expect_refused("too few samples (text PPM)", {write_file(d + "few.ppm", "P3\n40000 40000\n255\n1 2 3")});
  expect_refused("one row short", {write_file(d + "short.pgm", "P5 2 2 255\nABC")});
  expect_refused("no columns (PGM)", {write_file(d + "none.pgm", "P5\n0 2000000000\n255\n")});
  expect_refused("no rows (PGM)", {write_file(d + "none.pgm", "P5 5 0 255\n")});
  expect_refused("no columns (PGX)", {write_file(d + "none.pgx", "PG ML + 8 0 7\n")});
  // A PPM file is the three components of the image, and cannot be part of a list.
  expect_refused("PGM then PPM",
                 {write_file(d + "m0.pgm", "P5 1 1 255\nA"), write_file(d + "m1.ppm", "P6 1 1 255\nABC")});
  expect_refused("PPM then PGM",
                 {write_file(d + "m1.ppm", "P6 1 1 255\nABC"), write_file(d + "m0.pgm", "P5 1 1 255\nA")});
  // Values out of range, and characters that are no digits.
  expect_refused("maximum value 0", {write_file(d + "v.pgm", "P5 1 1 0\nA")});
  expect_refused("maximum value 70000", {write_file(d + "v.pgm", "P5 1 1 70000\nAAAA")});
  expect_refused("PGX depth 80", {write_file(d + "v.pgx", "PG ML + 80 1 1\nAAAAAAAAAA")});
  expect_refused("PGX depth 0", {write_file(d + "v.pgx", "PG ML + 0 1 1\nA")});
  expect_refused("width is not a number", {write_file(d + "v.pgm", "P5 x 1 255\nA")});
  expect_refused("width past 2^31", {write_file(d + "v.pgm", "P5 99999999999 1 255\nA")});

  if (failures == 0) std::printf("PASS\n");
  return failures == 0 ? 0 : 1;
}
