// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Checks two properties of the shared worker pool that concurrent decoders and encoders rely on:
//   * a push never fails when many tasks are queued at once: the task ring grows past its
//     initial capacity (several instances encoding large images at the same time used to
//     overflow it, and the throw part-way through a batch crashed a worker);
//   * inside a SingleThreadScope, get() returns no pool, so an instance created with one thread
//     runs on the calling thread even when another instance has created the pool.

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "ThreadPool.hpp"
#include "decoder.hpp"
#include "encoder.hpp"

using open_htj2k::ThreadPool;

#if !defined(_WIN32) || defined(OHTJ2K_STATIC)
// A 1024x1024 three-component 8-bit image, encoded losslessly on the calling thread.
static std::vector<uint8_t> encode_test_image() {
  const uint32_t W = 1024, H = 1024;
  open_htj2k::siz_params siz{};
  siz.Xsiz = siz.XTsiz = W;
  siz.Ysiz = siz.YTsiz = H;
  siz.Csiz             = 3;
  siz.Ssiz             = {7, 7, 7};
  siz.XRsiz = siz.YRsiz = {1, 1, 1};
  open_htj2k::cod_params cod{};
  cod.blkwidth = cod.blkheight = 4;
  cod.is_max_precincts         = true;
  cod.number_of_layers         = 1;
  cod.use_color_trafo          = 1;
  cod.dwt_levels               = 5;
  cod.codeblock_style          = 0x40;
  cod.transformation           = 1;
  open_htj2k::qcd_params qcd{};
  qcd.number_of_guardbits = 1;
  qcd.base_step           = 1.0 / 256;
  std::vector<int32_t *> none;
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", none, siz, cod, qcd, 0xFF, false, 0, 1);
  enc.set_output_buffer(out);
  enc.invoke_line_based_stream([](uint32_t y, int32_t **rows, uint16_t nc) {
    for (uint16_t c = 0; c < nc; ++c)
      for (uint32_t x = 0; x < W; ++x) rows[c][x] = static_cast<int32_t>((x * 7 + y * 13 + c * 61) & 255);
  });
  return out;
}

// Parses and decodes; returns the number of tasks the pool's worker threads ran meanwhile.
static uint64_t decode_tasks(open_htj2k::openhtj2k_decoder &dec) {
  const uint64_t before = ThreadPool::worker_tasks_run();
  dec.parse();
  std::vector<uint32_t> w, h;
  std::vector<uint8_t> d;
  std::vector<bool> s;
  dec.invoke_line_based_stream([](uint32_t, int32_t *const *, uint16_t) {}, w, h, d, s);
  return ThreadPool::worker_tasks_run() - before;
}
#endif

int main() {
  int failures = 0;

  // Ring growth: both workers block on the first tasks they take, so every other task stays
  // queued; 100000 of them exceed the initial capacity several times over.
  {
    ThreadPool pool(2);
    const int n = 100000;
    std::atomic<bool> go(false);
    std::atomic<int> done(0);
    std::vector<int> items(static_cast<size_t>(n), 0);
    pool.push_batch(items, [&go, &done](int) {
      return [&go, &done]() {
        while (!go.load()) std::this_thread::yield();
        done.fetch_add(1);
      };
    });
    for (int i = 0; i < 1000; ++i) {
      pool.push([&go, &done]() {
        while (!go.load()) std::this_thread::yield();
        done.fetch_add(1);
      });
    }
    go.store(true);
    while (done.load() < n + 1000) {
      if (!pool.try_run_one()) std::this_thread::yield();
    }
    std::printf("ring growth: %d tasks ran\n", done.load());
  }

  // SingleThreadScope hides the shared pool on this thread only.  The shared pool's state is in
  // static members that a Windows DLL does not export, so this part needs the static library there.
#if !defined(_WIN32) || defined(OHTJ2K_STATIC)
  ThreadPool::retain();
  ThreadPool *shared = ThreadPool::instance(4);
  if (shared == nullptr || ThreadPool::get() != shared) {
    std::printf("FAIL: no shared pool\n");
    ++failures;
  }
  {
    ThreadPool::SingleThreadScope scope(true);
    if (ThreadPool::get() != nullptr) {
      std::printf("FAIL: get() returned the pool inside a SingleThreadScope\n");
      ++failures;
    }
    ThreadPool *seen_elsewhere = nullptr;
    std::thread other([&seen_elsewhere]() { seen_elsewhere = ThreadPool::get(); });
    other.join();
    if (seen_elsewhere != shared) {
      std::printf("FAIL: the scope hid the pool from another thread\n");
      ++failures;
    }
  }
  {
    ThreadPool::SingleThreadScope scope(false);
    if (ThreadPool::get() != shared) {
      std::printf("FAIL: SingleThreadScope(false) hid the pool\n");
      ++failures;
    }
  }
  if (ThreadPool::get() != shared) {
    std::printf("FAIL: get() did not return the pool after the scope ended\n");
    ++failures;
  }
  ThreadPool::release();

  // A thread count beyond MAX_THREADS (here from a library caller's num_threads) is clamped.
  ThreadPool::retain();
  ThreadPool *big = ThreadPool::instance(100000);
  if (big == nullptr || big->num_threads() != ThreadPool::MAX_THREADS) {
    std::printf("FAIL: instance(100000) made %zu threads, want %zu\n", big ? big->num_threads() : 0,
                static_cast<size_t>(ThreadPool::MAX_THREADS));
    ++failures;
  }
  ThreadPool::release();

  // A decoder decoding one codestream after another: the three-argument init() and
  // init_borrow() keep its thread count, init() with num_threads == 1 decodes on the calling
  // thread only.  Counted with the tasks the pool's worker threads run, not timed.  (Up to
  // 0.28.0 the WASM wrapper re-initialised with 1, so every frame after the first ran on one
  // thread; tests/wasm/mt_reinit_threads_check.mjs checks the wrapper itself.)
  {
    const std::vector<uint8_t> j2c = encode_test_image();
    std::vector<uint8_t> padded(j2c);
    padded.resize(j2c.size() + 16);  // init_borrow() reads up to 16 bytes past the end
    open_htj2k::openhtj2k_decoder dec(j2c.data(), j2c.size(), 0, 4);
    struct Step {
      const char *what;
      bool uses_pool;
      uint64_t tasks;
    };
    std::vector<Step> steps;
    steps.push_back({"constructed with 4 threads", true, decode_tasks(dec)});
    dec.init(j2c.data(), j2c.size(), 0);
    steps.push_back({"then init() without a thread count", true, decode_tasks(dec)});
    dec.init_borrow(padded.data(), j2c.size(), 0);
    steps.push_back({"then init_borrow() without a thread count", true, decode_tasks(dec)});
    dec.init(j2c.data(), j2c.size(), 0, 1);
    steps.push_back({"then init() with 1 thread", false, decode_tasks(dec)});
    dec.init(j2c.data(), j2c.size(), 0);
    steps.push_back({"then init() without a thread count (keeps 1)", false, decode_tasks(dec)});
    dec.init(j2c.data(), j2c.size(), 0, 4);
    steps.push_back({"then init() with 4 threads", true, decode_tasks(dec)});
    for (const Step &st : steps) {
      const bool ok = st.uses_pool ? st.tasks > 0 : st.tasks == 0;
      std::printf("%s decoder %s: %llu pool tasks\n", ok ? "ok  " : "FAIL", st.what,
                  static_cast<unsigned long long>(st.tasks));
      if (!ok) ++failures;
    }
  }
#else
  std::printf("single-thread scope and decoder thread count: not checked with the Windows DLL\n");
#endif

  if (failures == 0) std::printf("threadpool_check: all cases passed\n");
  return failures == 0 ? 0 : 1;
}
