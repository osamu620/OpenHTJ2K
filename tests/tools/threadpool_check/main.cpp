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

using open_htj2k::ThreadPool;

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
#else
  std::printf("single-thread scope: not checked with the Windows DLL\n");
#endif

  if (failures == 0) std::printf("threadpool_check: all cases passed\n");
  return failures == 0 ? 0 : 1;
}
