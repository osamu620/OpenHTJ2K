// Copyright (c) 2021, Aaron Boxer
// Copyright (c) 2022, Osamu Watanabe
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

#ifdef OPENHTJ2K_THREAD
  #pragma once

  #include <atomic>
  #include <cassert>
  #include <cstdint>
  #include <cstring>
  #include <future>
  #include <memory>
  #include <mutex>
  #include <thread>
  #include <type_traits>
  #include <unordered_map>

// Thread-local flag: true when the current thread is running inside the pool worker loop.
// Set once in worker() — avoids the hash-map lookup in decode_strip_core's in_worker check.
inline thread_local bool g_in_worker_thread = false;

// ─── InlineTask ──────────────────────────────────────────────────────────────
// Type-erased callable with fixed inline storage — replaces std::function<void()>
// in the task queue to eliminate heap allocation and virtual-dispatch overhead.
//
// All hot-path lambdas in this codebase capture <= 32 bytes of trivially-copyable
// data (raw pointers + small integers), which hits the fast memcpy path (no
// per-task heap allocation, no destructor call).  Non-trivially-destructible
// captures (e.g. shared_ptr in enqueue()) are supported via an ops_ trampoline.
class InlineTask {
  static constexpr size_t CAPACITY = 32;
  alignas(8) unsigned char storage_[CAPACITY];
  void (*invoke_)(void *) = nullptr;
  // Relocate / destroy trampoline for non-trivially-copyable captures.
  //   ops_(dst, src) with src != nullptr  →  move-construct dst from src, destroy src
  //   ops_(dst, nullptr)                  →  destroy dst
  //   nullptr  →  type is trivially copyable+destructible; use memcpy, skip destroy.
  void (*ops_)(void *, void *) = nullptr;

 public:
  InlineTask() = default;

  template <typename F, typename = typename std::enable_if<
                            !std::is_same<typename std::decay<F>::type, InlineTask>::value>::type>
  InlineTask(F &&f) noexcept(std::is_nothrow_move_constructible<typename std::decay<F>::type>::value) {
    using Fn = typename std::decay<F>::type;
    static_assert(sizeof(Fn) <= CAPACITY, "Lambda capture exceeds InlineTask capacity (32 bytes)");
    static_assert(alignof(Fn) <= 8, "Lambda alignment exceeds InlineTask alignment");
    new (storage_) Fn(std::forward<F>(f));
    invoke_ = [](void *p) { (*static_cast<Fn *>(p))(); };
    // Trivially-copyable + trivially-destructible  →  fast path (no ops_ needed).
    // Otherwise install a relocate/destroy trampoline for correct lifetime management.
    if (!std::is_trivially_copyable<Fn>::value || !std::is_trivially_destructible<Fn>::value) {
      ops_ = [](void *dst, void *src) {
        if (src) {
          new (dst) Fn(std::move(*static_cast<Fn *>(src)));
          static_cast<Fn *>(src)->~Fn();
        } else {
          static_cast<Fn *>(dst)->~Fn();
        }
      };
    }
  }

  ~InlineTask() {
    if (ops_ && invoke_) ops_(storage_, nullptr);
  }

  InlineTask(InlineTask &&o) noexcept : invoke_(o.invoke_), ops_(o.ops_) {
    if (ops_) {
      ops_(storage_, o.storage_);
    } else {
      std::memcpy(storage_, o.storage_, CAPACITY);
    }
    o.invoke_ = nullptr;
  }

  InlineTask &operator=(InlineTask &&o) noexcept {
    if (this != &o) {
      if (ops_ && invoke_) ops_(storage_, nullptr);
      invoke_ = o.invoke_;
      ops_    = o.ops_;
      if (ops_) {
        ops_(storage_, o.storage_);
      } else {
        std::memcpy(storage_, o.storage_, CAPACITY);
      }
      o.invoke_ = nullptr;
    }
    return *this;
  }

  InlineTask(const InlineTask &)            = delete;
  InlineTask &operator=(const InlineTask &) = delete;

  explicit operator bool() const noexcept { return invoke_ != nullptr; }
  void operator()() { invoke_(storage_); }
};

class ThreadPool {
 public:
  inline explicit ThreadPool(size_t thread_count)
      : stop(false), ring_head_(0), ring_tail_(0), thread_count_(thread_count) {
    ring_ = std::unique_ptr<InlineTask[]>(new InlineTask[RING_CAP]);
    threads = std::make_unique<std::thread[]>(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i) {
      threads[i] = std::thread(&ThreadPool::worker, this);
    }
  }

  /**
   * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads.
   *
   */
  inline ~ThreadPool() {
    {
      // Lock task queue to prevent adding a new task.
      std::lock_guard<std::mutex> lock(tasks_mutex);
      stop = true;
    }

    // Wake up all threads so that they may exist
    condition.notify_all();

    for (size_t i = 0; i < thread_count_; ++i) {
      threads[i].join();
    }
  }

  int thread_number(std::thread::id id) {
    if (id_map.find(id) != id_map.end()) return (int)id_map[id];
    return -1;
  }

  size_t num_threads() const { return thread_count_; }

  // Returns true when called from inside a pool worker thread.
  // Uses a thread-local flag set in worker() — no hash map lookup.
  static bool is_worker_thread() { return g_in_worker_thread; }

  #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
  /**
   * @brief Enqueue a function with zero or more arguments and a return value into the task queue,
   * and get a future for its eventual returned value.
   */
  template <typename F, typename... Args,
            typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  #elif ((defined(_MSVC_LANG) && _MSVC_LANG >= 201402L) || __cplusplus >= 201402L)
  template <typename F, typename... Args,
            typename R = typename std::result_of<std::decay_t<F>(std::decay_t<Args>...)>::type>
  #else
  template <typename F, typename... Args,
            typename R = typename std::result_of<typename std::decay<F>::type(
                typename std::decay<Args>::type...)>::type>
  #endif
  std::future<R> enqueue(F&& func, Args&&... args) {
    auto task   = std::make_shared<std::packaged_task<R()>>([func, args...]() { return func(args...); });
    auto future = task->get_future();

    push_task([task]() { (*task)(); });
    return future;
  }

  /**
   * @brief Push a void task directly, without a future (lower overhead than enqueue).
   * Use with an external std::atomic<int> counter for barrier synchronization.
   */
  template <typename F>
  void push(F &&task) {
    push_task(std::forward<F>(task));
  }

  /**
   * @brief Push a batch of tasks built from a container in a single lock hold.
   * @param items  Container whose elements are passed one-by-one to @p factory.
   * @param factory  Callable: factory(item) -> void() callable.
   *
   * Acquires the task-queue mutex exactly once for the whole batch (replacing
   * N individual push() calls each of which would re-acquire the lock), then
   * issues ceil(n/BATCH) targeted notify_one() wakeups — one per worker that
   * is expected to find a non-empty BATCH-sized chunk to drain.  This is
   * cheaper than notify_all() which would wake every idle worker only to have
   * most of them go straight back to sleep.
   *
   * Throws std::runtime_error if the ring buffer cannot accommodate the
   * incoming batch (see RING_CAP for the limit).
   */
  template <typename Container, typename Factory>
  void push_batch(Container &&items, Factory &&factory) {
    const size_t n = items.size();
    if (n == 0) return;
    {
      const std::lock_guard<std::mutex> lock(tasks_mutex);
      if (stop) {
        throw std::runtime_error("Cannot schedule new task after shutdown.");
      }
      for (auto &&item : items) {
        if (ring_size_() >= RING_CAP) {
          throw std::runtime_error(
              "ThreadPool task ring overflow — too many in-flight tasks (raise RING_CAP)");
        }
        ring_[ring_tail_ & RING_MASK] = InlineTask(factory(item));
        ++ring_tail_;
      }
    }
    // Wake only as many workers as actually needed to drain the queue.
    // Each worker drains up to BATCH tasks per lock hold, so ceil(n/BATCH)
    // workers suffice.  notify_one() avoids waking idle workers needlessly.
    const size_t wakeups = std::min((n + BATCH - 1) / BATCH, thread_count_);
    for (size_t i = 0; i < wakeups; ++i)
      condition.notify_one();
  }

  // Lock-free fast path: the singleton is set once during init and cleared on
  // release, so an acquire load is sufficient after the first call.
  static ThreadPool* get() { return singleton_.load(std::memory_order_acquire); }

  // Try to dequeue and execute one pending task from the calling thread.
  // Returns true if a task was executed, false if the queue was empty or locked.
  // Safe to call from any thread (including the main thread) to do useful work
  // instead of spinning in a busy-wait loop.
  //
  // Re-entrancy contract: a worker thread may call push() / push_batch()
  // while running inside another task and then spin_wait() on a counter
  // decremented by the pushed subtasks.  This works because neither
  // push_batch nor try_run_one holds tasks_mutex across task invocation —
  // the popped task is moved out of the ring under the lock and invoked
  // unlocked (see line ~253), so a worker can drain its own subtasks
  // during spin_wait without self-deadlock, and other workers can
  // concurrently pop the same subtasks through the shared ring.
  bool try_run_one() {
    InlineTask task;
    {
      std::unique_lock<std::mutex> lock(tasks_mutex, std::try_to_lock);
      if (!lock.owns_lock() || ring_empty_()) return false;
      task = std::move(ring_[ring_head_ & RING_MASK]);
      ++ring_head_;
    }
    task();
    return true;
  }

  static ThreadPool* instance(size_t numthreads) {
    // Fast path: pool already created.
    auto *p = singleton_.load(std::memory_order_acquire);
    if (p) return p;

    std::unique_lock<std::mutex> lock(singleton_mutex);
    if (!singleton_.load(std::memory_order_relaxed)) {
      const size_t n = numthreads ? numthreads : std::thread::hardware_concurrency();
      // Skip pool creation entirely for single-threaded mode: all callers
      // already guard with `pool && pool->num_threads() > 1`.
      if (n > 1) {
        singleton_.store(new ThreadPool(n), std::memory_order_release);
      }
    }
    return singleton_.load(std::memory_order_relaxed);
  }

  static void release() {
    std::unique_lock<std::mutex> lock(singleton_mutex);
    delete singleton_.load(std::memory_order_relaxed);
    singleton_.store(nullptr, std::memory_order_release);
  }

 private:
  template <typename F>
  inline void push_task(F &&task) {
    {
      const std::lock_guard<std::mutex> lock(tasks_mutex);

      if (stop) {
        throw std::runtime_error("Cannot schedule new task after shutdown.");
      }

      if (ring_size_() >= RING_CAP) {
        throw std::runtime_error(
            "ThreadPool task ring overflow — too many in-flight tasks (raise RING_CAP)");
      }
      ring_[ring_tail_ & RING_MASK] = InlineTask(std::forward<F>(task));
      ++ring_tail_;
    }

    condition.notify_one();
  }

  static constexpr size_t BATCH    = 4;      // tasks drained per lock hold in worker()
  // RING_CAP must be a power of two.  Sized to comfortably hold the worst-case
  // burst from a 4K encode tile (a single precinct can enqueue ~5-8 K codeblock
  // tasks back-to-back via push_batch before workers begin draining).  At
  // sizeof(InlineTask) = 48 bytes, 32768 slots is ~1.5 MB per pool — allocated
  // once at process init.  Producers throw std::runtime_error on overflow
  // (see push_task / push_batch); never silently wrap-around.
  static constexpr size_t RING_CAP = 32768;
  static constexpr size_t RING_MASK = RING_CAP - 1;

  bool ring_empty_() const { return ring_head_ == ring_tail_; }
  size_t ring_size_() const { return ring_tail_ - ring_head_; }

  /**
   * @brief A worker function to be assigned to each thread in the pool.
   *
   *  Continuously pops tasks out of the queue and executes them, as long as the atomic variable running is
   * set to true.
   */
  void worker() {
    g_in_worker_thread = true;
    InlineTask batch[BATCH];

    for (;;) {
      size_t n = 0;

      {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        condition.wait(lock, [&] { return !ring_empty_() || stop; });

        if (stop && ring_empty_()) {
          return;
        }

        // Drain up to BATCH tasks in one lock hold.
        while (n < BATCH && !ring_empty_()) {
          batch[n++] = std::move(ring_[ring_head_ & RING_MASK]);
          ++ring_head_;
        }
      }

      for (size_t i = 0; i < n; ++i) {
        batch[i]();
      }
    }
  }

 private:
  /**
   * @brief A mutex to synchronize access to the task queue by different threads.
   */
  mutable std::mutex tasks_mutex{};

  /**
   * @brief An atomic variable indicating to the workers to keep running.
   *
   * When set to false, the workers permanently stop working.
   */
  std::atomic<bool> stop;

  std::unordered_map<std::thread::id, size_t> id_map;

  /**
   * @brief Pre-allocated ring buffer of tasks (replaces std::queue<std::function>).
   * Eliminates per-task heap allocation from std::deque chunk management and
   * std::function's type-erased storage.
   *
   * ring_head_ and ring_tail_ are cache-line aligned so that the consumer
   * (worker threads incrementing head_) and producer (main thread incrementing
   * tail_) do not false-share the cache line that holds them.  Both indices
   * are still mutated under tasks_mutex, so the alignment is purely a
   * micro-architectural optimization for the lock-acquire path.
   */
  std::unique_ptr<InlineTask[]> ring_;
  alignas(64) size_t ring_head_;  // consumer index (next pop position)
  alignas(64) size_t ring_tail_;  // producer index (next push position)

  /**
   * @brief The number of threads in the pool.
   */
  size_t thread_count_;

  /**
   * @brief A smart pointer to manage the memory allocated for the threads.
   */
  std::unique_ptr<std::thread[]> threads;

  /**
   * @brief A condition variable used to notify worker threads of state changes.
   */
  std::condition_variable condition;

  /**
   * @brief An atomic singleton for the instance (lock-free read via get()).
   */
  static std::atomic<ThreadPool*> singleton_;

  /**
   * @brief A mutex to synchronize access to the instance.
   */
  static std::mutex singleton_mutex;
};

#endif
