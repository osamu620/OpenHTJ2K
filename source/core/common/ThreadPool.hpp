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
  #include <cstdint>
  #include <functional>
  #include <future>
  #include <memory>
  #include <mutex>
  #include <queue>
  #include <thread>
  #include <type_traits>
  #include <unordered_map>

// Thread-local flag: true when the current thread is running inside the pool worker loop.
// Set once in worker() — avoids the hash-map lookup in decode_strip_core's in_worker check.
inline thread_local bool g_in_worker_thread = false;

class ThreadPool {
 public:
  inline explicit ThreadPool(size_t thread_count) : stop(false), thread_count_(thread_count) {
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
   * @brief Push a batch of tasks built from a container in a single lock+notify_all.
   * @param items  Container whose elements are passed one-by-one to @p factory.
   * @param factory  Callable: factory(item) → void() callable.
   *
   * Acquires the task-queue mutex exactly once and notifies all worker threads once,
   * replacing N individual push() calls (each of which locks + notify_one()).
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
        tasks.push(std::function<void()>(factory(item)));
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
  bool try_run_one() {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(tasks_mutex, std::try_to_lock);
      if (!lock.owns_lock() || tasks.empty()) return false;
      task = std::move(tasks.front());
      tasks.pop();
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
  inline void push_task(const F& task) {
    {
      const std::lock_guard<std::mutex> lock(tasks_mutex);

      if (stop) {
        throw std::runtime_error("Cannot schedule new task after shutdown.");
      }

      tasks.push(std::function<void()>(task));
    }

    condition.notify_one();
  }

  static constexpr size_t BATCH = 4;  // tasks drained per lock hold in worker()

  /**
   * @brief A worker function to be assigned to each thread in the pool.
   *
   *  Continuously pops tasks out of the queue and executes them, as long as the atomic variable running is
   * set to true.
   */
  void worker() {
    g_in_worker_thread = true;
    std::function<void()> batch[BATCH];

    for (;;) {
      size_t n = 0;

      {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        condition.wait(lock, [&] { return !tasks.empty() || stop; });

        if (stop && tasks.empty()) {
          return;
        }

        // Drain up to BATCH tasks in one lock hold.
        while (n < BATCH && !tasks.empty()) {
          batch[n++] = std::move(tasks.front());
          tasks.pop();
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
   * @brief A queue of tasks to be executed by the threads.
   */
  std::queue<std::function<void()>> tasks;

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
