// Copyright (c) 2021, Aaron Boxer
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


#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <map>
#include <type_traits>
#include <iostream>

class ThreadPool
{
  public:
	ThreadPool(size_t);
	template<class F, class... Args>
	auto enqueue(F&& f, Args&&... args)
		-> std::future<typename std::invoke_result<F, Args...>::type>;
	~ThreadPool();
	int thread_number(std::thread::id id)
	{
		if(id_map.find(id) != id_map.end())
			return (int)id_map[id];
		return -1;
	}
	size_t num_threads()
	{
		return m_num_threads;
	}

	static ThreadPool* get()
	{
		return instance(0);
	}
	static ThreadPool* instance(uint32_t numthreads)
	{
		std::unique_lock<std::mutex> lock(singleton_mutex);
		if(!singleton)
			singleton = new ThreadPool(numthreads ? numthreads : hardware_concurrency());
		return singleton;
	}
	static void release()
	{
		std::unique_lock<std::mutex> lock(singleton_mutex);
		delete singleton;
		singleton = nullptr;
	}
	static uint32_t hardware_concurrency()
	{
		return std::thread::hardware_concurrency();
	}

  private:
	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;
	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop;

	std::map<std::thread::id, size_t> id_map;
	size_t m_num_threads;

	static ThreadPool* singleton;
	static std::mutex singleton_mutex;
};

inline ThreadPool::ThreadPool(size_t threads) : stop(false), m_num_threads(threads)
{
	if(threads == 1)
		return;

	for(size_t i = 0; i < threads; ++i)
		workers.emplace_back([this] {
			for(;;)
			{
				std::function<void()> task;

				{
					std::unique_lock<std::mutex> lock(this->queue_mutex);
					this->condition.wait(lock,
										 [this] { return this->stop || !this->tasks.empty(); });
					if(this->stop && this->tasks.empty())
						return;
					task = std::move(this->tasks.front());
					this->tasks.pop();
				}

				task();
			}
		});
	size_t thread_count = 0;
	for(std::thread& worker : workers)
	{
		id_map[worker.get_id()] = thread_count;
		thread_count++;
	}
}
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
	-> std::future<typename std::invoke_result<F, Args...>::type>
{
	assert(m_num_threads > 1);
	using return_type = typename std::invoke_result<F, Args...>::type;

	auto task = std::make_shared<std::packaged_task<return_type()>>(
		std::bind(std::forward<F>(f), std::forward<Args>(args)...));

	std::future<return_type> res = task->get_future();
	{
		std::unique_lock<std::mutex> lock(queue_mutex);

		// don't allow enqueueing after stopping the pool
		if(stop)
			throw std::runtime_error("enqueue on stopped ThreadPool");

		tasks.emplace([task]() { (*task)(); });
	}
	condition.notify_one();
	return res;
}
inline ThreadPool::~ThreadPool()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for(std::thread& worker : workers)
		worker.join();
}
