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

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

#include "openhtj2k_message.hpp"
#include "message_internal.hpp"

namespace open_htj2k {
namespace {
std::mutex handler_mutex;  // constant-initialised (constexpr constructor)
message_handler current_handler = nullptr;
void *current_user_data         = nullptr;

std::string format_message(const char *fmt, va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);
  char small[256];
  const int n = vsnprintf(small, sizeof(small), fmt, args_copy);
  va_end(args_copy);
  std::string text;
  if (n < 0) {
    text = fmt;
  } else if (static_cast<size_t>(n) < sizeof(small)) {
    text.assign(small, static_cast<size_t>(n));
  } else {
    text.resize(static_cast<size_t>(n) + 1);
    vsnprintf(&text[0], text.size(), fmt, args);
    text.resize(static_cast<size_t>(n));
  }
  while (!text.empty() && (text[text.size() - 1] == '\n' || text[text.size() - 1] == '\r')) {
    text.erase(text.size() - 1);
  }
  return text;
}

void emit(message_level level, const char *fmt, va_list args) {
  message_handler handler;
  void *user_data;
  {
    std::lock_guard<std::mutex> lock(handler_mutex);
    handler   = current_handler;
    user_data = current_user_data;
  }
  if (handler == nullptr) {
    return;
  }
  const std::string text = format_message(fmt, args);
  handler(level, text.c_str(), user_data);
}
}  // namespace

void set_message_handler(message_handler handler, void *user_data) {
  std::lock_guard<std::mutex> lock(handler_mutex);
  current_handler   = handler;
  current_user_data = user_data;
}

namespace detail {
void warn(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  emit(message_level::warning, fmt, args);
  va_end(args);
}

void info(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  emit(message_level::info, fmt, args);
  va_end(args);
}

std::string format(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::string text = format_message(fmt, args);
  va_end(args);
  return text;
}

void fail(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::string text = format_message(fmt, args);
  va_end(args);
  throw std::runtime_error(text);
}
}  // namespace detail
}  // namespace open_htj2k
