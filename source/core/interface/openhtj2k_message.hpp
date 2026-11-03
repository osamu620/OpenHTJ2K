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

#pragma once
// Windows DLL linkage, as in decoder.hpp and encoder.hpp.
#ifndef OPENHTJ2K_EXPORT
  #if defined(_MSC_VER) && !defined(OHTJ2K_STATIC)
    #if defined(open_htj2k_EXPORTS)
      #define OPENHTJ2K_EXPORT __declspec(dllexport)
    #else
      #define OPENHTJ2K_EXPORT __declspec(dllimport)
    #endif
  #else
    #define OPENHTJ2K_EXPORT
  #endif
#endif

namespace open_htj2k {
// Kind of a message passed to a message handler. Errors are never passed to the handler: they are
// thrown as exceptions derived from std::exception, with the description in what().
enum class message_level : int {
  warning = 0,  // the input or the parameters are questionable; the operation goes on
  info    = 1   // informational, e.g. an experimental feature is in use
};

// Receives the library's warnings and informational messages. `text` is a single message without a
// trailing newline and is valid only during the call. The handler can be called from the library's
// worker threads, and from several threads at once, so it must be thread-safe.
typedef void (*message_handler)(message_level level, const char *text, void *user_data);

// Installs a process-wide message handler; `user_data` is passed back to every call. The library
// writes nothing to stdout or stderr: without a handler (the default, or after passing nullptr)
// messages are discarded.
OPENHTJ2K_EXPORT void set_message_handler(message_handler handler, void *user_data);
}  // namespace open_htj2k
