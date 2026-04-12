// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

#include "cli.hpp"

namespace open_htj2k::rtp_recv {

// v1 single-threaded main loop (--threading=off).
int run_receiver_single_threaded(const CliOptions& opts);

}  // namespace open_htj2k::rtp_recv
