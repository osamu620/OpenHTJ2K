// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.

#include "channel_manager.hpp"

#include <cstdio>

namespace open_htj2k {
namespace jpip {

std::string ChannelManager::negotiate_transport(const std::string &cnew_list) {
  std::size_t pos = 0;
  while (pos <= cnew_list.size()) {
    std::size_t comma = cnew_list.find(',', pos);
    if (comma == std::string::npos) comma = cnew_list.size();
    std::size_t b = pos, e = comma;
    while (b < e && (cnew_list[b] == ' ' || cnew_list[b] == '\t')) ++b;
    while (e > b && (cnew_list[e - 1] == ' ' || cnew_list[e - 1] == '\t')) --e;
    if (cnew_list.compare(b, e - b, "http") == 0 && e - b == 4) return "http";
    pos = comma + 1;
  }
  return {};
}

std::string ChannelManager::open() {
  std::lock_guard<std::mutex> lk(mtx_);
  if (channels_.size() >= max_channels_) {
    auto victim = channels_.begin();
    for (auto it = channels_.begin(); it != channels_.end(); ++it) {
      if (it->second.last_used < victim->second.last_used) victim = it;
    }
    channels_.erase(victim);
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "JPH%llu", static_cast<unsigned long long>(next_cid_++));
  std::string cid(buf);
  channels_[cid].last_used = ++use_clock_;
  return cid;
}

bool ChannelManager::snapshot(const std::string &cid, CacheModel *out) {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = channels_.find(cid);
  if (it == channels_.end()) return false;
  it->second.last_used = ++use_clock_;
  if (out) *out = it->second.model;
  return true;
}

void ChannelManager::commit(const std::string &cid, const std::vector<SentBin> &sent) {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = channels_.find(cid);
  if (it == channels_.end()) return;
  it->second.last_used = ++use_clock_;
  for (const auto &b : sent) {
    if (b.complete) {
      it->second.model.mark(b.class_id, b.in_class_id);
    } else {
      it->second.model.mark_partial(b.class_id, b.in_class_id, b.end_bytes);
    }
  }
}

bool ChannelManager::apply_model(const std::string &cid, const std::string &model_str) {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = channels_.find(cid);
  if (it == channels_.end()) return false;
  it->second.last_used = ++use_clock_;
  it->second.model.apply(model_str);
  return true;
}

bool ChannelManager::close(const std::string &cid) {
  std::lock_guard<std::mutex> lk(mtx_);
  return channels_.erase(cid) > 0;
}

std::size_t ChannelManager::size() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return channels_.size();
}

}  // namespace jpip
}  // namespace open_htj2k
