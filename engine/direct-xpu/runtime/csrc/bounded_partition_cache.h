#pragma once
#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

// Thread/queue-owned cache. Completion and wait are supplied by the native
// executor so the eviction policy can be tested without a GPU or oneDNN.
template <class Key, class Value> class BoundedPartitionCache {
    struct Node { Value value; uint64_t used; };
    std::map<Key, Node> entries_;
    size_t capacity_;
    uint64_t tick_ = 0;
public:
    struct Stats {
        uint64_t hits=0, misses=0, evictions=0, pressure_waits=0;
        uint64_t compile_microseconds=0, peak_entries=0;
    } stats;
    explicit BoundedPartitionCache(size_t capacity): capacity_(capacity) {
        if (!capacity) throw std::invalid_argument("Partition cache capacity must be positive");
    }
    size_t size() const { return entries_.size(); }
    size_t capacity() const { return capacity_; }
    template <class Factory, class Ready, class Wait>
    Value& get(const Key& key, Factory factory, Ready ready, Wait wait) {
        auto found=entries_.find(key);
        if (found!=entries_.end()) {
            ++stats.hits; found->second.used=++tick_; return found->second.value;
        }
        ++stats.misses;
        if (entries_.size()==capacity_) {
            auto victim=entries_.end();
            auto oldest=entries_.end();
            for (auto it=entries_.begin();it!=entries_.end();++it) {
                if (oldest==entries_.end() || it->second.used<oldest->second.used) oldest=it;
                if (ready(it->second.value) && (victim==entries_.end() || it->second.used<victim->second.used)) victim=it;
            }
            if (victim==entries_.end()) {
                victim=oldest; ++stats.pressure_waits;
                // Only a full-cache MISS waits, and only for this victim's
                // completion chain. No hit/normal call synchronizes the queue.
                wait(victim->second.value);
            }
            entries_.erase(victim); ++stats.evictions;
        }
        auto start=std::chrono::steady_clock::now();
        Value value=factory();
        stats.compile_microseconds+=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-start).count();
        auto inserted=entries_.emplace(key,Node{std::move(value),++tick_}).first;
        if (entries_.size()>stats.peak_entries) stats.peak_entries=entries_.size();
        return inserted->second.value;
    }
};
