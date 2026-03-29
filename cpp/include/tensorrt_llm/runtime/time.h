#pragma once
#include <time.h>

namespace tensorrt_llm::runtime {
class Timer {
  public:
    // a timer class for profiling
    // Reset() will be called during initialization
    // all timing variables will be set 0 in Reset()
    Timer() { reset(); }
    void reset() {
        _count = 0;
        _start = 0;
        _elapsed = 0;
        _paused = true;
    }
    void start() {
        reset();
        resume();
    }
    void pause() {
        if (_paused) {
            return;
        }
        _elapsed += (time_ns() - _start);
        ++_count;
        _paused = true;
    }
    // Resume will get current system time
    void resume() {
        _start = time_ns();
        _paused = false;
    }
    int count() const { return _count; }
    // return elapsed time in us
    double elapsed_us() { return static_cast<double>(_elapsed / 1000.0); }
    // return elapsed time in ms
    double elapsed_ms() { return _elapsed / 1000000.0; }
    // return elapsed time in sec
    double elapsed_sec() { return _elapsed / 1000000000.0; }

  private:
    int _count;
    int64_t _start;
    int64_t _elapsed;
    bool _paused;

    // get us difference between start and now
    int64_t time_ns() {
        struct timespec tm;
        clock_gettime(CLOCK_REALTIME, &tm);
        return (tm.tv_sec) * 1000000000L + tm.tv_nsec;
    }
};
} // namespace tensorrt_llm::runtime