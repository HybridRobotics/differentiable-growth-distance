// Copyright 2025 Akshay Thirugnanam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file timer.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-07-10
 * @brief Timer class.
 */

#ifndef DGD_UTILS_TIMER_H_
#define DGD_UTILS_TIMER_H_

#include <chrono>

namespace dgd {

/**
 * @brief Timer class for benchmarking.
 *
 * Adapted from:
 * https://github.com/coal-library/coal/blob/devel/include/coal/timings.h
 */
class Timer {
 public:
  /**
   * @brief Constructs a new Timer object.
   */
  explicit Timer(bool start_on_construction = true);

  /**
   * @brief Restarts the timer from zero, if it is not already running.
   */
  void Start();

  /**
   * @brief Stops the timer if it is running.
   */
  void Stop();

  /**
   * @brief Resumes the timer if it is not already running.
   */
  void Resume();

  /**
   * @brief Returns the elapsed time since the last start in microseconds.
   */
  double Elapsed() const;

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_, end_;
  double elapsed_;
  bool running_;
};

inline Timer::Timer(bool start_on_construction) : running_(false) {
  if (start_on_construction) Start();
}

inline void Timer::Start() {
  if (!running_) {
    running_ = true;
    elapsed_ = 0.0;
    start_ = std::chrono::steady_clock::now();
  }
}

inline void Timer::Stop() {
  if (running_) {
    end_ = std::chrono::steady_clock::now();
    running_ = false;
    elapsed_ +=
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_)
                .count()) *
        1e-3;
  }
}

inline void Timer::Resume() {
  if (!running_) {
    running_ = true;
    start_ = std::chrono::steady_clock::now();
  }
}

inline double Timer::Elapsed() const {
  const auto current = std::chrono::steady_clock::now();

  if (running_) {
    return elapsed_ + static_cast<double>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              current - start_)
                              .count()) *
                          1e-3;
  } else {
    return elapsed_;
  }
}

}  // namespace dgd

#endif  // DGD_UTILS_TIMER_H_
