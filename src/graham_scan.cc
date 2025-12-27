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
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief Graham scan implementation.
 */

#include "dgd/graham_scan.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

namespace {

inline Real Ccw(const Vec2r& u, const Vec2r& v, const Vec2r& w) {
  return (v - u).cross(w - u);
}

}  // namespace

int GrahamScan(const std::vector<Vec2r>& pts, std::vector<Vec2r>& vert) {
  vert.clear();
  if (pts.empty()) return 0;

  // Find points with the least y-coordinate, and among those, the one with the
  // least x-coordinate.
  Vec2r p0 = pts[0];
  for (const auto& p : pts) {
    if (p(1) < p0(1) || (p(1) == p0(1) && p(0) < p0(0))) p0 = p;
  }

  // Copy the vector while removing any duplicates of the minimal point.
  std::vector<Vec2r> pts_c;
  pts_c.reserve(pts.size());
  pts_c.push_back(p0);
  for (const auto& p : pts) {
    if ((p - p0).squaredNorm() > kEps * kEps) pts_c.push_back(p);
  }
  const int len = static_cast<int>(pts_c.size());
  if (len == 1) {
    vert = pts_c;
    return 1;
  }

  // Sort points by polar angles with respect to p0; when equal, sort by
  // distance to p0.
  std::sort(pts_c.begin() + 1, pts_c.end(),
            [&p0](const Vec2r& p1, const Vec2r& p2) -> bool {
              const Real cross = Ccw(p0, p1, p2);
              return (std::abs(cross) > kEps)
                         ? cross > Real(0.0)
                         : (p1 - p0).squaredNorm() < (p2 - p0).squaredNorm();
            });

  // Replace collinear points along a polar angle by the farthest point.
  int idx = 1;
  for (int i = 2; i < len; ++i) {
    if (Ccw(p0, pts_c[idx], pts_c[i]) > kEps) {
      pts_c[++idx] = pts_c[i];
    } else {
      pts_c[idx] = pts_c[i];
    }
  }
  pts_c.resize(++idx);

  // Find convex hull based on turning direction.
  vert.reserve(pts_c.size());
  for (const auto& p : pts_c) {
    while (vert.size() > 1 &&
           Ccw(vert.end()[-2], vert.end()[-1], p) <= Real(0.0)) {
      vert.pop_back();
    }
    vert.push_back(p);
  }
  assert(vert.size() >= 2);
  return static_cast<int>(vert.size());
}

}  // namespace dgd
