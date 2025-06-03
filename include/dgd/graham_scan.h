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
 * @file graham_scan.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-04-02
 * @brief Graham scan function.
 */

#ifndef DGD_GRAHAM_SCAN_H_
#define DGD_GRAHAM_SCAN_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Convex hull of points in the 2D plane using Graham scan algorithm.
 *
 * See https://en.wikipedia.org/wiki/Graham_scan#Pseudocode
 *
 * @param[in]  pts  Points in the 2D plane.
 * @param[out] vert CCW sorted convex hull vertex vector.
 * @return     Number of points in the convex hull.
 */
inline int GrahamScan(const std::vector<Vec2r>& pts, std::vector<Vec2r>& vert) {
  vert.clear();
  if (pts.empty()) return 0;

  // Find points with the least y-coordinate, and among those, the one with the
  // least x-coordinate.
  Vec2r p0{pts[0]};
  for (const auto& p : pts) {
    if (p(1) < p0(1) || (p(1) == p0(1) && p(0) < p0(0))) p0 = p;
  }

  // Copy the vector while removing any duplicates of the minimal point.
  std::vector<Vec2r> pts_c;
  pts_c.push_back(p0);
  for (const auto& p : pts) {
    if ((p - p0).lpNorm<1>() > kEps) pts_c.push_back(p);
  }
  int len{static_cast<int>(pts_c.size())};
  if (len == 1) {
    vert = pts_c;
    return 1;
  }

  // Sort points by polar angles with respect to p0; when equal, sort by
  // distance to p0.
  auto ccw = [](const Vec2r& u, const Vec2r& v, const Vec2r& w) -> Real {
    return (v - u).cross(w - u);
  };
  std::sort(pts_c.begin() + 1, pts_c.end(),
            [&ccw, &p0](const Vec2r& p1, const Vec2r& p2) -> bool {
              const Real cross = ccw(p0, p1, p2);
              return (std::abs(cross) > kEps)
                         ? cross > 0.0
                         : (p1 - p0).squaredNorm() < (p2 - p0).squaredNorm();
            });

  // Replace collinear points along a polar angle by the farthest point.
  int idx{1};
  for (int i = 2; i < len; ++i) {
    if (ccw(p0, pts_c[idx], pts_c[i]) > kEps) {
      pts_c[++idx] = pts_c[i];
    } else {
      pts_c[idx] = pts_c[i];
    }
  }
  pts_c.resize(++idx);

  // Find convex hull based on turning direction.
  for (const auto& p : pts_c) {
    while (vert.size() > 1 && ccw(vert.end()[-2], vert.end()[-1], p) <= 0.0) {
      vert.pop_back();
    }
    vert.push_back(p);
  }
  assert(vert.size() >= 2);
  return static_cast<int>(vert.size());
}

/**
 * @brief Computes the inradius with respect to an interior point given the
 * convex hull vertices and an interior point.
 *
 * @param  vert           Convex hull vertices in CCW order.
 * @param  interior_point A point in the convex hull interior (can be the
 *                        average of the convex hull vertices).
 * @return Inradius about the interior point. Inradius is negative when
 *         interior_point is not in the convex hull.
 */
inline Real ComputePolygonInradius(const std::vector<Vec2r>& vert,
                                   const Vec2r& interior_point) {
  if (vert.size() < 3) return 0.0;

  Vec2r t, n, prev{vert.end()[-1]};
  Real max{-kInf};
  for (auto it = vert.begin(); it != vert.end(); ++it) {
    t = *it - prev;
    if (t.norm() < kEps) {
      throw std::domain_error("Convex hull contains coincident vertices");
    }
    n = Vec2r(t(1), -t(0)).normalized();
    max = std::max(max, n.dot(interior_point - *it));
    prev = *it;
  }

  return -max;
}

}  // namespace dgd

#endif  // DGD_GRAHAM_SCAN_H_
