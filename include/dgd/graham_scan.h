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
int GrahamScan(const std::vector<Vec2r>& pts, std::vector<Vec2r>& vert);

/**
 * @brief Computes the inradius with respect to an interior point given the
 * convex hull vertices and an interior point.
 *
 * @param  vert           Convex hull vertices in CCW order.
 * @param  interior_point A point in the convex hull interior.
 * @return Inradius about the interior point. Inradius is negative when
 *         the interior point is not in the convex hull.
 */
inline Real ComputePolygonInradius(const std::vector<Vec2r>& vert,
                                   const Vec2r& interior_point) {
  if (vert.size() < 3) return 0.0;

  Vec2r t, n, prev = vert.end()[-1];
  Real max = -kInf;
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
