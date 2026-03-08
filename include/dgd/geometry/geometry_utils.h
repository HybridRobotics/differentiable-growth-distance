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
 * @brief Utility functions for convex set computations.
 */

#ifndef DGD_GEOMETRY_GEOMETRY_UTILS_H_
#define DGD_GEOMETRY_GEOMETRY_UTILS_H_

#include <utility>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

namespace detail {

/**
 * @brief Merges duplicate indices given index hints for a point in a convex set
 * and orders element groups by barycentric weight.
 *
 * Returns the number of unique nonnegative indices with total barycentric
 * weight > eps, or -1 if the total barycentric weight of idx = -1 exceeds eps.
 *
 * @attention The barycentric coordinates must sum to 1.0 and eps must be less
 * than (1.0 / dim).
 *
 * @param[in,out] h   Base point hint.
 * @param[in]     eps Barycentric coordinate tolerance.
 * @return        Number of unique indices with bc > eps; -1 if the total bc of
 *                idx = -1 exceeds eps.
 */
template <int dim>
int MergeIndices(BasePointHint<dim>& h, Real eps);

template <>
inline int MergeIndices<2>(BasePointHint<2>& h, Real eps) {
  auto swap = [&h]() {
    std::swap(h.idx(0), h.idx(1));
    std::swap(h.bc(0), h.bc(1));
  };

  if (h.idx(0) == -1) {
    if ((h.idx(1) == -1) || (h.bc(0) > eps)) return -1;
    swap();
  } else if (h.idx(1) == -1) {
    if (h.bc(1) > eps) return -1;
  } else if (h.idx(0) == h.idx(1)) {
    h.bc(0) += h.bc(1);
    h.idx(1) = -1;
    h.bc(1) = Real(0.0);
  } else if (h.bc(0) <= eps) {
    swap();
  } else if (h.bc(1) > eps) {
    return 2;
  }
  return 1;
}

template <>
inline int MergeIndices<3>(BasePointHint<3>& h, Real eps) {
  auto swap = [&h](int i, int j) {
    std::swap(h.idx(i), h.idx(j));
    std::swap(h.bc(i), h.bc(j));
  };

  int n;
  auto merge = [&h, &n](int i, int j) {
    h.bc(i) += h.bc(j);
    h.idx(j) = -1;
    h.bc(j) = Real(0.0);
    --n;
  };

  n = (h.idx(0) != -1) + (h.idx(1) != -1) + (h.idx(2) != -1);
  if (n == 0) return -1;
  // n >= 1.
  if (h.idx(0) == -1) swap(0, 1 + (h.idx(1) == -1));
  if (n == 1) {
    if (h.bc(1) + h.bc(2) > eps) return -1;
    return 1;
  }
  // n >= 2.
  if (h.idx(1) == -1) swap(1, 2);
  // h.idx[0..n-1] >= 0, h.idx[n..2] == -1.

  // Deduplicate among [0..n-1].
  if (h.idx(0) == h.idx(1)) {
    merge(0, 1);
    if (n == 2) swap(1, 2);
  }
  if (n == 3) {
    if (h.idx(0) == h.idx(2))
      merge(0, 2);
    else if (h.idx(1) == h.idx(2))
      merge(1, 2);
  }
  if (n == 2 && h.idx(0) == h.idx(1)) merge(0, 1);

  Real bc_1 = Real(0.0);
  for (int i = n; i < 3; ++i) bc_1 += h.bc(i);
  if (bc_1 > eps) return -1;

  // Bring bc > eps to the front of [0..n-1].
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (h.bc(i) > eps) swap(count++, i);
  }
  return count;
}

}  // namespace detail

}  // namespace dgd

#endif  // DGD_GEOMETRY_GEOMETRY_UTILS_H_
