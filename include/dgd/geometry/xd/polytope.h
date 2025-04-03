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
 * @file polytope.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 2D/3D polytope class.
 */

#ifndef DGD_GEOMETRY_XD_POLYTOPE_H_
#define DGD_GEOMETRY_XD_POLYTOPE_H_

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief 2D/3D convex polytope class.
 *
 * @tparam dim Dimension of the polytope.
 */
template <int dim>
class Polytope : public ConvexSet<dim> {
 public:
  /**
   * @brief Constructs a Polytope object.
   *
   * @note The polytope must contain the origin in its interior.
   *
   * @param pts      Vector consisting of n dim-dimensional vertices.
   * @param margin   Safety margin.
   * @param inradius Polytope inradius.
   */
  Polytope(const std::vector<Vecf<dim>>& pts, Real margin, Real inradius);

  ~Polytope() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

 private:
  const std::vector<Vecf<dim>> pts_; /**< Polytope vertices. */
  const Real margin_;                /**< Safety margin. */
};

template <int dim>
inline Polytope<dim>::Polytope(const std::vector<Vecf<dim>>& pts, Real margin,
                               Real inradius)
    : ConvexSet<dim>(margin + inradius), pts_(pts), margin_(margin) {
  static_assert((dim == 2) || (dim == 3),
                "Incompatible dimension (not 2 or 3)!");
  assert(margin >= Real(0.0));

  const int num_pts{static_cast<int>(pts.size())};
  assert(num_pts >= dim + 1);

  MatXf<dim> aff_pts(dim, num_pts - 1);
  for (int i = 1; i < num_pts; ++i) aff_pts.col(i - 1) = pts[i] - pts[0];
  const Eigen::ColPivHouseholderQR<MatXf<dim>> qr(aff_pts);
  const int rank{static_cast<int>(qr.rank())};
  assert(rank == dim);
}

template <int dim>
inline Real Polytope<dim>::SupportFunction(const Vecf<dim>& n,
                                           Vecf<dim>& sp) const {
  int idx{0};
  Real s{0.0}, sv{n.dot(pts_[0])};
  for (int i = 1; i < static_cast<int>(pts_.size()); ++i) {
    s = n.dot(pts_[i]);
    if (s > sv) {
      idx = i;
      sv = s;
    }
  }

  sp = pts_[idx] + margin_ * n;
  return sv + margin_;
}

typedef Polytope<2> Polygon;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_POLYTOPE_H_
