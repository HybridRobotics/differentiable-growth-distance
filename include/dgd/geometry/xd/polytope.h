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
   * @note The convex hull of pts must contain the origin in its interior.
   *
   * @param pts      (dim, n) matrix consisting of n dim-dimensional vertices.
   * @param margin   Safety margin.
   * @param inradius Polytope inradius.
   */
  Polytope(const MatXf<dim>& pts, Real margin, Real inradius);

  ~Polytope() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vecf<dim>& sp) const;

 private:
  const MatXf<dim> pts_; /**< Polytope vertices. */
  const Real margin_;    /**< Safety margin. */
};

template <int dim>
inline Polytope<dim>::Polytope(const MatXf<dim>& pts, Real margin,
                               Real inradius)
    : ConvexSet<dim>(margin + inradius), pts_(pts), margin_(margin) {
  static_assert((dim == 2) || (dim == 3),
                "Incompatible dimension (not 2 or 3)!");
  assert(margin >= Real(0.0));

  const int num_pts{static_cast<int>(pts.cols())};
  assert(num_pts >= dim + 1);

  MatXf<dim> aff_pts{pts.rightCols(num_pts - 1)};
  aff_pts.colwise() -= pts.col(0);
  const Eigen::ColPivHouseholderQR<MatXf<dim>> qr(aff_pts);
  const int rank{static_cast<int>(qr.rank())};
  assert(rank == dim);
}

template <int dim>
template <typename Derived>
inline Real Polytope<dim>::SupportFunction(const MatrixBase<Derived>& n,
                                           Vecf<dim>& sp) const {
  static_assert(Derived::RowsAtCompileTime == dim,
                "Size of normal is not equal to dim!");

  int idx{0};
  const Real v{(pts_.transpose() * n).maxCoeff(&idx)};
  sp = pts_.col(idx) + margin_ * n;
  return v + margin_;
}

template <int dim>
inline Real Polytope<dim>::SupportFunction(const Vecf<dim>& n,
                                           Vecf<dim>& sp) const {
  return SupportFunction<Vecf<dim>>(n, sp);
}

typedef Polytope<2> Polygon;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_POLYTOPE_H_
