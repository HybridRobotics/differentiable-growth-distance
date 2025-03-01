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
 * @file capsule.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 2D/3D capsule class.
 */

#ifndef DGD_GEOMETRY_XD_CAPSULE_H_
#define DGD_GEOMETRY_XD_CAPSULE_H_

#include <cassert>
#include <cmath>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 2D/3D capsule class.
 *
 * @note The capsule is oriented along the x-axis. The axis length of the
 * capsule does not include the radius.
 *
 * @tparam dim Dimension of the capsule.
 */
template <int dim>
class Capsule : public ConvexSet<dim> {
 public:
  /**
   * @brief Constructs a Capsule object.
   *
   * @param hlx    Half axis length.
   * @param radius Radius.
   * @param margin Safety margin.
   */
  Capsule(Real hlx, Real radius, Real margin);

  ~Capsule() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vecf<dim>& sp) const;

 private:
  const Real hlx_;    /**< Half axis length. */
  const Real radius_; /**< Radius. */
  const Real margin_; /**< Safety margin. */
};

template <int dim>
inline Capsule<dim>::Capsule(Real hlx, Real radius, Real margin)
    : ConvexSet<dim>(margin + radius),
      hlx_(hlx),
      radius_(radius),
      margin_(margin) {
  static_assert((dim == 2) || (dim == 3),
                "Incompatible dimension (not 2 or 3)!");
  assert((hlx > Real(0.0)) && (radius > Real(0.0)));
  assert(margin >= Real(0.0));
}

template <int dim>
template <typename Derived>
inline Real Capsule<dim>::SupportFunction(const MatrixBase<Derived>& n,
                                          Vecf<dim>& sp) const {
  static_assert(Derived::RowsAtCompileTime == dim,
                "Size of normal is not equal to dim!");

  sp = Capsule<dim>::inradius_ * n;
  sp(0) += std::copysign(hlx_, n(0));
  return sp.dot(n);
}

template <int dim>
inline Real Capsule<dim>::SupportFunction(const Vecf<dim>& n,
                                          Vecf<dim>& sp) const {
  return SupportFunction<Vecf<dim>>(n, sp);
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_CAPSULE_H_
