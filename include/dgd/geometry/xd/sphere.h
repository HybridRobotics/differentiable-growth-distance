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
 * @file sphere.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 2D/3D sphere class.
 */

#ifndef DGD_GEOMETRY_XD_SPHERE_H_
#define DGD_GEOMETRY_XD_SPHERE_H_

#include <cassert>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief 2D/3D sphere class.
 *
 * @tparam dim Dimension of the sphere.
 */
template <int dim>
class Sphere : public ConvexSet<dim> {
 public:
  /**
   * @brief Constructs a Sphere object.
   *
   * @param radius Radius.
   */
  Sphere(Real radius);

  ~Sphere() {};

  Real SupportFunction(const Vecf<dim>& n, Vecf<dim>& sp) const final;

  template <typename Derived>
  Real SupportFunction(const MatrixBase<Derived>& n, Vecf<dim>& sp) const;

 private:
  const Real radius_; /**< Radius. */
};

template <int dim>
inline Sphere<dim>::Sphere(Real radius)
    : ConvexSet<dim>(radius), radius_(radius) {
  static_assert((dim == 2) || (dim == 3), "Incompatible dim!");
  assert(radius > Real(0.0));
}

template <int dim>
template <typename Derived>
inline Real Sphere<dim>::SupportFunction(const MatrixBase<Derived>& n,
                                         Vecf<dim>& sp) const {
  static_assert(Derived::RowsAtCompileTime == dim,
                "Size of normal is not equal to dim!");

  sp = radius_ * n;
  return radius_;
}

template <int dim>
inline Real Sphere<dim>::SupportFunction(const Vecf<dim>& n,
                                         Vecf<dim>& sp) const {
  return SupportFunction<Vecf<dim>>(n, sp);
}

typedef Sphere<2> Circle;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_SPHERE_H_
