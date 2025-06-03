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

#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief 2D/3D sphere class.
 *
 * @tparam dim Dimension of the sphere.
 */
template <int dim>
class TSphere : public ConvexSet<dim> {
 public:
  /**
   * @brief Constructs a Sphere object.
   *
   * @param radius Radius.
   */
  explicit TSphere(Real radius);

  ~TSphere() = default;

  Real SupportFunction(
      const Vecr<dim>& n, Vecr<dim>& sp,
      SupportFunctionHint<dim>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

 private:
  const Real radius_; /**< Radius. */
};

template <int dim>
inline TSphere<dim>::TSphere(Real radius)
    : ConvexSet<dim>(radius), radius_(radius) {
  static_assert((dim == 2) || (dim == 3), "Incompatible dim");
  if (radius <= 0.0) {
    throw std::domain_error("Radius is nonpositive");
  }
}

template <int dim>
inline Real TSphere<dim>::SupportFunction(
    const Vecr<dim>& n, Vecr<dim>& sp,
    SupportFunctionHint<dim>* /*hint*/) const {
  sp = radius_ * n;
  return radius_;
}

template <int dim>
inline bool TSphere<dim>::RequireUnitNormal() const {
  return true;
}

typedef TSphere<2> Circle;
typedef TSphere<3> Sphere;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_SPHERE_H_
