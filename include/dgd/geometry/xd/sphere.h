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
 * @brief 2D/3D sphere class.
 */

#ifndef DGD_GEOMETRY_XD_SPHERE_H_
#define DGD_GEOMETRY_XD_SPHERE_H_

#include <iostream>
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
class SphereImpl : public ConvexSet<dim> {
 public:
  /**
   * @param radius Radius.
   */
  explicit SphereImpl(Real radius);

  ~SphereImpl() = default;

  Real SupportFunction(
      const Vecr<dim>& n, Vecr<dim>& sp,
      SupportFunctionHint<dim>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vecr<dim>& n, SupportFunctionDerivatives<dim>& deriv,
      SupportFunctionHint<dim>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<dim>& /*zn*/, SupportPatchHull<dim>& sph,
      NormalConeSpan<dim>& ncs,
      const BasePointHint<dim>* /*hint*/ = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real radius_; /**< Radius. */
};

template <int dim>
inline SphereImpl<dim>::SphereImpl(Real radius)
    : ConvexSet<dim>(radius), radius_(radius) {
  if (radius < Real(0.0)) throw std::domain_error("Radius is negative");
}

template <int dim>
inline Real SphereImpl<dim>::SupportFunction(
    const Vecr<dim>& n, Vecr<dim>& sp,
    SupportFunctionHint<dim>* /*hint*/) const {
  sp = radius_ * n;
  return radius_;
}

template <int dim>
inline Real SphereImpl<dim>::SupportFunction(
    const Vecr<dim>& n, SupportFunctionDerivatives<dim>& deriv,
    SupportFunctionHint<dim>* /*hint*/) const {
  deriv.Dsp = radius_ * (Matr<dim, dim>::Identity() - n * n.transpose());
  deriv.sp = radius_ * n;
  deriv.differentiable = true;
  return radius_;
}

template <int dim>
inline bool SphereImpl<dim>::RequireUnitNormal() const {
  return (radius_ > Real(0.0));
}

template <int dim>
inline void SphereImpl<dim>::ComputeLocalGeometry(
    const NormalPair<dim>& /*zn*/, SupportPatchHull<dim>& sph,
    NormalConeSpan<dim>& ncs, const BasePointHint<dim>* /*hint*/) const {
  sph.aff_dim = 0;
  ncs.span_dim = IsPolytopic() ? dim : 1;
}

template <int dim>
inline bool SphereImpl<dim>::IsPolytopic() const {
  return (radius_ == Real(0.0));
}

template <int dim>
inline void SphereImpl<dim>::PrintInfo() const {
  std::cout << "Type: Sphere (dim = " << dim << ")" << std::endl
            << "  Radius: " << radius_ << std::endl;
}

using Circle = SphereImpl<2>;
using Sphere = SphereImpl<3>;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_SPHERE_H_
