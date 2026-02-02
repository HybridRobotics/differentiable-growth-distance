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
 * @brief 2D/3D capsule class.
 */

#ifndef DGD_GEOMETRY_XD_CAPSULE_H_
#define DGD_GEOMETRY_XD_CAPSULE_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

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
class CapsuleImpl : public ConvexSet<dim> {
 public:
  /**
   * @param hlx    Half axis length.
   * @param radius Radius.
   * @param margin Safety margin.
   */
  explicit CapsuleImpl(Real hlx, Real radius, Real margin = Real(0.0));

  ~CapsuleImpl() = default;

  Real SupportFunction(
      const Vecr<dim>& n, Vecr<dim>& sp,
      SupportFunctionHint<dim>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vecr<dim>& n, SupportFunctionDerivatives<dim>& deriv,
      SupportFunctionHint<dim>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<dim>& zn, SupportPatchHull<dim>& sph,
      NormalConeSpan<dim>& ncs,
      const BasePointHint<dim>* /*hint*/ = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx_;    /**< Half axis length. */
  const Real radius_; /**< Radius. */
  const Real margin_; /**< Safety margin. */
};

template <int dim>
inline CapsuleImpl<dim>::CapsuleImpl(Real hlx, Real radius, Real margin)
    : ConvexSet<dim>(margin + radius),
      hlx_(hlx),
      radius_(radius),
      margin_(margin) {
  if ((hlx <= Real(0.0)) || (radius < Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid axis length, radius, or margin");
  }
}

template <int dim>
inline Real CapsuleImpl<dim>::SupportFunction(
    const Vecr<dim>& n, Vecr<dim>& sp,
    SupportFunctionHint<dim>* /*hint*/) const {
  sp = CapsuleImpl<dim>::inradius_ * n;
  sp(0) += std::copysign(hlx_, n(0));
  return sp.dot(n);
}

template <int dim>
inline Real CapsuleImpl<dim>::SupportFunction(
    const Vecr<dim>& n, SupportFunctionDerivatives<dim>& deriv,
    SupportFunctionHint<dim>* /*hint*/) const {
  if (std::abs(hlx_ * n(0)) < Real(0.5) * CapsuleImpl<dim>::eps_sp_) {
    deriv.differentiable = false;
  } else {
    deriv.Dsp = CapsuleImpl<dim>::inradius_ *
                (Matr<dim, dim>::Identity() - n * n.transpose());
    deriv.differentiable = true;
  }
  return SupportFunction(n, deriv.sp);
}

template <int dim>
inline bool CapsuleImpl<dim>::RequireUnitNormal() const {
  return (CapsuleImpl<dim>::inradius_ > Real(0.0));
}

template <int dim>
inline void CapsuleImpl<dim>::ComputeLocalGeometry(
    const NormalPair<dim>& zn, SupportPatchHull<dim>& sph,
    NormalConeSpan<dim>& ncs, const BasePointHint<dim>* /*hint*/) const {
  if (std::abs(zn.n(0)) <= CapsuleImpl<dim>::eps_d_) {
    // Support patch is a line segment.
    sph.aff_dim = 1;
    if constexpr (dim == 3) sph.basis.col(0) = Vec3r::UnitX();
  } else {
    // Support patch is a point.
    sph.aff_dim = 0;
  }

  if (!IsPolytopic()) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else if (hlx_ - std::abs(zn.z(0)) > CapsuleImpl<dim>::eps_p_) {
    // Normal cone is a ray (dim = 2) or a 2D plane (dim = 3).
    ncs.span_dim = dim - 1;
    if constexpr (dim == 3)
      ncs.basis.col(0) = Vec3r(Real(0.0), zn.n(2), -zn.n(1));
  } else {
    // Normal cone is a halfspace.
    ncs.span_dim = dim;
  }
}

template <int dim>
inline bool CapsuleImpl<dim>::IsPolytopic() const {
  return (CapsuleImpl<dim>::inradius_ == Real(0.0));
}

template <int dim>
inline void CapsuleImpl<dim>::PrintInfo() const {
  std::cout << "Type: Capsule (dim = " << dim << ")" << std::endl
            << "  Half axis length: " << hlx_ << std::endl
            << "  Radius: " << radius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

using Stadium = CapsuleImpl<2>;
using Capsule = CapsuleImpl<3>;

}  // namespace dgd

#endif  // DGD_GEOMETRY_XD_CAPSULE_H_
