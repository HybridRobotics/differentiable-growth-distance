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
 * @brief 3D cylinder class.
 */

#ifndef DGD_GEOMETRY_3D_CYLINDER_H_
#define DGD_GEOMETRY_3D_CYLINDER_H_

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D cylinder class.
 *
 * @note The cylinder axis is oriented along the x-axis.
 */
class Cylinder : public ConvexSet<3> {
 public:
  /**
   * @param hlx    Half axis length.
   * @param radius Radius.
   * @param margin Safety margin.
   */
  explicit Cylinder(Real hlx, Real radius, Real margin = Real(0.0));

  ~Cylinder() = default;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
      const BasePointHint<3>* /*hint*/ = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

 private:
  const Real hlx_;    /**< Half axis length. */
  const Real radius_; /**< Radius. */
  const Real margin_; /**< Safety margin. */
};

inline Cylinder::Cylinder(Real hlx, Real radius, Real margin)
    : ConvexSet<3>(), hlx_(hlx), radius_(radius), margin_(margin) {
  if ((hlx <= Real(0.0)) || (radius <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid axis length, radius, or margin");
  }
  set_inradius(std::min(hlx, radius) + margin);
}

inline Real Cylinder::SupportFunction(const Vec3r& n, Vec3r& sp,
                                      SupportFunctionHint<3>* /*hint*/) const {
  const Real k = std::sqrt(n(1) * n(1) + n(2) * n(2));
  sp = margin_ * n;
  if (k >= kEps) sp.tail<2>() += radius_ * n.tail<2>() / k;
  sp(0) += std::copysign(hlx_, n(0));
  return sp.dot(n);
}

inline Real Cylinder::SupportFunction(const Vec3r& n,
                                      SupportFunctionDerivatives<3>& deriv,
                                      SupportFunctionHint<3>* /*hint*/) const {
  const Real k2 = n(1) * n(1) + n(2) * n(2);
  const Real k = std::sqrt(k2);
  const Real diff = std::min(radius_ * k, std::abs(hlx_ * n(0)));
  if (diff < Real(0.5) * eps_sp_) {
    deriv.differentiable = false;
  } else {
    deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
    deriv.Dsp.block<2, 2>(1, 1) += radius_ / (k2 * k) * Vec2r(n(2), -n(1)) *
                                   Vec2r(n(2), -n(1)).transpose();
    deriv.differentiable = true;
  }
  deriv.sp = margin_ * n;
  if (k >= kEps) deriv.sp.tail<2>() += radius_ * n.tail<2>() / k;
  deriv.sp(0) += std::copysign(hlx_, n(0));
  return deriv.sp.dot(n);
}

inline bool Cylinder::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

inline void Cylinder::ComputeLocalGeometry(
    const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
    const BasePointHint<3>* /*hint*/) const {
  if (std::abs(zn.n(0)) <= eps_d_) {
    // Support patch is a line segment.
    sph.aff_dim = 1;
    sph.basis.col(0) = Vec3r::UnitX();
  } else if (zn.n.tail<2>().squaredNorm() <= eps_d_ * eps_d_) {
    // Support patch is a disk.
    sph.aff_dim = 2;
  } else {
    // Support patch is a point.
    sph.aff_dim = 0;
  }

  Real r2;
  if ((margin_ > Real(0.0)) || (hlx_ - std::abs(zn.z(0)) > eps_p_) ||
      ((r2 = zn.z.tail<2>().squaredNorm()) <
       (radius_ - eps_p_) * (radius_ - eps_p_))) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else {
    // Normal cone is a 2D cone.
    ncs.span_dim = 2;
    ncs.basis.col(0) =
        Vec3r(Real(0.0), zn.z(2), -zn.z(1)).cross(zn.n) / std::sqrt(r2);
  }
}

inline bool Cylinder::IsPolytopic() const { return false; }

inline void Cylinder::PrintInfo() const {
  std::cout << "Type: Cylinder (dim = 3)" << std::endl
            << "  Half axis length: " << hlx_ << std::endl
            << "  Radius: " << radius_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_CYLINDER_H_
