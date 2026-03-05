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

  bool ProjectionDerivative(
      const Vec3r& p, const Vec3r& pi, Matr<3, 3>& d_pi_p,
      const BasePointHint<3>* /*hint*/ = nullptr) const final override;

  Real Bounds(Vec3r* min = nullptr, Vec3r* max = nullptr) const final override;

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
  const Real k = n.tail<2>().norm();
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
    deriv.d_sp_n.noalias() = -margin_ * (n * n.transpose());
    deriv.d_sp_n.diagonal().array() += margin_;
    const Vec2r t(n(2), -n(1));
    deriv.d_sp_n.block<2, 2>(1, 1).noalias() +=
        (radius_ / (k2 * k)) * (t * t.transpose());
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

inline bool Cylinder::ProjectionDerivative(
    const Vec3r& p, const Vec3r& pi, Matr<3, 3>& d_pi_p,
    const BasePointHint<3>* /*hint*/) const {
  const Real dx = std::abs(p(0)) - hlx_;
  const Real r = p.tail<2>().norm();

  if ((std::abs(dx) <= eps_p_) || std::abs(r - radius_) <= eps_p_) return false;

  if (r < radius_) {
    // Projection lies on the left/right disk.
    d_pi_p.setIdentity();
    d_pi_p(0, 0) = Real(0.0);
  } else if (dx < Real(0.0)) {
    // Projection lies on the cylindrical surface.
    d_pi_p.setZero();
    const Real s = (radius_ + margin_) / r;
    d_pi_p.bottomRightCorner<2, 2>().noalias() =
        -(s / (r * r)) * (p.tail<2>() * p.tail<2>().transpose());
    d_pi_p.bottomRightCorner<2, 2>().diagonal().array() += s;
    d_pi_p(0, 0) = Real(1.0);
  } else {
    // Projection lies on the left/right circular edge.
    const Vec3r w = Vec3r(Real(0.0), -p(2), p(1)) / r;
    const Vec3r v(std::copysign(dx, p(0)), p(1) - radius_ * w(2),
                  p(2) + radius_ * w(1));
    const Real v2 = v.squaredNorm();
    const Real s = margin_ / std::sqrt(v2);
    d_pi_p.noalias() = -(s / v2) * (v * v.transpose());
    d_pi_p.noalias() += (pi.tail<2>().norm() / r - s) * (w * w.transpose());
    d_pi_p.diagonal().array() += s;
  }

  return true;
}

inline Real Cylinder::Bounds(Vec3r* min, Vec3r* max) const {
  const Real r_m = radius_ + margin_;
  const Real hlx_m = hlx_ + margin_;
  if (min) *min = -Vec3r(hlx_m, r_m, r_m);
  if (max) *max = Vec3r(hlx_m, r_m, r_m);
  return Real(2.0) * std::sqrt(hlx_m * hlx_m + Real(2.0) * r_m * r_m);
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
