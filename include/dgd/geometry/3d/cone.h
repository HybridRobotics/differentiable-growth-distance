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
 * @brief 3D cone class.
 */

#ifndef DGD_GEOMETRY_3D_CONE_H_
#define DGD_GEOMETRY_3D_CONE_H_

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D cone class with radius \f$r\f$ and height \f$h\f$.
 *
 * @note The origin is located in the incenter of the cone. The center of the
 * base of the cone is at \f$(0, 0, -\rho)\f$, where \f$\rho\f$ is the inradius
 * of the cone and
 * \f[
 * \rho = \frac{r(\sqrt{r^2 + h^2} - r)}{h}.
 * \f]
 */
class Cone : public ConvexSet<3> {
 public:
  /**
   * @param radius Radius.
   * @param height Height.
   * @param margin Safety margin.
   */
  explicit Cone(Real radius, Real height, Real margin = Real(0.0));

  ~Cone() = default;

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

  /**
   * @brief Gets the z-offset of the base of the cone.
   *
   * The center of the base of the cone is at \f$(0, 0, -\rho)\f$, where
   * \f$\rho\f$ is the offset (also the inradius).
   */
  Real offset() const;

 private:
  const Real r_;      /**< Radius. */
  const Real h_;      /**< Height. */
  Real tha_;          /**< Tangent of the cone half angle. */
  Real rho_;          /**< Cone inradius (not considering the safety margin). */
  const Real margin_; /**< Safety margin. */
};

inline Cone::Cone(Real radius, Real height, Real margin)
    : ConvexSet<3>(), r_(radius), h_(height), margin_(margin) {
  if ((radius <= Real(0.0)) || (height <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid radius, height, or margin");
  }
  tha_ = r_ / h_;
  rho_ = (std::sqrt(r_ * r_ + h_ * h_) * r_ - r_ * r_) / h_;
  set_inradius(rho_ + margin);
}

inline Real Cone::SupportFunction(const Vec3r& n, Vec3r& sp,
                                  SupportFunctionHint<3>* /*hint*/) const {
  const Real k = std::sqrt(n(0) * n(0) + n(1) * n(1));
  sp = margin_ * n;
  if (n(2) >= tha_ * k) {
    // The cone vertex is the support point.
    sp(2) += (h_ - rho_);
    return (h_ - rho_) * n(2) + margin_;
  } else {
    // The support point lies in the cone base.
    if (k >= kEps) sp.head<2>() += r_ * n.head<2>() / k;
    sp(2) -= rho_;
    return sp.dot(n);
  }
}

inline Real Cone::SupportFunction(const Vec3r& n,
                                  SupportFunctionDerivatives<3>& deriv,
                                  SupportFunctionHint<3>* /*hint*/) const {
  const Real k2 = n(0) * n(0) + n(1) * n(1);
  const Real k = std::sqrt(k2);
  const Real diff = h_ * n(2) - r_ * k;
  deriv.sp = margin_ * n;
  if (diff >= Real(0.0)) {
    // The cone vertex is the support point.
    if (diff < eps_sp_) {
      deriv.differentiable = false;
    } else {
      deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
      deriv.differentiable = true;
    }
    deriv.sp(2) += (h_ - rho_);
    return (h_ - rho_) * n(2) + margin_;
  } else {
    // The support point lies in the cone base.
    if (std::min(Real(2.0) * r_ * k, -diff) < eps_sp_) {
      deriv.differentiable = false;
    } else {
      deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
      deriv.Dsp.block<2, 2>(0, 0) +=
          r_ / (k2 * k) * Vec2r(n(1), -n(0)) * Vec2r(n(1), -n(0)).transpose();
      deriv.differentiable = true;
    }
    if (k >= kEps) deriv.sp.head<2>() += r_ * n.head<2>() / k;
    deriv.sp(2) -= rho_;
    return deriv.sp.dot(n);
  }
}

inline bool Cone::RequireUnitNormal() const { return (margin_ > Real(0.0)); }

inline void Cone::ComputeLocalGeometry(const NormalPair<3>& zn,
                                       SupportPatchHull<3>& sph,
                                       NormalConeSpan<3>& ncs,
                                       const BasePointHint<3>* /*hint*/) const {
  const Real k = std::sqrt(zn.n(0) * zn.n(0) + zn.n(1) * zn.n(1));
  if (zn.n(2) > (tha_ + eps_d_) * k) {
    // Support patch is a point.
    sph.aff_dim = 0;
  } else if (zn.n(2) >= (tha_ - eps_d_) * k) {
    // Support patch is a line segment.
    sph.aff_dim = 1;
    sph.basis.col(0).head<2>() = tha_ * zn.n.head<2>();
    sph.basis.col(0)(2) = -k;
    sph.basis.col(0) /= k * std::sqrt(Real(1.0) + tha_ * tha_);
  } else if (k <= eps_d_) {
    // Support patch is a disk.
    sph.aff_dim = 2;
  } else {
    // Support patch is a point.
    sph.aff_dim = 0;
  }

  if (margin_ > Real(0.0)) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else if (zn.z(2) >= h_ - rho_ - eps_p_) {
    // Normal cone is a 3D cone.
    ncs.span_dim = 3;
  } else {
    Real r2;
    if ((zn.z(2) > -rho_ + eps_p_) ||
        ((r2 = zn.z.head<2>().squaredNorm()) < (r_ - eps_p_) * (r_ - eps_p_))) {
      // Normal cone is a ray.
      ncs.span_dim = 1;
    } else {
      // Normal cone is a 2D cone.
      ncs.span_dim = 2;
      ncs.basis.col(0) =
          Vec3r(zn.z(1), -zn.z(0), Real(0.0)).cross(zn.n) / std::sqrt(r2);
    }
  }
}

inline bool Cone::IsPolytopic() const { return false; }

inline void Cone::PrintInfo() const {
  std::cout << "Type: Cone (dim = 3)" << std::endl
            << "  Radius: " << r_ << std::endl
            << "  Height: " << h_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

inline Real Cone::offset() const { return rho_; }

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_CONE_H_
