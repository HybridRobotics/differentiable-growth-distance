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
 * @brief 3D frustum class.
 */

#ifndef DGD_GEOMETRY_3D_FRUSTUM_H_
#define DGD_GEOMETRY_3D_FRUSTUM_H_

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Axis-aligned 3D frustum class with base radius \f$r_b\f$, top radius
 * \f$r_t\f$, and height \f$h\f$.
 *
 * @note The origin is located in the incenter of the frustum. If
 * \f$r_b >= r_t\f$, the center of the base of the frustum is at
 * \f$(0, 0, -\rho)\f$, where \f$\rho\f$ is the inradius of the frustum and
 * \f[
 * \rho = \min\bigg\{
 *        \frac{h}{2},
 *        \frac{r_b(\sqrt{r_b^2 + h_c^2} - r_b)}{h_c}
 *        \biggr\},
 * \f]
 * where \f$h_c = r_b / (r_b - r_t) h\f$. If \f$r_b < r_t\f$, the center of the
 * base of the frustum is at \f$(0, 0, h - \rho)\f$.
 */
class Frustum : public ConvexSet<3> {
 public:
  /**
   * @param base_radius Base radius.
   * @param top_radius  Top radius.
   * @param height      Height.
   * @param margin      Safety margin.
   */
  explicit Frustum(Real base_radius, Real top_radius, Real height,
                   Real margin = Real(0.0));

  ~Frustum() = default;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* /*hint*/ = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  void ComputeLocalGeometry(
      const NormalPair<3>& zn, SupportPatchHull<3>& sph, NormalConeSpan<3>& ncs,
      BasePointHint<3>* /*hint*/ = nullptr) const final override;

  bool ProjectionDerivative(
      const Vec3r& p, const Vec3r& pi, Matr<3, 3>& d_pi_p,
      BasePointHint<3>* /*hint*/ = nullptr) const final override;

  Real Bounds(Vec3r* min = nullptr, Vec3r* max = nullptr) const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

  /**
   * @brief Gets the z-offset of the base of the frustum.
   *
   * The center of the base of the frustum is at \f$(0, 0, -o)\f$, where
   * \f$o\f$ is the offset.
   */
  Real offset() const;

 private:
  const Real rb_;     /**< Base radius. */
  const Real rt_;     /**< Top radius. */
  const Real h_;      /**< Height. */
  Real tha_;          /**< Tangent of the frustum half angle. */
  Real offset_;       /**< z-offset of the base of the frustum. */
  const Real margin_; /**< Safety margin. */
};

inline Frustum::Frustum(Real base_radius, Real top_radius, Real height,
                        Real margin)
    : ConvexSet<3>(),
      rb_(base_radius),
      rt_(top_radius),
      h_(height),
      margin_(margin) {
  if ((base_radius < Real(0.0)) || (top_radius < Real(0.0)) ||
      (std::max(base_radius, top_radius) <= Real(0.0)) ||
      (height <= Real(0.0)) || (margin < Real(0.0))) {
    throw std::domain_error("Invalid radii, height, or margin");
  }
  tha_ = (rb_ - rt_) / h_;
  const Real r = std::max(rb_, rt_);
  const Real rho =
      std::min(h_ / Real(2.0),
               (std::sqrt(Real(1.0) + tha_ * tha_) - std::abs(tha_)) * r);
  offset_ = (rb_ >= rt_) ? rho : h_ - rho;
  set_inradius(rho + margin);
}

inline Real Frustum::SupportFunction(const Vec3r& n, Vec3r& sp,
                                     SupportFunctionHint<3>* /*hint*/) const {
  const Real k = n.head<2>().norm();
  sp = margin_ * n;
  if (n(2) >= tha_ * k) {
    // The support point lies in the frustum top.
    if (k >= kEps) sp.head<2>() += rt_ * n.head<2>() / k;
    sp(2) += (h_ - offset_);
  } else {
    // The support point lies in the frustum base.
    if (k >= kEps) sp.head<2>() += rb_ * n.head<2>() / k;
    sp(2) -= offset_;
  }
  return sp.dot(n);
}

inline Real Frustum::SupportFunction(const Vec3r& n,
                                     SupportFunctionDerivatives<3>& deriv,
                                     SupportFunctionHint<3>* /*hint*/) const {
  const Real k2 = n(0) * n(0) + n(1) * n(1);
  const Real k = std::sqrt(k2);
  const Real diff = h_ * n(2) - (rb_ - rt_) * k;
  deriv.sp = margin_ * n;
  if (diff >= Real(0.0)) {
    // The support point lies in the frustum top.
    if (std::min(Real(2.0) * rt_ * k, diff) < eps_sp_) {
      deriv.differentiable = false;
    } else {
      deriv.d_sp_n.noalias() = -margin_ * (n * n.transpose());
      deriv.d_sp_n.diagonal().array() += margin_;
      const Vec2r t(n(1), -n(0));
      deriv.d_sp_n.block<2, 2>(0, 0).noalias() +=
          (rt_ / (k2 * k)) * (t * t.transpose());
      deriv.differentiable = true;
    }
    if (k >= kEps) deriv.sp.head<2>() += rt_ * n.head<2>() / k;
    deriv.sp(2) += (h_ - offset_);
  } else {
    // The support point lies in the frustum base.
    if (std::min(Real(2.0) * rb_ * k, -diff) < eps_sp_) {
      deriv.differentiable = false;
    } else {
      deriv.d_sp_n.noalias() = -margin_ * (n * n.transpose());
      deriv.d_sp_n.diagonal().array() += margin_;
      const Vec2r t(n(1), -n(0));
      deriv.d_sp_n.block<2, 2>(0, 0).noalias() +=
          (rb_ / (k2 * k)) * (t * t.transpose());
      deriv.differentiable = true;
    }
    if (k >= kEps) deriv.sp.head<2>() += rb_ * n.head<2>() / k;
    deriv.sp(2) -= offset_;
  }
  return deriv.sp.dot(n);
}

inline bool Frustum::RequireUnitNormal() const { return (margin_ > Real(0.0)); }

inline void Frustum::ComputeLocalGeometry(const NormalPair<3>& zn,
                                          SupportPatchHull<3>& sph,
                                          NormalConeSpan<3>& ncs,
                                          BasePointHint<3>* /*hint*/) const {
  const Real k = zn.n.head<2>().norm();
  if (zn.n(2) > (tha_ + eps_d_) * k) {
    if ((rt_ > Real(0.0)) && (k <= eps_d_)) {
      // Support patch is a disk.
      sph.aff_dim = 2;
    } else {
      // Support patch is a point.
      sph.aff_dim = 0;
    }
  } else if (zn.n(2) >= (tha_ - eps_d_) * k) {
    // Support patch is a line segment.
    sph.aff_dim = 1;
    sph.basis.col(0).head<2>() = tha_ * zn.n.head<2>();
    sph.basis.col(0)(2) = -k;
    sph.basis.col(0) /= k * std::sqrt(Real(1.0) + tha_ * tha_);
  } else if ((rb_ > Real(0.0)) && (k <= eps_d_)) {
    // Support patch is a disk.
    sph.aff_dim = 2;
  } else {
    // Support patch is a point.
    sph.aff_dim = 0;
  }

  const Real h = zn.z(2) + offset_;
  if ((margin_ > Real(0.0)) || ((h < h_ - eps_p_) && (h > eps_p_))) {
    // Normal cone is a ray.
    ncs.span_dim = 1;
  } else {
    const Real rd = (h <= eps_p_) ? rb_ : rt_;
    Real r2;
    if (rd > Real(0.0)) {
      if ((r2 = zn.z.head<2>().squaredNorm()) < (rd - eps_p_) * (rd - eps_p_)) {
        // Normal cone is a ray.
        ncs.span_dim = 1;
      } else {
        // Normal cone is a 2D cone.
        ncs.span_dim = 2;
        ncs.basis.col(0) =
            Vec3r(zn.z(1), -zn.z(0), Real(0.0)).cross(zn.n) / std::sqrt(r2);
      }
    } else {
      // Normal cone is a 3D cone.
      ncs.span_dim = 3;
    }
  }
}

inline bool Frustum::ProjectionDerivative(const Vec3r& p, const Vec3r& pi,
                                          Matr<3, 3>& d_pi_p,
                                          BasePointHint<3>* /*hint*/) const {
  const Real r = p.head<2>().norm();
  Real rs, rl, hs, hl, sgn;
  Real ds = p(2) - (h_ - offset_) - tha_ * (r - rt_);
  Real dl = tha_ * (r - rb_) - (p(2) + offset_);
  if (tha_ < Real(0.0)) {
    rs = rb_, rl = rt_, hs = -offset_, hl = h_ - offset_, sgn = Real(-1.0);
    std::swap(ds, dl);
  } else {
    rs = rt_, rl = rb_, hs = h_ - offset_, hl = -offset_, sgn = Real(1.0);
  }

  auto set_edge = [&](Real rd, Real h) -> void {
    const Vec3r w = Vec3r(-p(1), p(0), Real(0.0)) / r;
    const Real f = rd / r;
    const Vec3r v(p(0) - f * p(0), p(1) - f * p(1), p(2) - h);
    const Real v2 = v.squaredNorm();
    const Real s = margin_ / std::sqrt(v2);
    d_pi_p.noalias() = -(s / v2) * (v * v.transpose());
    d_pi_p.noalias() += (f * (Real(1.0) - s)) * (w * w.transpose());
    d_pi_p.diagonal().array() += s;
  };

  if ((sgn * (p(2) - hs) > Real(0.0)) && (ds >= -(r - rs) * eps_d_)) {
    if (ds <= (r - rs) * eps_d_) return false;
    if (rs > Real(0.0)) {
      if (r <= rs + eps_p_) {
        if (r >= rs - eps_p_) return false;
        // Projection lies on the smaller disk.
        d_pi_p.setIdentity();
        d_pi_p(2, 2) = Real(0.0);
      } else {
        // Projection lies on the smaller circular edge.
        set_edge(rs, hs);
      }
    } else {
      // Projection lies on the vertex.
      const Vec3r v(p(0), p(1), p(2) - hs);
      const Real v2 = v.squaredNorm();
      const Real s = margin_ / std::sqrt(v2);
      d_pi_p.noalias() = -(s / v2) * (v * v.transpose());
      d_pi_p.diagonal().array() += s;
    }
  } else if ((sgn * (p(2) - hl) > Real(0.0)) && (dl <= (r - rl) * eps_d_)) {
    if (dl >= -(r - rl) * eps_d_) return false;
    // Projection lies on the frustum surface.
    const Vec3r w = Vec3r(-p(1), p(0), Real(0.0)) / r;
    const Vec3r u = Vec3r(tha_ * p(0), tha_ * p(1), -r) /
                    (r * std::sqrt(Real(1.0) + tha_ * tha_));
    d_pi_p.noalias() = u * u.transpose();
    d_pi_p.noalias() += (pi.head<2>().norm() / r) * (w * w.transpose());
  } else {
    // rl is always positive.
    if (r <= rl + eps_p_) {
      if (r >= rl - eps_p_) return false;
      // Projection lies on the larger disk.
      d_pi_p.setIdentity();
      d_pi_p(2, 2) = Real(0.0);
    } else {
      // Projection lies on the larger circular edge.
      set_edge(rl, hl);
    }
  }

  return true;
}

inline Real Frustum::Bounds(Vec3r* min, Vec3r* max) const {
  const Real r_m = std::max(rb_, rt_) + margin_;
  if (min) *min = -Vec3r(r_m, r_m, offset_ + margin_);
  if (max) *max = Vec3r(r_m, r_m, h_ - offset_ + margin_);
  return Real(2.0) * Vec3r(r_m, r_m, Real(0.5) * h_ + margin_).norm();
}

inline bool Frustum::IsPolytopic() const { return false; }

inline void Frustum::PrintInfo() const {
  std::cout << "Type: Frustum (dim = 3)" << std::endl
            << "  Base radius: " << rb_ << std::endl
            << "  Top radius: " << rt_ << std::endl
            << "  Height: " << h_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

inline Real Frustum::offset() const { return offset_; }

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_FRUSTUM_H_
