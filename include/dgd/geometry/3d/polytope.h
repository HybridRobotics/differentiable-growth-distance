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
 * @brief 3D polytope class.
 */

#ifndef DGD_GEOMETRY_3D_POLYTOPE_H_
#define DGD_GEOMETRY_3D_POLYTOPE_H_

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief 3D convex polytope class.
class Polytope : public ConvexSet<3> {
 public:
  /**
   * @param vert     Vector of n three-dimensional vertices.
   * @param inradius Polytope inradius.
   * @param margin   Safety margin.
   * @param thresh   Support function threshold.
   */
  explicit Polytope(const std::vector<Vec3r>& vert, Real inradius,
                    Real margin = Real(0.0), Real thresh = Real(0.75));

  ~Polytope() = default;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  Real SupportFunction(
      const Vec3r& n, SupportFunctionDerivatives<3>& deriv,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  bool IsPolytopic() const final override;

  void PrintInfo() const final override;

  const std::vector<Vec3r>& vertices() const;

  int nvertices() const;

 private:
  const std::vector<Vec3r> vert_; /**< Polytope vertices. */
  const Real margin_;             /**< Safety margin. */

  const Real thresh_;  // Support function threshold.
};

inline Polytope::Polytope(const std::vector<Vec3r>& vert, Real inradius,
                          Real margin, Real thresh)
    : ConvexSet<3>(margin + inradius),
      vert_(vert),
      margin_(margin),
      thresh_(thresh) {
  if ((margin < Real(0.0)) || (inradius < Real(0.0))) {
    throw std::domain_error("Invalid margin or inradius");
  }
}

inline Real Polytope::SupportFunction(const Vec3r& n, Vec3r& sp,
                                      SupportFunctionHint<3>* hint) const {
  // Current best index.
  int idx = (hint && hint->n_prev.dot(n) > thresh_) ? hint->idx_ws : 0;
  assert(idx >= 0);
  // Current support value, current best support value.
  Real s = Real(0.0), sv = n.dot(vert_[idx]);

  for (int i = 0; i < static_cast<int>(vert_.size()); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      idx = i;
      sv = s;
    }
  }

  if (hint) {
    hint->n_prev = n;
    hint->idx_ws = idx;
  }

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline Real Polytope::SupportFunction(const Vec3r& n,
                                      SupportFunctionDerivatives<3>& deriv,
                                      SupportFunctionHint<3>* hint) const {
  int idx = (hint && hint->n_prev.dot(n) > thresh_) ? hint->idx_ws : 0;
  Real s = Real(0.0), sv = n.dot(vert_[idx]);

  deriv.differentiable = true;
  for (int i = 0; i < static_cast<int>(vert_.size()); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      deriv.differentiable = (s >= sv + eps_diff());
      idx = i;
      sv = s;
    } else {
      if (s > sv - eps_diff()) deriv.differentiable = false;
    }
  }
  if (deriv.differentiable) {
    deriv.Dsp = margin_ * (Matr<3, 3>::Identity() - n * n.transpose());
  }

  if (hint) {
    hint->n_prev = n;
    hint->idx_ws = idx;
  }

  deriv.sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

inline bool Polytope::RequireUnitNormal() const {
  return (margin_ > Real(0.0));
}

inline bool Polytope::IsPolytopic() const { return (margin_ == Real(0.0)); }

inline void Polytope::PrintInfo() const {
  std::cout << "Type: Polytope (dim = 3)" << std::endl
            << "  #Vertices: " << vert_.size() << std::endl;
  for (const auto& v : vert_) {
    std::cout << "    (" << v(0) << ", " << v(1) << ", " << v(2) << ")"
              << std::endl;
  }
  std::cout << "  Inradius: " << margin_ << std::endl
            << "  Margin: " << margin_ << std::endl;
}

inline const std::vector<Vec3r>& Polytope::vertices() const { return vert_; }

inline int Polytope::nvertices() const {
  return static_cast<int>(vert_.size());
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_POLYTOPE_H_
