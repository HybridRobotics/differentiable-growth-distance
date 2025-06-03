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
 * @file polytope.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-03-01
 * @brief 3D polytope class.
 */

#ifndef DGD_GEOMETRY_3D_POLYTOPE_H_
#define DGD_GEOMETRY_3D_POLYTOPE_H_

// #include <Eigen/Dense>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief 3D convex polytope class.
 */
class Polytope : public ConvexSet<3> {
 public:
  /**
   * @brief Constructs a Polytope object.
   *
   * @attention When used as a standalone set, the polytope must contain the
   * origin in its interior. This property is not enforced and must be
   * guaranteed by the user, whenever necessary.
   *
   * @see MeshLoader::ProcessPoints(const std::vector<Vec3r>&)
   * @see MeshLoader::MakeVertexGraph
   *
   * @param vert     Vector of n three-dimensional vertices.
   * @param margin   Safety margin.
   * @param inradius Polytope inradius.
   * @param thresh   Support function threshold (default = 0.75).
   */
  explicit Polytope(const std::vector<Vec3r>& vert, Real margin, Real inradius,
                    Real thresh = Real(0.75));

  ~Polytope() = default;

  Real SupportFunction(
      const Vec3r& n, Vec3r& sp,
      SupportFunctionHint<3>* hint = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  /**
   * @brief Gets the polytope vertices.
   *
   * @return Polytope vertices.
   */
  const std::vector<Vec3r>& vertices() const;

  /**
   * @brief Returns the number of vertices in the polytope.
   *
   * @return Number of vertices.
   */
  int nvert() const;

 private:
  const std::vector<Vec3r> vert_; /**< Polytope vertices. */
  const Real margin_;             /**< Safety margin. */

  const Real thresh_;  // Support function threshold.
};

inline Polytope::Polytope(const std::vector<Vec3r>& vert, Real margin,
                          Real inradius, Real thresh)
    : ConvexSet<3>(margin + inradius),
      vert_(vert),
      margin_(margin),
      thresh_(thresh) {
  if ((margin < 0.0) || (inradius <= 0.0)) {
    throw std::domain_error("Invalid margin or inradius");
  }

  // const int nvert{static_cast<int>(vert.size())};
  // if (nvert < 4) {
  //   throw std::domain_error("Polytope is not solid");
  // }

  // Matr<3, -1> aff_vert(3, nvert - 1);
  // for (int i = 1; i < nvert; ++i) aff_vert.col(i - 1) = vert[i] - vert[0];
  // const Eigen::ColPivHouseholderQR<Matr<3, -1>> qr(aff_vert);
  // const int rank{static_cast<int>(qr.rank())};
  // if (rank != 3) {
  //   throw std::domain_error("Polytope is not solid");
  // }
}

inline Real Polytope::SupportFunction(const Vec3r& n, Vec3r& sp,
                                      SupportFunctionHint<3>* hint) const {
  // Current best index.
  int idx{(hint && hint->n_prev.dot(n) > thresh_) ? hint->idx_ws : 0};
  assert(idx >= 0);
  // Current support value, current best support value.
  Real s{0.0}, sv{n.dot(vert_[idx])};

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

inline bool Polytope::RequireUnitNormal() const { return (margin_ > 0.0); }

inline const std::vector<Vec3r>& Polytope::vertices() const { return vert_; }

inline int Polytope::nvert() const { return static_cast<int>(vert_.size()); }

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_POLYTOPE_H_
