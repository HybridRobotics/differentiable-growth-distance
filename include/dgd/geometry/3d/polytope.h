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
   * @see MeshLoader::ProcessPoints(const std::vector<Vec3f>&)
   * @see MeshLoader::MakeVertexGraph
   *
   * @param vert     Vector of n three-dimensional vertices.
   * @param margin   Safety margin.
   * @param inradius Polytope inradius.
   * @param thresh   (advanced) Support function threshold (default = 0.75).
   */
  Polytope(const std::vector<Vec3f>& vert, Real margin, Real inradius,
           Real thresh = 0.75);

  ~Polytope() {};

  Real SupportFunction(const Vec3f& n, Vec3f& sp) final;

 private:
  const std::vector<Vec3f> vert_; /**< Polytope vertices. */
  const Real margin_;             /**< Safety margin. */

  Vec3f n_prev_;       // Previous normal vector.
  const Real thresh_;  // Support function threshold.
  int idx_hint_;       // Best index hint for support function.
};

inline Polytope::Polytope(const std::vector<Vec3f>& vert, Real margin,
                          Real inradius, Real thresh)
    : ConvexSet<3>(margin + inradius),
      vert_(vert),
      margin_(margin),
      thresh_(thresh) {
  if ((margin < 0.0) || (inradius <= 0.0))
    throw std::domain_error("Invalid margin or inradius");

  // const int nvert{static_cast<int>(vert.size())};
  // if (nvert < 4)
  //   throw std::domain_error("Polytope is not solid");

  // Mat3Xf aff_vert(3, nvert - 1);
  // for (int i = 1; i < nvert; ++i) aff_vert.col(i - 1) = vert[i] - vert[0];
  // const Eigen::ColPivHouseholderQR<Mat3Xf> qr(aff_vert);
  // const int rank{static_cast<int>(qr.rank())};
  // if (rank != 3)
  //   throw std::domain_error("Polytope is not solid");

  n_prev_ = Vec3f::Zero();
  idx_hint_ = -1;
}

inline Real Polytope::SupportFunction(const Vec3f& n, Vec3f& sp) {
  // Current best index.
  int idx{(n_prev_.dot(n) > thresh_) ? idx_hint_ : 0};
  // Current support value, current best support value.
  Real s{0.0}, sv{n.dot(vert_[idx])};

  for (int i = 0; i < vert_.size(); ++i) {
    s = n.dot(vert_[i]);
    if (s > sv) {
      idx = i;
      sv = s;
    }
  }

  n_prev_ = n;
  idx_hint_ = idx;

  sp = vert_[idx] + margin_ * n;
  return sv + margin_;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_3D_POLYTOPE_H_
