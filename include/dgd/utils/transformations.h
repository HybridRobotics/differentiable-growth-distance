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
 * @file transformations.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-07-10
 * @brief Rotation transformation functions.
 */

#ifndef DGD_UTILS_TRANSFORMATIONS_H_
#define DGD_UTILS_TRANSFORMATIONS_H_

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Sets the rotation matrix using the body ZYX Euler angles.
 *
 * @param[in]  euler ZYX Euler angles, in the form (roll, pitch, yaw).
 * @param[out] rot   Rotation matrix.
 */
inline void EulerToRotation(const Vec3r& euler, Rotation3r& rot) {
  const Eigen::AngleAxis<Real> R(euler(0), Vec3r::UnitX());
  const Eigen::AngleAxis<Real> P(euler(1), Vec3r::UnitY());
  const Eigen::AngleAxis<Real> Y(euler(2), Vec3r::UnitZ());
  rot.noalias() = (Y * P * R).matrix();
}

/**
 * @brief Sets the rotation matrix using the angle-axis representation.
 *
 * @param[in]  ax  Rotation axis, must be a unit vector.
 * @param[in]  ang Rotation angle.
 * @param[out] rot Rotation matrix.
 */
inline void AngleAxisToRotation(const Vec3r& ax, Real ang, Rotation3r& rot) {
  rot = Eigen::AngleAxis<Real>(ang, ax).matrix();
}

}  // namespace dgd

#endif  // DGD_UTILS_TRANSFORMATIONS_H_
