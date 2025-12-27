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
 * @brief Rotation transformation functions.
 */

#ifndef DGD_UTILS_TRANSFORMATIONS_H_
#define DGD_UTILS_TRANSFORMATIONS_H_

#include <cmath>
#include <type_traits>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Returns the skew-symmetric matrix of a 3D vector.
 *
 * @param  v 3D vector.
 * @return Skew-symmetric matrix.
 */
inline Rotation3r Hat(const Vec3r& v) {
  Rotation3r rot;
  rot << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return rot;
}

/**
 * @brief Returns a rotation matrix using the body ZYX Euler angles.
 *
 * @param  euler ZYX Euler angles, in the form (roll, pitch, yaw).
 * @return Rotation matrix.
 */
inline Rotation3r EulerToRotation(const Vec3r& euler) {
  const Real cr = std::cos(euler(0)), sr = std::sin(euler(0));
  const Real cp = std::cos(euler(1)), sp = std::sin(euler(1));
  const Real cy = std::cos(euler(2)), sy = std::sin(euler(2));

  // R = Rz(yaw) * Ry(pitch) * Rx(roll).
  Rotation3r rot;
  // clang-format off
  rot << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
         sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
         -sp,     cp * sr,                cp * cr;
  // clang-format on
  return rot;
}

/**
 * @brief Returns a rotation matrix using the angle-axis representation.
 *
 * @param  ax  Rotation axis with unit 2-norm.
 * @param  ang Rotation angle.
 * @return Rotation matrix.
 */
inline Rotation3r AngleAxisToRotation(const Vec3r& ax, Real ang) {
  const Real c = std::cos(ang);

  // R = c*I + (1-c)*ax*ax' + s*hat(ax).
  Rotation3r rot;
  rot = c * Rotation3r::Identity() + (Real(1.0) - c) * (ax * ax.transpose()) +
        std::sin(ang) * Hat(ax);
  return rot;
}

}  // namespace dgd

#endif  // DGD_UTILS_TRANSFORMATIONS_H_
