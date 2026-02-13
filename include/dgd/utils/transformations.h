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

#include <Eigen/Geometry>
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
 * @attention The rotation axis must be a unit vector.
 *
 * @param  ax  Rotation axis with unit norm.
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

/**
 * @name Velocity computation functions
 * @brief Returns the velocity of a point on a rigid body given its twist.
 *
 * @attention The twist, point, and the velocity are expressed in the same frame
 * of reference.
 *
 * @param  tw Rigid body twist.
 * @param  pt Point on the rigid body.
 * @return Velocity of the point.
 */
///@{
inline Vec2r VelocityAtPoint(const Twist2r& tw, const Vec2r& pt) {
  return Linear(tw) + Angular(tw) * Vec2r(-pt(1), pt(0));
}

inline Vec3r VelocityAtPoint(const Twist3r& tw, const Vec3r& pt) {
  return Linear(tw) + Angular(tw).cross(pt);
}
///@}

/**
 * @name Dual twist computation functions
 * @brief Returns the dual twist on a rigid body given a dual velocity at a
 * point.
 *
 * This function is the adjoint of the VelocityAtPoint function (for a given
 * point).
 *
 * @attention The dual velocity, point, and the dual twist are expressed in the
 * same frame of reference.
 *
 * @param  f  Dual velocity at a point.
 * @param  pt Point on the rigid body.
 * @return Dual twist on the rigid body.
 */
///@{
inline Twist2r DualTwistAtPoint(const Vec2r& f, const Vec2r& pt) {
  return Twist2r(Linear(f)(0), Linear(f)(1), pt(0) * f(1) - pt(1) * f(0));
}

inline Twist3r DualTwistAtPoint(const Vec3r& f, const Vec3r& pt) {
  Twist3r wr;
  Linear(wr) = Linear(f);
  Angular(wr) = pt.cross(Linear(f));
  return wr;
}
///@}

}  // namespace dgd

#endif  // DGD_UTILS_TRANSFORMATIONS_H_
