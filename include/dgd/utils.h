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
 * @file utils.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-28
 * @brief Utility functions.
 */

#ifndef DGD_UTILS_H_
#define DGD_UTILS_H_

#include <cassert>
#include <cmath>
#include <random>

#include "dgd/data_types.h"

namespace dgd {

namespace {
std::random_device rd;
std::mt19937 gen{rd()};
}  // namespace

/**
 * @brief Sets default seed for the RNG.
 */
inline void SetDefaultSeed() { gen.seed(5489u); }

/**
 * @brief Sets random seed for the RNG.
 */
inline void SetRandomSeed() { gen.seed(rd()); }

/**
 * @brief Returns a uniform random real number.
 *
 * @param  range_from Lower bound.
 * @param  range_to   Upper bound.
 * @return Random real number.
 */
inline Real Random(Real range_from, Real range_to) {
  assert(range_from < range_to);
  std::uniform_real_distribution<Real> dis(range_from, range_to);
  return dis(gen);
}

/**
 * @brief Returns a uniform random real number with zero mean.
 *
 * @param  range Maximum absolute value of random number.
 * @return Random real number.
 */
inline Real Random(Real range) { return Random(-range, range); }

/**
 * @name Alignment rotation functions
 * @brief Sets the rotation matrix such that the given normal vector is mapped
 * to the z (\f$e_2\f$ or \f$e_3\f$) axis.
 *
 * @param[in]  n   Normal vector (with unit 2-norm) to be oriented.
 * @param[out] rot Rotation matrix.
 */
///@{
inline void RotationToZAxis(const Vec2f& n, Rot2f& rot) {
  rot(0, 0) = n(1);
  rot(1, 0) = n(0);
  rot(0, 1) = -n(0);
  rot(1, 1) = n(1);
}

inline void RotationToZAxis(const Vec3f& n, Rot3f& rot) {
  Vec3f axis{n + Vec3f::UnitZ()};
  const Real norm{axis.norm()};
  if (norm > kEps) {
    axis = axis / norm;
    rot = Real(2.0) * axis * axis.transpose() - Rot3f::Identity();
  } else
    rot = Vec3f{1.0, -1.0, -1.0}.asDiagonal();
}
///@}

/**
 * @brief Sets the rotation matrix using ZYX Euler angles.
 *
 * @param[in]  euler ZYX Euler angles, in the form (roll, pitch, yaw).
 * @param[out] rot   Rotation matrix.
 */
inline void EulerToRotation(const Vec3f& euler, Rot3f& rot) {
  const Eigen::AngleAxis<Real> R(euler(0), Vec3f::UnitX());
  const Eigen::AngleAxis<Real> P(euler(1), Vec3f::UnitY());
  const Eigen::AngleAxis<Real> Y(euler(2), Vec3f::UnitZ());
  rot = (Y * P * R).matrix();
}

/**
 * @name Random rotation functions
 * @brief Sets a random rotation matrix.
 *
 * @param[out] rot Rotation matrix.
 */
///@{
inline void RandomRotation(Rot2f& rot) {
  const Real angle{Random(kPi)};
  rot(0, 0) = std::cos(angle);
  rot(1, 0) = std::sin(angle);
  rot(0, 1) = -rot(1, 0);
  rot(1, 1) = rot(0, 0);
}

inline void RandomRotation(Rot3f& rot) {
  Vec3f euler;
  euler << Random(kPi), Random(kPi / Real(2.0)), Random(kPi);
  EulerToRotation(euler, rot);
}
///@}

/**
 * @brief Sets a random rigid body transformation matrix.
 *
 * @tparam     dim        Dimension.
 * @param[in]  range_from Lower bound of position.
 * @param[in]  range_to   Upper bound of position.
 * @param[out] tf         Transformation matrix.
 */
template <int dim>
inline void RandomRigidBodyTransform(const Vecf<dim>& range_from,
                                     const Vecf<dim>& range_to,
                                     Transformf<dim>& tf) {
  assert((range_from.array() < range_to.array()).all());
  Rotf<dim> rot;
  RandomRotation(rot);
  tf.template block<dim, dim>(0, 0) = rot;
  for (int i = 0; i < dim; ++i) tf(i, dim) = Random(range_from(i), range_to(i));
  tf.template block<1, dim>(dim, 0) = Vecf<dim>::Zero().transpose();
  tf(dim, dim) = Real(1.0);
}

/**
 * @brief Sets a random rigid body transformation matrix with position in a cube
 * range.
 *
 * @tparam     dim        Dimension.
 * @param      range_from Lower bound of position.
 * @param      range_to   Upper bound of position.
 * @param[out] tf         Transformation matrix.
 */
template <int dim>
inline void RandomRigidBodyTransform(Real range_from, Real range_to,
                                     Transformf<dim>& tf) {
  RandomRigidBodyTransform<dim>(range_from * Vecf<dim>::Ones(),
                                range_to * Vecf<dim>::Ones(), tf);
}

}  // namespace dgd

#endif  // DGD_UTILS_H_
