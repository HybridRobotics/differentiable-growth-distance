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

#include <cmath>
#include <random>
#include <stdexcept>

#include "dgd/data_types.h"

namespace dgd {

/**
 * RNG utility functions.
 */

namespace {
std::random_device random_device;
std::mt19937 generator{random_device()};
}  // namespace

/**
 * @brief Sets default seed for the RNG.
 */
inline void SetDefaultSeed() { generator.seed(5489u); }

/**
 * @brief Sets random seed for the RNG.
 */
inline void SetRandomSeed() { generator.seed(random_device()); }

/**
 * @brief Returns a uniform random real number.
 *
 * @param  range_from Lower bound.
 * @param  range_to   Upper bound.
 * @return Random real number.
 */
inline Real Random(Real range_from, Real range_to) {
  if (range_from >= range_to) {
    throw std::range_error("Invalid range");
  }
  std::uniform_real_distribution<Real> dis(range_from, range_to);
  return dis(generator);
}

/**
 * @brief Returns a uniform random real number with zero mean.
 *
 * @param  range Maximum absolute value of random number.
 * @return Random real number.
 */
inline Real Random(Real range) { return Random(-range, range); }

void EulerToRotation(const Vec3r& euler, Rotation3r& rot);

/**
 * @name Random rotation functions
 * @brief Sets a random rotation matrix.
 *
 * @param[out] rot Rotation matrix.
 */
///@{
inline void RandomRotation(Rotation2r& rot) {
  const Real angle{Random(kPi)};
  rot(0, 0) = std::cos(angle);
  rot(1, 0) = std::sin(angle);
  rot(0, 1) = -rot(1, 0);
  rot(1, 1) = rot(0, 0);
}

inline void RandomRotation(Rotation3r& rot) {
  Vec3r euler{Random(kPi), Random(kPi / Real(2.0)), Random(kPi)};
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
inline void RandomRigidBodyTransform(const Vecr<dim>& range_from,
                                     const Vecr<dim>& range_to,
                                     Transformr<dim>& tf) {
  if ((range_from.array() >= range_to.array()).any()) {
    throw std::range_error("Invalid range");
  }
  Rotationr<dim> rot;
  RandomRotation(rot);
  tf.template block<dim, dim>(0, 0) = rot;
  for (int i = 0; i < dim; ++i) tf(i, dim) = Random(range_from(i), range_to(i));
  tf.template block<1, dim>(dim, 0) = Vecr<dim>::Zero().transpose();
  tf(dim, dim) = Real(1.0);
}

/**
 * @brief Sets a random rigid body transformation matrix with position in a cube
 * range.
 *
 * @tparam     dim        Dimension.
 * @param[in]  range_from Lower bound of position.
 * @param[in]  range_to   Upper bound of position.
 * @param[out] tf         Transformation matrix.
 */
template <int dim>
inline void RandomRigidBodyTransform(Real range_from, Real range_to,
                                     Transformr<dim>& tf) {
  RandomRigidBodyTransform<dim>(Vecr<dim>::Constant(range_from),
                                Vecr<dim>::Constant(range_to), tf);
}

/**
 * Math utility functions.
 */

/**
 * @brief Sets the rotation matrix using ZYX Euler angles.
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

}  // namespace dgd

#endif  // DGD_UTILS_H_
