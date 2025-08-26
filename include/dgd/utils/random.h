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
 * @brief Random number generation utilities.
 */

#ifndef DGD_UTILS_RANDOM_H_
#define DGD_UTILS_RANDOM_H_

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/utils/transformations.h"

namespace dgd {

/// @brief Random number generator class.
class Rng {
 public:
  explicit Rng();

  /// @brief Sets a default seed for the generator.
  void SetDefaultSeed();

  /// @brief Sets a true random seed for the generator.
  void SetRandomSeed();

  /**
   * @name Random real number functions
   * @brief Returns a random real number in the specified range.
   */
  ///@{
  Real Random(Real range_low, Real range_high);

  Real Random(Real range = 1.0);

  Real Random(const std::array<Real, 2>& range);
  ///@}

  /// @brief Returns 1 with probability prob, and 0 with probability 1 - prob.
  int CoinFlip(Real prob = 0.5);

  /**
   * @name Random integer functions
   * @brief Returns a random integer in the specified range.
   */
  ///@{
  int RandomInt(int range_low, int range_high);

  int RandomInt(const std::array<int, 2>& range);
  ///@}

  /// @brief Returns a sample from a Gaussian distribution.
  Real RandomGaussian(Real mean = 0.0, Real stddev = 1.0);

  /// @brief Sets a random uniformly distributed unit vector.
  template <int dim>
  void RandomUnitVector(Vecr<dim>& n);

  /**
   * @name Random rotation functions
   * @brief Sets a random rotation matrix.
   *
   * @param[out] rot     Rotation matrix.
   * @param[in]  ang_max Maximum rotation angle.
   */
  ///@{
  void RandomRotation(Rotation2r& rot, Real ang_max = kPi);

  void RandomRotation(Rotation3r& rot, Real ang_max = kPi);
  ///@}

  /**
   * @brief Sets a random rigid body transformation matrix.
   *
   * @param[in]  range_low  Lower bound of position.
   * @param[in]  range_high Upper bound of position.
   * @param[out] tf         Transformation matrix.
   */
  ///@{
  template <int dim>
  void RandomTransform(const Vecr<dim>& range_low, const Vecr<dim>& range_high,
                       Matr<dim + 1, dim + 1>& tf);

  template <int hdim>
  void RandomTransform(Real range_low, Real range_high, Matr<hdim, hdim>& tf);
  ///@}

 private:
  std::random_device rd_;
  std::mt19937 generator_;
};

inline Rng::Rng() : generator_(rd_()) {}

inline void Rng::SetDefaultSeed() { generator_.seed(5489u); }

inline void Rng::SetRandomSeed() { generator_.seed(rd_()); }

inline Real Rng::Random(Real range_low, Real range_high) {
  if (range_low > range_high) {
    throw std::range_error("Invalid range");
  }
  std::uniform_real_distribution<Real> dis(range_low, range_high);
  return dis(generator_);
}

inline Real Rng::Random(Real range) { return Random(-range, range); }

inline Real Rng::Random(const std::array<Real, 2>& range) {
  return Random(range[0], range[1]);
}

inline int Rng::CoinFlip(Real prob) { return Random(0.0, 1.0) < prob ? 1 : 0; }

inline int Rng::RandomInt(int range_low, int range_high) {
  if (range_low > range_high) {
    throw std::range_error("Invalid range");
  }
  std::uniform_int_distribution<int> dis(range_low, range_high);
  return dis(generator_);
}

inline int Rng::RandomInt(const std::array<int, 2>& range) {
  return RandomInt(range[0], range[1]);
}

inline Real Rng::RandomGaussian(Real mean, Real stddev) {
  std::normal_distribution<Real> dis(mean, stddev);
  return dis(generator_);
}

template <int dim>
inline void Rng::RandomUnitVector(Vecr<dim>& n) {
  for (int i = 0; i < dim; ++i) n(i) = RandomGaussian();
  const Real norm = n.norm();
  if (norm < kEps) {
    n = Vecr<dim>::UnitX();
  } else {
    n /= norm;
  }
}

inline void Rng::RandomRotation(Rotation2r& rot, Real ang_max) {
  const Real ang = Random(ang_max);
  rot(0, 0) = std::cos(ang);
  rot(1, 0) = std::sin(ang);
  rot(0, 1) = -rot(1, 0);
  rot(1, 1) = rot(0, 0);
}

inline void Rng::RandomRotation(Rotation3r& rot, Real ang_max) {
  Vec3r n;
  RandomUnitVector(n);
  AngleAxisToRotation(n, Random(ang_max), rot);
}

template <int dim>
inline void Rng::RandomTransform(const Vecr<dim>& range_low,
                                 const Vecr<dim>& range_high,
                                 Matr<dim + 1, dim + 1>& tf) {
  static_assert((dim == 2) || (dim == 3), "Dimension must be 2 or 3");
  if ((range_low.array() > range_high.array()).any()) {
    throw std::range_error("Invalid range");
  }
  Rotationr<dim> rot;
  RandomRotation(rot);
  Linear(tf) = rot;
  for (int i = 0; i < dim; ++i) {
    tf(i, dim) = Random(range_low(i), range_high(i));
  }
  tf.template block<1, dim>(dim, 0) = Vecr<dim>::Zero().transpose();
  tf(dim, dim) = 1.0;
}

template <int hdim>
inline void Rng::RandomTransform(Real range_low, Real range_high,
                                 Matr<hdim, hdim>& tf) {
  RandomTransform<hdim - 1>(Vecr<hdim - 1>::Constant(range_low),
                            Vecr<hdim - 1>::Constant(range_high), tf);
}

}  // namespace dgd

#endif  // DGD_UTILS_RANDOM_H_
