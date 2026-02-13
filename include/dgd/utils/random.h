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
#include <fstream>
#include <random>
#include <stdexcept>

#include "dgd/data_types.h"
#include "dgd/utils/transformations.h"

namespace dgd {

/// @brief Random number generator class.
class Rng {
 public:
  /// @brief Constructs an Rng object with a random seed.
  explicit Rng();

  /// @brief Constructs an Rng object with the specified seed.
  explicit Rng(unsigned int seed);

  /// @brief Sets a true random seed for the generator.
  void SetRandomSeed();

  /// @brief Sets the given seed for the generator.
  void SetSeed(unsigned int seed = 5489u);

  /**
   * @brief Saves the state of the MT generator to a stream.
   *
   * Caller-provided stream state is preserved; the generator state is appended.
   *
   * @param os Stream to save the state.
   */
  void SaveState(std::ostream& os) const;

  /**
   * @brief Loads the state of the MT generator from a stream.
   *
   * Throws on parse failure.
   *
   * @param is Stream to load the state from.
   */
  void LoadState(std::istream& is);

  /// @brief Saves the state of the MT generator to a file.
  void SaveStateToFile(const std::string& filename) const;

  /// @brief Loads the state of the MT generator from a file.
  void LoadStateFromFile(const std::string& filename);

  /**
   * @name Random real number functions
   * @brief Returns a random real number in the specified range.
   */
  ///@{
  Real Random(Real low, Real high);

  Real Random(Real range = Real(1.0));

  Real Random(const std::array<Real, 2>& range);
  ///@}

  /// @brief Returns 1 with probability prob, and 0 with probability 1 - prob.
  int CoinFlip(Real prob = Real(0.5));

  /**
   * @name Random integer functions
   * @brief Returns a random integer in the specified (inclusive) range.
   */
  ///@{
  int RandomInt(int low, int high);

  int RandomInt(const std::array<int, 2>& range);
  ///@}

  /// @brief Returns a sample from a Gaussian distribution.
  Real RandomGaussian(Real mean = Real(0.0), Real stddev = Real(1.0));

  /// @brief Returns a random uniformly distributed unit vector.
  template <int dim>
  Vecr<dim> RandomUnitVector();

  /**
   * @brief Returns a random rotation matrix.
   *
   * @param  ang_max Maximum rotation angle.
   * @return Rotation matrix.
   */
  template <int dim>
  Rotationr<dim> RandomRotation(Real ang_max = kPi);

  /**
   * @name Random transformation functions
   * @brief Sets a random rigid body transformation.
   *
   * @param[in]  low  Lower bound of position.
   * @param[in]  high Upper bound of position.
   * @param[out] tf   Transformation matrix.
   */
  ///@{
  template <int dim>
  void RandomTransform(const Vecr<dim>& low, const Vecr<dim>& high,
                       Matr<dim + 1, dim + 1>& tf);

  template <int hdim>
  void RandomTransform(Real low, Real high, Matr<hdim, hdim>& tf);
  ///@}

 private:
  std::random_device rd_;
  std::mt19937 generator_;
};

inline Rng::Rng() : generator_(rd_()) {}

inline Rng::Rng(unsigned int seed) : generator_(seed) {}

inline void Rng::SetRandomSeed() { generator_.seed(rd_()); }

inline void Rng::SetSeed(unsigned int seed) { generator_.seed(seed); }

inline void Rng::SaveState(std::ostream& os) const {
  if (!(os << generator_)) {
    throw std::runtime_error("Failed to write RNG state to stream");
  }
}

inline void Rng::LoadState(std::istream& is) {
  if (!(is >> generator_)) {
    throw std::runtime_error("Failed to read RNG state from stream");
  }
}

inline void Rng::SaveStateToFile(const std::string& filename) const {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) throw std::runtime_error("Failed to open file: " + filename);
  SaveState(ofs);
  ofs.close();
  if (!ofs) throw std::runtime_error("Failed to write to file: " + filename);
}

inline void Rng::LoadStateFromFile(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) throw std::runtime_error("Failed to open file: " + filename);
  LoadState(ifs);
  if (!ifs) throw std::runtime_error("Failed to read from file: " + filename);
}

inline Real Rng::Random(Real low, Real high) {
  if (low > high) throw std::range_error("Invalid range");
  std::uniform_real_distribution<Real> dis(low, high);
  return dis(generator_);
}

inline Real Rng::Random(Real range) { return Random(-range, range); }

inline Real Rng::Random(const std::array<Real, 2>& range) {
  return Random(range[0], range[1]);
}

inline int Rng::CoinFlip(Real prob) {
  return (Random(Real(0.0), Real(1.0)) < prob ? 1 : 0);
}

inline int Rng::RandomInt(int low, int high) {
  if (low > high) throw std::range_error("Invalid range");
  std::uniform_int_distribution<int> dis(low, high);
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
inline Vecr<dim> Rng::RandomUnitVector() {
  Vecr<dim> n;
  for (int i = 0; i < dim; ++i) n(i) = RandomGaussian();
  const Real norm = n.norm();
  if (norm < kEps) {
    n = Vecr<dim>::UnitX();
  } else {
    n /= norm;
  }
  return n;
}

template <>
inline Rotation2r Rng::RandomRotation(Real ang_max) {
  const Real ang = Random(ang_max);
  const Real c = std::cos(ang), s = std::sin(ang);
  return (Rotation2r() << c, -s, s, c).finished();
}

template <>
inline Rotation3r Rng::RandomRotation(Real ang_max) {
  return AngleAxisToRotation(RandomUnitVector<3>(), Random(ang_max));
}

template <int dim>
inline void Rng::RandomTransform(const Vecr<dim>& low, const Vecr<dim>& high,
                                 Matr<dim + 1, dim + 1>& tf) {
  static_assert((dim == 2) || (dim == 3), "Dimension must be 2 or 3");

  Linear(tf) = RandomRotation<dim>();
  for (int i = 0; i < dim; ++i) tf(i, dim) = Random(low(i), high(i));
  tf.template bottomLeftCorner<1, dim>() = Vecr<dim>::Zero();
  tf(dim, dim) = Real(1.0);
}

template <int hdim>
inline void Rng::RandomTransform(Real low, Real high, Matr<hdim, hdim>& tf) {
  constexpr int dim = hdim - 1;
  RandomTransform<dim>(Vecr<dim>::Constant(low), Vecr<dim>::Constant(high), tf);
}

}  // namespace dgd

#endif  // DGD_UTILS_RANDOM_H_
