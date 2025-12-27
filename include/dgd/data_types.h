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
 * @brief Data types and constants.
 */

#ifndef DGD_DATA_TYPES_H_
#define DGD_DATA_TYPES_H_

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace dgd {

/// @brief Precision of floating-point real numbers.
#ifdef DGD_USE_32BIT_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

/// @brief Infinity.
constexpr Real kInf = std::numeric_limits<Real>::infinity();
/// @brief Machine epsilon.
constexpr Real kEps = std::numeric_limits<Real>::epsilon();
/// @brief Square root of machine epsilon.
inline const Real kSqrtEps = std::sqrt(kEps);
/// @brief Pi.
inline const Real kPi = static_cast<Real>(EIGEN_PI);

/**
 * @brief Fixed-size vector.
 *
 * @tparam T   Floating-point type.
 * @tparam dim Dimension of the vector.
 */
template <typename T, int dim>
using Vec = Eigen::Matrix<T, dim, 1>;
template <int dim>
using Vecr = Vec<Real, dim>;
using Vec2r = Vecr<2>;
using Vec3r = Vecr<3>;

/// @brief Dynamic-size real-valued vector.
using VecXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

/**
 * @brief Fixed-size real-valued matrix.
 *
 * @note Vectors are stored along columns.
 *
 * @tparam row Number of rows.
 * @tparam col Number of columns.
 */
template <int row, int col>
using Matr = Eigen::Matrix<Real, row, col>;

/// @brief Dynamic row-size real-valued matrix.
template <int dim>
using MatXr = Eigen::Matrix<Real, dim, Eigen::Dynamic>;
/// @brief Dynamic-size real-valued matrix.
using MatXXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Rotation matrix.
 *
 * @tparam dim Dimension of the space.
 */
template <int dim>
using Rotationr = Matr<dim, dim>;
using Rotation2r = Rotationr<2>;
using Rotation3r = Rotationr<3>;

/**
 * @brief Rigid body transformation matrix.
 *
 * Rigid body transformations are of the form:
 * \f[
 * \left( \matrix{ R & p \cr 0 & 1 \cr} \right),
 * \f]
 * where \f$R\f$ is the rotation matrix and \f$p\f$ is the translation vector.
 *
 * @tparam dim Dimension of the space.
 */
template <int dim>
using Transformr = Matr<dim + 1, dim + 1>;
using Transform2r = Transformr<2>;
using Transform3r = Transformr<3>;

/**
 * @name Affine and linear block functions
 * @brief Affine and linear block functions for rigid body transformations.
 */
///@{
template <int hdim>
inline auto Affine(Matr<hdim, hdim>& tf)
    -> decltype(tf.template topRightCorner<hdim - 1, 1>()) {
  return tf.template topRightCorner<hdim - 1, 1>();
}

template <int hdim>
inline auto Affine(const Matr<hdim, hdim>& tf)
    -> decltype(tf.template topRightCorner<hdim - 1, 1>()) {
  return tf.template topRightCorner<hdim - 1, 1>();
}

template <int hdim>
inline auto Linear(Matr<hdim, hdim>& tf)
    -> decltype(tf.template topLeftCorner<hdim - 1, hdim - 1>()) {
  return tf.template topLeftCorner<hdim - 1, hdim - 1>();
}

template <int hdim>
inline auto Linear(const Matr<hdim, hdim>& tf)
    -> decltype(tf.template topLeftCorner<hdim - 1, hdim - 1>()) {
  return tf.template topLeftCorner<hdim - 1, hdim - 1>();
}
///@}

}  // namespace dgd

#endif  // DGD_DATA_TYPES_H_
