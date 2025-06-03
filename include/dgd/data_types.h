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
 * @file data_types.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-18
 * @brief Interface for data types and constants.
 */

#ifndef DGD_DATA_TYPES_H_
#define DGD_DATA_TYPES_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>

namespace dgd {

/**
 * @brief Precision of floating-point real numbers.
 */
#ifdef DGD_USE_32BIT_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

/**
 * @brief Infinity.
 *
 * @note This constant is only used for the initial lower bound for the growth
 * distance algorithm. For a pair of convex sets with circumradii \f$R_1\f$ and
 * \f$R_2\f$, kInf can be any number greater than \f$R_1 + R_2\f$.
 */
constexpr Real kInf{std::numeric_limits<Real>::infinity()};
constexpr Real kEps{
    std::numeric_limits<Real>::epsilon()}; /**< Machine epsilon. */
constexpr Real kSqrtEps{
    std::sqrt(kEps)}; /**< Square root of machine epsilon. */
constexpr Real kPi{static_cast<Real>(EIGEN_PI)}; /**< Pi. */

/**
 * @brief Fixed-dimension vector.
 *
 * @tparam T   Floating-point type.
 * @tparam dim Dimension of the vector.
 */
template <typename T, int dim>
using Vec = Eigen::Vector<T, dim>;
template <int dim>
using Vecr = Vec<Real, dim>;
typedef Vecr<2> Vec2r;
typedef Vecr<3> Vec3r;

/**
 * @brief Alias for Eigen::MatrixBase.
 */
template <typename Derived>
using MatrixBase = Eigen::MatrixBase<Derived>;

/**
 * @brief Fixed-size matrix.
 *
 * @note Vectors are stored along columns.
 *
 * @tparam row Number of rows of the matrix.
 * @tparam col Number of columns of the matrix.
 */
template <int row, int col>
using Matr = Eigen::Matrix<Real, row, col>;

/**
 * @brief Rotation matrix.
 *
 * @tparam dim Dimension of the space.
 */
template <int dim>
using Rotationr = Matr<dim, dim>;
typedef Rotationr<2> Rotation2r;
typedef Rotationr<3> Rotation3r;

/**
 * @brief Rigid body transformation matrix.
 *
 * Rigid body transformations are of the form:
 * \f[
 * \left( \matrix{ R & p \cr 0 & 1 \cr} \right),
 * \f]
 * where \f$R\f$ is the rotation matrix and \f$p\f$ is the position vector.
 *
 * @tparam dim Dimension of the space.
 */
template <int dim>
using Transformr = Matr<dim + 1, dim + 1>;
typedef Transformr<2> Transform2r;
typedef Transformr<3> Transform3r;

}  // namespace dgd

#endif  // DGD_DATA_TYPES_H_
