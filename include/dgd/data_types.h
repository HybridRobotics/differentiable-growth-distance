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
 * @brief Data types and constants.
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
 * distance algorithm. For a pair of convex sets, it can be any number greater
 * than the sum of their circumradii.
 */
constexpr Real kInf = std::numeric_limits<Real>::infinity();
constexpr Real kEps =
    std::numeric_limits<Real>::epsilon(); /**< Machine epsilon. */
constexpr Real kSqrtEps = std::sqrt(kEps);
constexpr Real kPi = static_cast<Real>(EIGEN_PI);

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
using VecXr = Vecr<Eigen::Dynamic>;
using Vec2r = Vecr<2>;
using Vec3r = Vecr<3>;

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
using MatXr = Matr<Eigen::Dynamic, Eigen::Dynamic>;

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
 * where \f$R\f$ is the rotation matrix and \f$p\f$ is the position vector.
 *
 * @tparam dim Dimension of the space.
 */
template <int dim>
using Transformr = Matr<dim + 1, dim + 1>;
using Transform2r = Transformr<2>;
using Transform3r = Transformr<3>;

}  // namespace dgd

#endif  // DGD_DATA_TYPES_H_
