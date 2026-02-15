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
 * @brief 2D/3D half-space class.
 */

#ifndef DGD_GEOMETRY_HALFSPACE_H
#define DGD_GEOMETRY_HALFSPACE_H

#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief 2D/3D half-space class.
 *
 * @note The half-space center is at the origin, with the normal vector pointing
 * along the positive y-axis (for 2D) or the positive z-axis (for 3D).
 *
 * @tparam dim Dimension of the half-space.
 */
template <int dim>
struct Halfspace {
  Real margin; /**< Safety margin. */

  /**
   * @param margin Safety margin.
   */
  explicit Halfspace(Real margin = Real(0.0));

  /// @brief Prints information about the half-space.
  void PrintInfo() const;

  /// @brief Returns the dimension of the half-space.
  static constexpr int dimension();
};

template <int dim>
inline Halfspace<dim>::Halfspace(Real margin) : margin(margin) {
  if (margin < Real(0.0)) {
    throw std::domain_error("Margin is negative");
  }
}

template <int dim>
inline void Halfspace<dim>::PrintInfo() const {
  std::cout << "Type: Half-space (dim = " << dim << ")" << std::endl
            << "  Margin: " << margin << std::endl;
}

template <int dim>
constexpr int Halfspace<dim>::dimension() {
  return dim;
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_HALFSPACE_H
