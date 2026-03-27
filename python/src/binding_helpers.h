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

#ifndef DGD_PYTHON_SRC_BINDING_HELPERS_H_
#define DGD_PYTHON_SRC_BINDING_HELPERS_H_

#include <sstream>
#include <string>
#include <vector>

#include "dgd/data_types.h"

namespace dgd {

namespace pybind_helpers {

using VecXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;

template <int dim>
using MatXdr = Eigen::Matrix<Real, Eigen::Dynamic, dim, Eigen::RowMajor>;
using MatX2r = MatXdr<2>;
using MatX3r = MatXdr<3>;

// ------------------------------------------------------------------
// Eigen matrix to std::vector conversion functions
// ------------------------------------------------------------------

template <int dim>
inline std::vector<Vecr<dim>> MatXToVec(const MatXdr<dim>& points) {
  std::vector<Vecr<dim>> out(static_cast<size_t>(points.rows()));
  for (int i = 0; i < points.rows(); ++i) {
    out[static_cast<size_t>(i)] = points.row(i).transpose();
  }
  return out;
}

template <int dim>
inline MatXdr<dim> VecToMatX(const std::vector<Vecr<dim>>& points) {
  MatXdr<dim> out(static_cast<int>(points.size()), dim);
  for (int i = 0; i < out.rows(); ++i) {
    out.row(i) = points[static_cast<size_t>(i)].transpose();
  }
  return out;
}

inline std::vector<Real> VecXrToVecr(const VecXr& values) {
  std::vector<Real> out(static_cast<size_t>(values.size()));
  for (int i = 0; i < values.size(); ++i) {
    out[static_cast<size_t>(i)] = values(i);
  }
  return out;
}

inline VecXr VecrToVecXr(const std::vector<Real>& values) {
  VecXr out(static_cast<int>(values.size()));
  for (int i = 0; i < out.size(); ++i) out(i) = values[static_cast<size_t>(i)];
  return out;
}

inline std::vector<int> VecXiToVeci(const VecXi& values) {
  std::vector<int> out(static_cast<size_t>(values.size()));
  for (int i = 0; i < values.size(); ++i) {
    out[static_cast<size_t>(i)] = values(i);
  }
  return out;
}

inline VecXi VeciToVecXi(const std::vector<int>& values) {
  VecXi out(static_cast<int>(values.size()));
  for (int i = 0; i < out.size(); ++i) out(i) = values[static_cast<size_t>(i)];
  return out;
}

// ------------------------------------------------------------------
// Formatting functions
// ------------------------------------------------------------------

template <int dim>
inline std::string VectorToLiteral(const Vecr<dim>& v) {
  std::ostringstream oss;
  oss << "[" << v(0);
  for (int i = 1; i < dim; ++i) oss << ", " << v(i);
  oss << "]";
  return oss.str();
}

template <int row, int col>
inline std::string MatrixToLiteral(const Matr<row, col>& m) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < row; ++i) {
    if (i > 0) oss << ", ";
    oss << "[" << m(i, 0);
    for (int j = 1; j < col; ++j) oss << ", " << m(i, j);
    oss << "]";
  }
  oss << "]";
  return oss.str();
}

}  // namespace pybind_helpers

}  // namespace dgd

#endif  // DGD_PYTHON_SRC_BINDING_HELPERS_H_
