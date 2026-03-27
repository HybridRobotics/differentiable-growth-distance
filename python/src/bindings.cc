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
 * @file bindings.cc
 * @brief Top-level pybind11 module definition for the DGD library.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations - defined in the individual translation units.
void bind_types(py::module_& m);
void bind_settings(py::module_& m);
void bind_output(py::module_& m);
void bind_geometry(py::module_& m);
void bind_algorithms(py::module_& m);
void bind_utils(py::module_& m);

PYBIND11_MODULE(_dgd_core, m) {
  m.doc() = "Differentiable growth distance algorithm for convex sets.";

  bind_types(m);
  bind_settings(m);
  bind_output(m);
  bind_geometry(m);
  bind_algorithms(m);
  bind_utils(m);
}
