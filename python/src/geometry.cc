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
 * @file geometry.cc
 * @brief Bindings for geometry base classes and helper structs.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

#include "dgd/geometry/convex_set.h"

namespace py = pybind11;
using namespace dgd;

namespace {

template <int dim>
void bind_convex_set_helpers(py::module_& m) {
  py::class_<SupportFunctionHint<dim>>(
      m, ("SupportFunctionHint" + std::to_string(dim)).c_str(),
      "Support function hint.")
      .def(py::init<>())
      .def_readwrite("n_prev", &SupportFunctionHint<dim>::n_prev,
                     "Normal vector at the previous iteration.")
      .def_readwrite("idx_ws", &SupportFunctionHint<dim>::idx_ws,
                     "Integer warm-start hint.");

  py::class_<NormalPair<dim>>(m, ("NormalPair" + std::to_string(dim)).c_str(),
                              "Base point-normal vector pair.")
      .def(py::init<>())
      .def_readwrite("z", &NormalPair<dim>::z, "Base point.")
      .def_readwrite("n", &NormalPair<dim>::n, "Normal vector.");

  py::class_<BasePointHint<dim>>(
      m, ("BasePointHint" + std::to_string(dim)).c_str(), "Base point hint.")
      .def(py::init<>())
      .def_readwrite("bc", &BasePointHint<dim>::bc,
                     "Barycentric coordinates for the base point corresponding "
                     "to the simplex.")
      .def_readwrite(
          "idx", &BasePointHint<dim>::idx,
          "Index hints for the base point corresponding to the simplex.");

  auto sph_cls = py::class_<SupportPatchHull<dim>>(
                     m, ("SupportPatchHull" + std::to_string(dim)).c_str(),
                     "Support patch hull.")
                     .def(py::init<>())
                     .def_readwrite("aff_dim", &SupportPatchHull<dim>::aff_dim,
                                    "Affine dimension of the support patch.");

  auto ncs_cls = py::class_<NormalConeSpan<dim>>(
                     m, ("NormalConeSpan" + std::to_string(dim)).c_str(),
                     "Normal cone span.")
                     .def(py::init<>())
                     .def_readwrite("span_dim", &NormalConeSpan<dim>::span_dim,
                                    "Dimension of the normal cone span.");

  if constexpr (dim == 3) {
    sph_cls.def_readwrite("basis", &SupportPatchHull<dim>::basis,
                          "Basis for the support patch affine hull.");
    ncs_cls.def_readwrite("basis", &NormalConeSpan<dim>::basis,
                          "Basis for the normal cone span (excluding n).");
  }
}

template <int dim>
void bind_convex_set(py::module_& m) {
  py::class_<ConvexSet<dim>, std::shared_ptr<ConvexSet<dim>>>(
      m, ("ConvexSet" + std::to_string(dim)).c_str(),
      "Abstract convex set interface.")
      .def(
          "support_function",
          [](const ConvexSet<dim>& self, const Vecr<dim>& n,
             SupportFunctionHint<dim>* hint) {
            Vecr<dim> sp = Vecr<dim>::Zero();
            const Real sv = self.SupportFunction(n, sp, hint);
            return py::make_tuple(sv, sp);
          },
          py::arg("n"), py::arg("hint") = nullptr,
          "Computes the support function value and a support point at n. "
          "Returns (sv, sp).")
      .def(
          "support_function_hess",
          [](const ConvexSet<dim>& self, const Vecr<dim>& n,
             SupportFunctionHint<dim>* hint) {
            SupportFunctionDerivatives<dim> deriv;
            deriv.sp.setZero();
            deriv.d_sp_n.setZero();
            deriv.differentiable = false;
            const Real sv = self.SupportFunction(n, deriv, hint);
            return py::make_tuple(sv, deriv.sp, deriv.differentiable,
                                  deriv.d_sp_n);
          },
          py::arg("n"), py::arg("hint") = nullptr,
          "Computes the support function value and its derivatives at n. "
          "Returns (sv, sp, differentiable, d_sp_n).")
      .def(
          "compute_local_geometry",
          [](const ConvexSet<dim>& self, const Vecr<dim>& z, const Vecr<dim>& n,
             BasePointHint<dim>* hint) {
            NormalPair<dim> zn;
            zn.z = z;
            zn.n = n;
            SupportPatchHull<dim> sph;
            NormalConeSpan<dim> ncs;
            self.ComputeLocalGeometry(zn, sph, ncs, hint);
            if constexpr (dim == 3) {
              if (sph.aff_dim != 1) sph.basis.setZero();
              if (ncs.span_dim != 2) ncs.basis.setZero();
            }
            return py::make_tuple(sph, ncs);
          },
          py::arg("z"), py::arg("n"), py::arg("hint") = nullptr,
          "Computes the local geometry at a base point-normal vector pair. "
          "Returns (support_patch_hull, normal_cone_span).")
      .def(
          "projection_derivative",
          [](const ConvexSet<dim>& self, const Vecr<dim>& p,
             const Vecr<dim>& pi, BasePointHint<dim>* hint) {
            Matr<dim, dim> d_pi_p = Matr<dim, dim>::Zero();
            const bool differentiable =
                self.ProjectionDerivative(p, pi, d_pi_p, hint);
            if (!differentiable) d_pi_p.setZero();
            return py::make_tuple(differentiable, d_pi_p);
          },
          py::arg("p"), py::arg("pi"), py::arg("hint") = nullptr,
          "Computes the derivative of the projection map at an exterior point. "
          "Returns (differentiable, d_pi_p).")
      .def(
          "bounds",
          [](const ConvexSet<dim>& self) {
            Vecr<dim> min = Vecr<dim>::Zero();
            Vecr<dim> max = Vecr<dim>::Zero();
            const Real diagonal = self.Bounds(&min, &max);
            return py::make_tuple(diagonal, min, max);
          },
          "Computes the AABB bounds. Returns (diagonal, min, max).")
      .def("require_unit_normal", &ConvexSet<dim>::RequireUnitNormal,
           "Returns the normalization requirement for support normals.")
      .def("is_polytopic", &ConvexSet<dim>::IsPolytopic,
           "Returns True if the set is polytopic.")
      .def("print_info", &ConvexSet<dim>::PrintInfo,
           "Prints information about the convex set.")
      .def_property("eps_p", &ConvexSet<dim>::eps_p, &ConvexSet<dim>::set_eps_p,
                    "Primal solution geometry tolerance.")
      .def_property("eps_d", &ConvexSet<dim>::eps_d, &ConvexSet<dim>::set_eps_d,
                    "Dual solution geometry tolerance.")
      .def_property("inradius", &ConvexSet<dim>::inradius,
                    &ConvexSet<dim>::set_inradius,
                    "Inradius lower bound at the origin.")
      .def_static("dimension", &ConvexSet<dim>::dimension,
                  "Returns the convex set dimension.")
      .def_static("eps_sp", &ConvexSet<dim>::eps_sp,
                  "Returns the support point differentiability tolerance.");
}

}  // namespace

void bind_geometry(py::module_& m) {
  bind_convex_set_helpers<2>(m);
  bind_convex_set_helpers<3>(m);

  bind_convex_set<2>(m);
  bind_convex_set<3>(m);
}
