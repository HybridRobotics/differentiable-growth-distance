#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "dgd/data_types.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/graham_scan.h"
#include "dgd/utils/random.h"

namespace dgd {

template <int dim>
void PrintTo(const SupportPatchHull<dim>& sph, std::ostream* os) {
  *os << "SupportPatchHull<" << dim << ">{aff_dim=" << sph.aff_dim;
  if constexpr (dim == 3) {
    if (sph.aff_dim == 1) {
      *os << ", basis.col(0)=[" << sph.basis.col(0).transpose() << "]";
    }
  }
  *os << "}";
}

template <int dim>
void PrintTo(const NormalConeSpan<dim>& ncs, std::ostream* os) {
  *os << "NormalConeSpan<" << dim << ">{span_dim=" << ncs.span_dim;
  if constexpr (dim == 3) {
    if (ncs.span_dim == 2) {
      *os << ", basis.col(0)=[" << ncs.basis.col(0).transpose() << "]";
    }
  }
  *os << "}";
}

template <int dim>
void PrintTo(const NormalPair<dim>& zn, std::ostream* os) {
  *os << "NormalPair<" << dim << ">{z=[" << zn.z.transpose() << "], n=["
      << zn.n.transpose() << "]}";
}

}  // namespace dgd

namespace {

using namespace dgd;

// Assertion functions
const Real kTol = kSqrtEps;
const Real kTolBasis = kSqrtEps;

template <int dim>
bool AssertVectorEQ(const Vecr<dim>& v1, const Vecr<dim>& v2, Real tol) {
  return (v1 - v2).template lpNorm<Eigen::Infinity>() < tol;
}

template <int dim>
bool AssertMatrixEQ(const Matr<dim, dim>& m1, const Matr<dim, dim>& m2,
                    Real tol) {
  return (m1 - m2).template lpNorm<Eigen::Infinity>() < tol;
}

template <int dim>
bool AssertSupportPatchHullEQ(const SupportPatchHull<dim>& sph1,
                              const SupportPatchHull<dim>& sph2,
                              const NormalPair<dim>& /*zn*/,
                              Real tol = kTolBasis) {
  // zn is for printing context, not used in comparison.
  if (sph1.aff_dim != sph2.aff_dim) return false;

  if constexpr (dim == 3) {
    if (sph1.aff_dim == 1) {
      const Vec3r cross =
          sph1.basis.col(0).normalized().cross(sph2.basis.col(0).normalized());
      if (cross.norm() > tol) return false;
    }
  }

  return true;
}

template <int dim>
bool AssertNormalConeSpanEQ(const NormalConeSpan<dim>& ncs1,
                            const NormalConeSpan<dim>& ncs2,
                            const NormalPair<dim>& /*zn*/,
                            Real tol = kTolBasis) {
  // zn is for printing context, not used in comparison.
  if (ncs1.span_dim != ncs2.span_dim) return false;

  if constexpr (dim == 3) {
    if (ncs1.span_dim == 2) {
      const Vec3r cross =
          ncs1.basis.col(0).normalized().cross(ncs2.basis.col(0).normalized());
      if (cross.norm() > tol) return false;
    }
  }

  return true;
}

// Test case struct.
template <int dim>
struct LocalGeometryTestCase {
  Vecr<dim> normal;
  Vecr<dim> base_pt;
  SupportPatchHull<dim> sph_true;
  NormalConeSpan<dim> ncs_true;
};

// Utility functions.
template <int dim>
inline Vecr<dim> SupportPointFunction(const ConvexSet<dim>* set,
                                      const Vecr<dim>& n) {
  Vecr<dim> sp;
  set->SupportFunction(n, sp);
  return sp;
}

inline Vec3r DoubleCross(const Vec3r v1, const Vec3r v2, const Vec3r v3) {
  return v1.cross(v2).cross(v3).normalized();
}

struct PolytopeTestData {
  std::vector<Vec3r> vertices;
  std::vector<int> graph;
  std::vector<LocalGeometryTestCase<3>> test_cases;
  std::vector<Matr<3, 3>> s;
  std::vector<Vec3r> bc;
  std::vector<SupportFunctionHint<3>> sfh;
  Real inradius;
};

// Creates polytope test cases for a hexagonal frustum.
PolytopeTestData CreatePolytopeTestData() {
  PolytopeTestData data;

  // Create a hexagonal frustum with regular hexagonal top and bottom faces.
  const Real rt = Real(0.5);   // Radius of the top layer.
  const Real rb = Real(1.0);   // Radius of the bottom layer.
  const Real ht = Real(0.25);  // Height offset of top layer.
  const Real hb = Real(0.75);  // Height offset of bottom layer.

  data.vertices.resize(12);
  for (int i = 0; i < 6; ++i) {
    Real theta = i * kPi / Real(3.0);
    data.vertices[i] = Vec3r(rt * std::cos(theta), rt * std::sin(theta), ht);
    data.vertices[i + 6] = data.vertices[i];
    data.vertices[i + 6](2) = -hb;
  }

  // Graph structure: [nvert, nface, vert_edgeadr[12], edge_localid[...]].
  const int nvert = 12;
  const int nface = 2 * 4 + 6 * 2;

  data.graph.push_back(nvert);  // Number of vertices.
  data.graph.push_back(nface);  // Number of faces (triangulated).

  for (int i = 0; i < nvert; ++i) data.graph.push_back(0);

  // Vertex adjacencies.
  data.graph.insert(data.graph.end(), {1, 2, 3, 4, 5, 6, 7, -1});    // vertex 0
  data.graph.insert(data.graph.end(), {0, 2, 7, 8, -1});             // vertex 1
  data.graph.insert(data.graph.end(), {0, 1, 3, 8, 9, -1});          // vertex 2
  data.graph.insert(data.graph.end(), {0, 2, 4, 9, 10, -1});         // vertex 3
  data.graph.insert(data.graph.end(), {0, 3, 5, 10, 11, -1});        // vertex 4
  data.graph.insert(data.graph.end(), {0, 4, 11, 6, -1});            // vertex 5
  data.graph.insert(data.graph.end(), {0, 5, 7, 8, 9, 10, 11, -1});  // vertex 6
  data.graph.insert(data.graph.end(), {0, 1, 6, 8, -1});             // vertex 7
  data.graph.insert(data.graph.end(), {1, 2, 6, 7, 9, -1});          // vertex 8
  data.graph.insert(data.graph.end(), {2, 3, 6, 8, 10, -1});         // vertex 9
  data.graph.insert(data.graph.end(), {3, 4, 6, 9, 11, -1});  // vertex 10
  data.graph.insert(data.graph.end(), {4, 5, 6, 10, -1});     // vertex 11

  // Vertex edge list indices.
  for (int i = nvert + 2, v = 0; v < nvert - 1; ++v) {
    for (; data.graph[++i] != -1;);
    data.graph[2 + v + 1] = (++i) - (nvert + 2);
  }

  data.inradius = std::min({Real(0.5) * rt, Real(0.5) * rb, ht, hb});

  const Vec3r ez = Vec3r::UnitZ();
  std::vector<Vec3r> n(6);
  for (int i = 0; i < 6; ++i) {
    int i1 = (i + 6) % 12, i2 = (i + 7) % 12;
    n[i] = (data.vertices[i1] - data.vertices[i])
               .cross(data.vertices[i2] - data.vertices[i])
               .normalized();
  }

  const auto& v = data.vertices;
  int aff_dim, span_dim;

  // Case 1: Support patch - hexagon, normal cone - ray.
  aff_dim = 2;
  span_dim = 1;

  //  Three distinct simplex points; bc1, bc2, bc3 > 0.
  data.s.push_back((Matr<3, 3>() << v[0], v[2], v[3]).finished());
  data.bc.push_back({Real(0.3), Real(0.3), Real(0.4)});
  data.sfh.push_back({ez, 4});
  data.test_cases.push_back({ez,
                             data.s.back() * data.bc.back(),
                             {Vec3r::Zero(), aff_dim},
                             {Vec3r::Zero(), span_dim}});

  //  Three distinct simplex points; bci = 0, bcj, bck > 0.
  for (int i = 0; i < 3; ++i) {
    data.s.push_back((Matr<3, 3>() << v[0], v[2], v[4]).finished());
    Vec3r bc;
    bc(i) = Real(0.3);
    bc((i + 1) % 3) = Real(0.7);
    bc((i + 2) % 3) = Real(0.0);
    data.bc.push_back(bc);
    data.sfh.push_back({ez, 5});
    data.test_cases.push_back({ez,
                               data.s.back() * data.bc.back(),
                               {Vec3r::Zero(), aff_dim},
                               {Vec3r::Zero(), span_dim}});
  }

  //  Two distinct simplex points; bc1, bc2, bc3 > 0.
  data.s.push_back((Matr<3, 3>() << v[0], v[2], v[0]).finished());
  data.bc.push_back({Real(0.55), Real(0.3), Real(0.15)});
  data.sfh.push_back({ez, 4});
  data.test_cases.push_back({ez,
                             data.s.back() * data.bc.back(),
                             {Vec3r::Zero(), aff_dim},
                             {Vec3r::Zero(), span_dim}});

  // Case 2: Support patch - hexagon, normal cone - 2D cone.
  span_dim = 2;

  //  Three distinct simplex points; bci = 0, bcj, bck > 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(0, 1, 2), i2(3, 5, 3), i3(4, 2, 0);
    Vec3i in(3, 1, 2);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(i) = Real(0.0);
    bc((i + 1) % 3) = Real(0.6);
    bc((i + 2) % 3) = Real(0.4);
    data.bc.push_back(bc);
    data.sfh.push_back({ez, 9});
    data.test_cases.push_back({ez,
                               data.s.back() * data.bc.back(),
                               {Vec3r::Zero(), aff_dim},
                               {DoubleCross(n[in(i)], ez, ez), span_dim}});
  }

  //  Two distinct simplex points; bc1, bc2, bc3 > 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(7, 9, 11), i2(8, 9, 6), i3(7, 10, 6);
    Vec3i in(1, 3, 5);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(i) = Real(0.05);
    bc((i + 1) % 3) = Real(0.55);
    bc((i + 2) % 3) = Real(0.4);
    data.bc.push_back(bc);
    data.sfh.push_back({-ez, 2});
    data.test_cases.push_back({-ez,
                               data.s.back() * data.bc.back(),
                               {Vec3r::Zero(), aff_dim},
                               {DoubleCross(n[in(i)], ez, ez), span_dim}});
  }

  // Case 3: Support patch - hexagon, normal cone - cone.
  span_dim = 3;

  //  Three distinct simplex points; bci = 1, bcj, bck = 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(8, 6, 3), i2(10, 9, 1), i3(6, 2, 11);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc = Vec3r::Zero();
    bc(i) = Real(1.0);
    data.bc.push_back(bc);
    data.sfh.push_back({-ez, 6});
    data.test_cases.push_back({-ez,
                               data.s.back() * data.bc.back(),
                               {Vec3r::Zero(), aff_dim},
                               {Vec3r::Zero(), span_dim}});
  }

  //  Two distinct simplex points.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(2, 5, 0), i2(5, 0, 0), i3(2, 0, 4);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(0) = (i == 2) ? Real(0.5) : Real(1.0);
    bc(1) = (i == 2) ? Real(0.5) : Real(0.0);
    bc(2) = Real(0.0);
    data.bc.push_back(bc);
    data.sfh.push_back({ez, 1});
    data.test_cases.push_back({ez,
                               data.s.back() * data.bc.back(),
                               {Vec3r::Zero(), aff_dim},
                               {Vec3r::Zero(), span_dim}});
  }

  //  One distinct simplex point; bc1, bc2, bc3 > 0.
  data.s.push_back((Matr<3, 3>() << v[7], v[7], v[7]).finished());
  data.bc.push_back({Real(0.55), Real(0.3), Real(0.15)});
  data.sfh.push_back({-ez, 4});
  data.test_cases.push_back({-ez,
                             data.s.back() * data.bc.back(),
                             {Vec3r::Zero(), aff_dim},
                             {Vec3r::Zero(), span_dim}});

  // Case 4: Support patch - top/bottom edge, normal cone - 2D cone.
  aff_dim = 1;
  span_dim = 2;

  //  Three distinct simplex points; bci = 0, bcj, bck > 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(0, 1, 2), i2(3, 5, 3), i3(4, 2, 0);
    Vec3i ie1(3, 1, 3), ie2(4, 2, 2);
    Vec3i in(3, 1, 2);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(i) = Real(0.0);
    bc((i + 1) % 3) = Real(0.6);
    bc((i + 2) % 3) = Real(0.4);
    data.bc.push_back(bc);
    data.sfh.push_back({ez, 9});
    Vec3r normal = (n[in(i)] + (i + 1) * ez).normalized();
    data.test_cases.push_back({normal,
                               data.s.back() * data.bc.back(),
                               {(v[ie1(i)] - v[ie2(i)]).normalized(), aff_dim},
                               {DoubleCross(normal, ez, normal), span_dim}});
  }

  //  Two distinct simplex points; bc1, bc2, bc3 > 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(7, 9, 11), i2(8, 9, 6), i3(7, 10, 6);
    Vec3i ie1(7, 9, 11), ie2(8, 10, 6);
    Vec3i in(1, 3, 5);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(i) = Real(0.05);
    bc((i + 1) % 3) = Real(0.55);
    bc((i + 2) % 3) = Real(0.4);
    data.bc.push_back(bc);
    data.sfh.push_back({-ez, 2});
    Vec3r normal = (n[in(i)] - (i + 1) * ez).normalized();
    data.test_cases.push_back({normal,
                               data.s.back() * data.bc.back(),
                               {(v[ie1(i)] - v[ie2(i)]).normalized(), aff_dim},
                               {DoubleCross(normal, ez, normal), span_dim}});
  }

  // Case 5: Support patch - top/bottom edge, normal cone - cone.
  aff_dim = 1;
  span_dim = 3;

  //  Three distinct simplex points; bci = 1, bcj, bck = 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(8, 6, 10), i2(9, 11, 1), i3(6, 2, 11);
    Vec3i ie1(8, 11, 10), ie2(9, 6, 11);
    Vec3i in(2, 5, 4);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc = Vec3r::Zero();
    bc(i) = Real(1.0);
    data.bc.push_back(bc);
    data.sfh.push_back({-ez, 6});
    Vec3r normal = (n[in(i)] - (i + 1) * ez).normalized();
    data.test_cases.push_back({normal,
                               data.s.back() * data.bc.back(),
                               {(v[ie1(i)] - v[ie2(i)]).normalized(), aff_dim},
                               {Vec3r::Zero(), span_dim}});
  }

  //  Two distinct simplex points.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(2, 5, 0), i2(3, 0, 0), i3(2, 0, 1);
    Vec3i ie1(2, 5, 0), ie2(3, 0, 1);
    Vec3i in(2, 5, 0);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(0) = (i == 2) ? Real(0.5) : Real(1.0);
    bc(1) = (i == 2) ? Real(0.5) : Real(0.0);
    bc(2) = Real(0.0);
    data.bc.push_back(bc);
    data.sfh.push_back({ez, 1});
    Vec3r normal = (n[in(i)] + (i + 1) * ez).normalized();
    data.test_cases.push_back({normal,
                               data.s.back() * data.bc.back(),
                               {(v[ie1(i)] - v[ie2(i)]).normalized(), aff_dim},
                               {Vec3r::Zero(), span_dim}});
  }

  //  One distinct simplex point; bc1, bc2, bc3 > 0.
  data.s.push_back((Matr<3, 3>() << v[7], v[7], v[7]).finished());
  data.bc.push_back({Real(0.55), Real(0.3), Real(0.15)});
  data.sfh.push_back({-ez, 4});
  data.test_cases.push_back({(n[1] - 2 * ez).normalized(),
                             data.s.back() * data.bc.back(),
                             {(v[7] - v[8]).normalized(), aff_dim},
                             {Vec3r::Zero(), span_dim}});

  // Case 6: Support patch - point, normal cone - cone.
  aff_dim = 0;
  span_dim = 3;

  //  Three distinct simplex points; bci = 1, bcj, bck = 0.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(8, 6, 3), i2(10, 9, 1), i3(6, 2, 11);
    Vec3i in1(1, 2, 4), in2(2, 3, 5);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc = Vec3r::Zero();
    bc(i) = Real(1.0);
    data.bc.push_back(bc);
    data.sfh.push_back({-ez, 6});
    data.test_cases.push_back(
        {(n[in1(i)] + (i + 1) * n[in2(i)] - (2 * i + 1) * ez).normalized(),
         data.s.back() * data.bc.back(),
         {Vec3r::Zero(), aff_dim},
         {Vec3r::Zero(), span_dim}});
  }

  //  Two distinct simplex points.
  for (int i = 0; i < 3; ++i) {
    Vec3i i1(2, 5, 0), i2(5, 0, 0), i3(2, 0, 4);
    Vec3i in1(1, 4, 5), in2(2, 5, 0);
    data.s.push_back((Matr<3, 3>() << v[i1(i)], v[i2(i)], v[i3(i)]).finished());
    Vec3r bc;
    bc(0) = (i == 2) ? Real(0.5) : Real(1.0);
    bc(1) = (i == 2) ? Real(0.5) : Real(0.0);
    bc(2) = Real(0.0);
    data.bc.push_back(bc);
    data.sfh.push_back({ez, 1});
    data.test_cases.push_back(
        {(n[in1(i)] + (i + 1) * n[in2(i)] + (2 * i + 1) * ez).normalized(),
         data.s.back() * data.bc.back(),
         {Vec3r::Zero(), aff_dim},
         {Vec3r::Zero(), span_dim}});
  }

  //  One distinct simplex point; bc1, bc2, bc3 > 0.
  data.s.push_back((Matr<3, 3>() << v[7], v[7], v[7]).finished());
  data.bc.push_back({Real(0.55), Real(0.3), Real(0.15)});
  data.sfh.push_back({-ez, 4});
  data.test_cases.push_back({(n[0] + 2 * n[1] - 3 * ez).normalized(),
                             data.s.back() * data.bc.back(),
                             {Vec3r::Zero(), aff_dim},
                             {Vec3r::Zero(), span_dim}});

  /* Not Implemented. */
  // Case 7: Support patch - side face, normal cone - ray.
  // Case 8: Support patch - side face, normal cone - 2D cone.
  // Case 9: Support patch - side face, normal cone - cone.
  // Case 10: Support patch - side edge, normal cone - 2D cone.

  return data;
}

// Local geometry tests
// 2D convex set tests
//  Ellipse test
TEST(EllipseTest, LocalGeometry) {
  const Real hlx = Real(3.0), hly = Real(2.0), margin = Real(0.0);
  auto set = Ellipse(hlx, hly, margin);

  EXPECT_FALSE(set.IsPolytopic());

  // Generate test cases.
  std::vector<LocalGeometryTestCase<2>> test_cases(4);

  SupportPatchHull<2> sph;
  NormalConeSpan<2> ncs;
  NormalPair<2> zn;

  test_cases[0].normal = Vec2r::UnitX();
  test_cases[1].normal = -Vec2r::UnitY();
  test_cases[2].normal = (Vec2r::UnitX() + Vec2r::UnitY()).normalized();
  test_cases[3].normal = (-Vec2r::UnitX() + Vec2r::UnitY()).normalized();

  for (auto& tc : test_cases) {
    tc.base_pt = SupportPointFunction(&set, tc.normal);
    tc.sph_true.aff_dim = 0;
    tc.ncs_true.span_dim = 1;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<2>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<2>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Polygon test
TEST(PolygonTest, LocalGeometry) {
  Rng rng;
  rng.SetSeed();
  const int npts_max = 100;
  const Real len = Real(5.0);

  const int nruns = 10, nvsamples = 10, nscan = 3;

  // Generate test cases.
  std::vector<LocalGeometryTestCase<2>> test_cases;
  test_cases.reserve(nvsamples * (3 * nscan - 2));

  int aff_dim, span_dim;

  SupportPatchHull<2> sph;
  NormalConeSpan<2> ncs;
  NormalPair<2> zn;

  for (int run = 0; run < nruns; ++run) {
    // Generate random polygon.
    int npts = rng.RandomInt(4, npts_max);
    std::vector<Vec2r> pts, vert;
    pts.push_back(Real(0.01) * len * Vec2r::UnitY());
    pts.push_back(Real(0.01) * len * (-Vec2r::UnitX() - Vec2r::UnitY()));
    pts.push_back(Real(0.01) * len * (Vec2r::UnitX() - Vec2r::UnitY()));
    for (int i = 0; i < npts - 3; ++i) {
      Real r = rng.Random(Real(0.25) * len, len);
      Real theta = rng.Random(Real(0.0), Real(2.0) * kPi);
      pts.push_back(Vec2r(r * std::cos(theta), r * std::sin(theta)));
    }
    GrahamScan(pts, vert);
    Real inradius = ComputePolygonInradius(vert, Vec2r::Zero());

    Real margin = Real(0.0);
    auto set = Polygon(vert, inradius, margin);
    const int nvert = set.nvertices();

    EXPECT_TRUE(set.IsPolytopic());

    test_cases.clear();
    for (int ns = 0; ns < nvsamples; ++ns) {
      // Pick a random vertex and its neighbors.
      int idx = rng.RandomInt(0, nvert - 1);
      int idx_m = (idx == 0) ? nvert - 1 : idx - 1;
      int idx_p = (idx == nvert - 1) ? 0 : idx + 1;

      Vec2r v = vert[idx], vm = vert[idx_m], vp = vert[idx_p];
      Vec2r nm = Vec2r(v(1) - vm(1), vm(0) - v(0)).normalized();
      Vec2r np = Vec2r(vp(1) - v(1), v(0) - vp(0)).normalized();

      ASSERT_GE(nm.dot(v), Real(0.0));
      ASSERT_GE(np.dot(v), Real(0.0));

      // Case 1: Support patch - edge 1, normal cone - cone/ray.
      aff_dim = 1;
      span_dim = 1;

      for (int i = 0; i < nscan; ++i) {
        Real f = static_cast<Real>(i) / static_cast<Real>(nscan - 1);
        test_cases.push_back({nm,
                              v + f * (vm - v),
                              {aff_dim},
                              {((i == 0) || (i == nscan - 1)) ? 2 : span_dim}});
      }

      // Case 2: Support patch - vertex, normal cone - cone.
      aff_dim = 0;
      span_dim = 2;

      for (int i = 1; i < nscan - 1; ++i) {
        Real f = static_cast<Real>(i) / static_cast<Real>(nscan - 1);
        test_cases.push_back({nm + f * (np - nm), v, {aff_dim}, {span_dim}});
      }

      // Case 3: Support patch - edge 2, normal cone - cone/ray.
      aff_dim = 1;
      span_dim = 1;

      for (int i = 0; i < nscan; ++i) {
        Real f = static_cast<Real>(i) / static_cast<Real>(nscan - 1);
        test_cases.push_back({np,
                              v + f * (vp - v),
                              {aff_dim},
                              {((i == 0) || (i == nscan - 1)) ? 2 : span_dim}});
      }
    }

    // Run test cases.
    for (const auto& tc : test_cases) {
      zn.n = tc.normal;
      zn.z = tc.base_pt;

      set.ComputeLocalGeometry(zn, sph, ncs);

      EXPECT_PRED4(AssertSupportPatchHullEQ<2>, sph, tc.sph_true, zn,
                   kTolBasis);
      EXPECT_PRED4(AssertNormalConeSpanEQ<2>, ncs, tc.ncs_true, zn, kTolBasis);
    }

    // Case 4: Positive margin.
    margin = Real(0.5);
    auto set_m = Polygon(std::move(vert), inradius, margin);

    EXPECT_FALSE(set_m.IsPolytopic());

    for (auto& tc : test_cases) {
      tc.base_pt += tc.normal * margin;
      tc.ncs_true.span_dim = 1;
    }

    // Run test cases.
    for (const auto& tc : test_cases) {
      zn.n = tc.normal;
      zn.z = tc.base_pt;

      set_m.ComputeLocalGeometry(zn, sph, ncs);

      EXPECT_PRED4(AssertSupportPatchHullEQ<2>, sph, tc.sph_true, zn,
                   kTolBasis);
      EXPECT_PRED4(AssertNormalConeSpanEQ<2>, ncs, tc.ncs_true, zn, kTolBasis);
    }
  }
}

//  Rectangle test
TEST(RectangleTest, LocalGeometry) {
  Real hlx = Real(3.0), hly = Real(2.0), margin = Real(0.0);
  auto set = Rectangle(hlx, hly, margin);

  EXPECT_TRUE(set.IsPolytopic());

  // Generate test cases.
  std::vector<LocalGeometryTestCase<2>> test_cases;

  std::vector<Vec2r> normals, base_pts;
  int aff_dim, span_dim;

  SupportPatchHull<2> sph;
  NormalConeSpan<2> ncs;
  NormalPair<2> zn;

  // Case 1: Support patch - vertex; normal cone - cone.
  aff_dim = 0;
  span_dim = 2;

  normals.clear();
  normals = {
      Vec2r(Real(1.0), Real(1.0)).normalized(),
      Vec2r(Real(1.0), Real(-1.0)).normalized(),
      Vec2r(Real(-1.0), Real(1.0)).normalized(),
  };

  for (const auto& n : normals) {
    test_cases.push_back(
        {n, SupportPointFunction(&set, n), {aff_dim}, {span_dim}});
  }

  // Case 2: Support patch - edge; normal cone - cone.
  aff_dim = 1;

  normals.clear();
  normals = {
      Vec2r::UnitX(),
      -Vec2r::UnitY(),
  };
  base_pts.clear();
  base_pts = {
      Vec2r(hlx, -hly),
      Vec2r(-hlx, -hly),
  };

  for (size_t i = 0; i < normals.size(); ++i) {
    test_cases.push_back({normals[i], base_pts[i], {aff_dim}, {span_dim}});
  }

  // Case 3: Support patch - edge; normal cone - ray.
  span_dim = 1;

  base_pts.clear();
  base_pts = {
      Vec2r(hlx, Real(0.99) * hly),
      Vec2r(Real(0.99) * hlx, -hly),
  };

  for (size_t i = 0; i < normals.size(); ++i) {
    test_cases.push_back({normals[i], base_pts[i], {aff_dim}, {span_dim}});
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<2>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<2>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 4: Positive margin.
  margin = Real(0.5);
  auto set_m = Rectangle(hlx, hly, margin);

  EXPECT_FALSE(set_m.IsPolytopic());

  for (auto& tc : test_cases) {
    tc.base_pt += tc.normal * margin;
    tc.ncs_true.span_dim = 1;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<2>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<2>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

// 3D convex set tests
//  Cone test
TEST(ConeTest, LocalGeometry) {
  Real ha = kPi / Real(6.0), radius = Real(1.0), margin = Real(0.0);
  Real tha = std::tan(ha);
  Real height = radius / tha;
  auto set = Cone(radius, height, margin);

  EXPECT_FALSE(set.IsPolytopic());

  // Generate test cases.
  std::vector<LocalGeometryTestCase<3>> test_cases;

  std::vector<Vec3r> normals, base_pts;
  std::vector<Real> angs;
  const Vec3r ez = Vec3r::UnitZ();
  int aff_dim, span_dim;

  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs;
  NormalPair<3> zn;

  // Case 1: Support patch - apex; normal cone - cone.
  aff_dim = 0;
  span_dim = 3;

  normals.clear();
  normals = {
      Vec3r::UnitZ(),
      Vec3r(Real(0.5) / tha, Real(0.5) / tha, Real(1.0)).normalized(),
  };

  for (const auto& n : normals) {
    test_cases.push_back({n,
                          (height - set.offset()) * ez,
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Case 2: Support patch - edge; normal cone - cone.
  aff_dim = 1;

  angs.clear();
  angs = {Real(0.0), Real(0.5) * kPi, kPi, Real(1.5) * kPi};

  normals.clear();
  for (const auto ang : angs) {
    normals.push_back(Vec3r(std::cos(ang), std::sin(ang), tha).normalized());
  }

  for (const auto& n : normals) {
    test_cases.push_back({n,
                          (height - set.offset()) * ez,
                          {DoubleCross(n, ez, n), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Case 3: Support patch - edge; normal cone - ray.
  span_dim = 1;

  base_pts.clear();
  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    const Real f = (i + Real(0.5)) / static_cast<Real>(angs.size());
    Vec3r sp = (-set.offset() + (Real(1.0) - f) * height) * ez;
    sp.head<2>() += f * height * tha * normals[i].head<2>().normalized();
    base_pts.push_back(sp);
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {DoubleCross(normals[i], ez, normals[i]), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Case 4: Support patch - edge; normal cone - 2D cone.
  span_dim = 2;

  for (int i = 0; i < static_cast<int>(base_pts.size()); ++i) {
    base_pts[i] = -set.offset() * ez;
    base_pts[i].head<2>() += radius * normals[i].head<2>().normalized();
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back(
        {normals[i],
         base_pts[i],
         {DoubleCross(normals[i], ez, normals[i]), aff_dim},
         {DoubleCross(base_pts[i], ez, normals[i]), span_dim}});
  }

  // Case 5: Support patch - base point; normal cone - 2D cone.
  aff_dim = 0;

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    const Real f = (i + Real(0.5)) / static_cast<Real>(normals.size());
    normals[i] = (normals[i] - f * ez).normalized();
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back(
        {normals[i],
         base_pts[i],
         {Vec3r::Zero(), aff_dim},
         {DoubleCross(base_pts[i], ez, normals[i]), span_dim}});
  }

  // Case 6: Support patch - base disk; normal cone - 2D cone.
  aff_dim = 2;

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases.push_back({-ez,
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {DoubleCross(base_pts[i], ez, ez), span_dim}});
  }

  // Case 7: Support patch - base disk; normal cone - ray.
  span_dim = 1;

  for (int i = 0; i < static_cast<int>(base_pts.size()); ++i) {
    const Real f = (i + Real(0.5)) / static_cast<Real>(base_pts.size());
    base_pts[i].head<2>() *= f;
  }

  for (int i = 0; i < static_cast<int>(base_pts.size()); ++i) {
    test_cases.push_back({-ez,
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 8: Positive margin.
  const Real margin2 = Real(0.5);
  auto set_m = Cone(radius, height, margin2);

  EXPECT_FALSE(set_m.IsPolytopic());

  for (auto& tc : test_cases) {
    tc.base_pt += tc.normal * margin;
    tc.ncs_true.span_dim = 1;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Cuboid test
TEST(CuboidTest, LocalGeometry) {
  Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5), margin = Real(0.0);
  auto set = Cuboid(hlx, hly, hlz, margin);

  EXPECT_TRUE(set.IsPolytopic());

  // Generate test cases.
  std::vector<LocalGeometryTestCase<3>> test_cases;

  std::vector<Vec3r> normals, base_pts;
  int aff_dim, span_dim;

  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs;
  NormalPair<3> zn;

  const Vec3r ex = Vec3r::UnitX();
  const Vec3r ey = Vec3r::UnitY();
  const Vec3r ez = Vec3r::UnitZ();

  // Case 1: Support patch - vertex; normal cone - cone.
  aff_dim = 0;
  span_dim = 3;

  normals.clear();
  normals = {
      (ex + ey + ez).normalized(),
      (-ex + ey + ez).normalized(),
      (ex + ey - ez).normalized(),
      (-ex + ey - ez).normalized(),
  };

  for (const auto& n : normals) {
    test_cases.push_back({n,
                          SupportPointFunction(&set, n),
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Case 2: Support patch - edge; normal cone - cone.
  aff_dim = 1;
  span_dim = 3;

  normals.clear();
  base_pts.clear();

  // Edges along x-axis.
  normals.push_back((ey + ez).normalized());
  base_pts.push_back(Vec3r(hlx, hly, hlz));
  normals.push_back((ey - ez).normalized());
  base_pts.push_back(Vec3r(hlx, hly, -hlz));
  normals.push_back((-ey + ez).normalized());
  base_pts.push_back(Vec3r(-hlx, -hly, hlz));

  // Edges along y-axis.
  normals.push_back((ex + ez).normalized());
  base_pts.push_back(Vec3r(hlx, hly, hlz));
  normals.push_back((ex - ez).normalized());
  base_pts.push_back(Vec3r(-hlx, hly, -hlz));

  // Edges along z-axis.
  normals.push_back((ex + ey).normalized());
  base_pts.push_back(Vec3r(hlx, hly, hlz));
  normals.push_back((ex - ey).normalized());
  base_pts.push_back(Vec3r(hlx, -hly, hlz));

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Case 3: Support patch - edge; normal cone - 2D cone.
  span_dim = 2;

  normals.clear();
  base_pts.clear();

  // Interior points on edges.
  normals.push_back((ey + ez).normalized());
  base_pts.push_back(Vec3r(Real(0.0), hly, hlz));
  normals.push_back((ex + ez).normalized());
  base_pts.push_back(Vec3r(hlx, Real(0.99) * hly, hlz));
  normals.push_back((ex + ey).normalized());
  base_pts.push_back(Vec3r(hlx, hly, -Real(0.5) * hlz));

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    Vec3r basis = (i == 0) ? ex : ((i == 1) ? ey : ez);
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {basis, aff_dim},
                          {basis.cross(normals[i]).normalized(), span_dim}});
  }

  // Case 4: Support patch - face; normal cone - cone.
  aff_dim = 2;
  span_dim = 3;

  normals.clear();
  base_pts.clear();

  // Face normals.
  normals.push_back(ex);
  base_pts.push_back(Vec3r(hlx, hly, hlz));
  normals.push_back(-ey);
  base_pts.push_back(Vec3r(hlx, -hly, hlz));
  normals.push_back(ez);
  base_pts.push_back(Vec3r(hlx, hly, hlz));

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Case 5: Support patch - face; normal cone - ray.
  span_dim = 1;

  base_pts.clear();

  // Interior points on faces
  base_pts.push_back(Vec3r(hlx, Real(0.5) * hly, Real(0.5) * hlz));
  base_pts.push_back(Vec3r(Real(0.5) * hlx, -hly, Real(0.5) * hlz));
  base_pts.push_back(Vec3r(Real(0.5) * hlx, Real(0.5) * hly, hlz));

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 6: Positive margin.
  margin = Real(0.5);
  auto set_m = Cuboid(hlx, hly, hlz, margin);

  EXPECT_FALSE(set_m.IsPolytopic());

  for (auto& tc : test_cases) {
    tc.base_pt += tc.normal * margin;
    tc.ncs_true.span_dim = 1;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Cylinder test
TEST(CylinderTest, LocalGeometry) {
  Real hlx = Real(2.0), radius = Real(2.5), margin = Real(0.0);
  auto set = Cylinder(hlx, radius, margin);

  EXPECT_FALSE(set.IsPolytopic());

  // Generate test cases.
  std::vector<LocalGeometryTestCase<3>> test_cases;

  std::vector<Vec3r> normals, base_pts;
  std::vector<Real> angs;
  const Vec3r ex = Vec3r::UnitX();
  int aff_dim, span_dim;

  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs;
  NormalPair<3> zn;

  // Case 1: Support patch - curved surface point; normal cone - 2D cone.
  aff_dim = 0;
  span_dim = 2;

  angs.clear();
  angs = {Real(0.0), Real(0.5) * kPi, kPi, Real(1.5) * kPi};

  normals.clear();
  for (const auto ang : angs) {
    normals.push_back(
        Vec3r(Real(0.5), std::cos(ang), std::sin(ang)).normalized());
  }
  normals[0](0) = -normals[0](0);
  normals[2](0) = -normals[2](0);

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    Vec3r sp = SupportPointFunction(&set, normals[i]);
    test_cases.push_back({normals[i],
                          sp,
                          {Vec3r::Zero(), aff_dim},
                          {DoubleCross(sp, ex, normals[i]), span_dim}});
  }

  // Case 2: Support patch - rim edge; normal cone - ray.
  aff_dim = 1;
  span_dim = 1;

  for (auto& n : normals) {
    n[0] = Real(0.0);
    n.normalize();
  }

  base_pts.clear();
  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    const Real f = (i + Real(0.5)) / static_cast<Real>(angs.size());
    Vec3r sp;
    sp.tail<2>() = normals[i].tail<2>().normalized() * radius;
    sp(0) = (Real(2.0) * f - Real(1.0)) * hlx;
    base_pts.push_back(sp);
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back(
        {normals[i], base_pts[i], {ex, aff_dim}, {Vec3r::Zero(), span_dim}});
  }

  // Case 3: Support patch - rim edge; normal cone - 2D cone.
  span_dim = 2;

  for (int i = 0; i < static_cast<int>(base_pts.size()); ++i) {
    base_pts[i](0) = (i % 2 == 0) ? hlx : -hlx;
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back(
        {normals[i],
         base_pts[i],
         {ex, aff_dim},
         {DoubleCross(base_pts[i], ex, normals[i]), span_dim}});
  }

  // Case 4: Support patch - disk; normal cone - 2D cone.
  aff_dim = 2;

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    const Real sign = Real(2.0) * (((i + 1) % 2) - Real(0.5));
    normals[i] = sign * ex;
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {DoubleCross(base_pts[i], ex, ex), span_dim}});
  }

  // Case 5: Support patch - disk; normal cone - ray.
  span_dim = 1;

  for (int i = 0; i < static_cast<int>(base_pts.size()); ++i) {
    const Real f = (i + Real(0.5)) / static_cast<Real>(base_pts.size());
    base_pts[i].tail<2>() *= f;
  }

  for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
    test_cases.push_back({normals[i],
                          base_pts[i],
                          {Vec3r::Zero(), aff_dim},
                          {Vec3r::Zero(), span_dim}});
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 6: Positive margin.
  margin = Real(0.5);
  auto set_m = Cylinder(hlx, radius, margin);

  EXPECT_FALSE(set_m.IsPolytopic());

  for (auto& tc : test_cases) {
    tc.base_pt += tc.normal * margin;
    tc.ncs_true.span_dim = 1;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Ellipsoid test
TEST(EllipsoidTest, LocalGeometry) {
  Real hlx = Real(3.0), hly = Real(2.0), hlz = Real(1.5), margin = Real(0.1);
  auto set = Ellipsoid(hlx, hly, hlz, margin);

  EXPECT_FALSE(set.IsPolytopic());

  // Generate test cases.
  std::vector<LocalGeometryTestCase<3>> test_cases(6);

  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs;
  NormalPair<3> zn;

  test_cases[0].normal = Vec3r::UnitX();
  test_cases[1].normal = -Vec3r::UnitY();
  test_cases[2].normal = -Vec3r::UnitZ();
  test_cases[3].normal = (Vec3r::UnitX() + Vec3r::UnitY()).normalized();
  test_cases[4].normal = (-Vec3r::UnitX() + Vec3r::UnitY()).normalized();
  test_cases[5].normal = (-Vec3r::UnitY() - Vec3r::UnitZ()).normalized();

  for (auto& tc : test_cases) {
    tc.base_pt = SupportPointFunction(&set, tc.normal);
    tc.sph_true.aff_dim = 0;
    tc.ncs_true.span_dim = 1;
  }

  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Frustum test
TEST(FrustumTest, LocalGeometry) {
  // Frustum types:
  //  t = 0: rb > 0, rt > 0,
  //  t = 1: rb > 0, rt = 0,
  //  t = 2: rb = 0, rt > 0.
  for (int t = 0; t < 3; ++t) {
    Real rb = (t == 2) ? Real(0.0) : Real(1.0);
    Real rt = (t == 1) ? Real(0.0) : Real(1.5);
    Real height = Real(2.0), margin = Real(0.0);
    Real tha = (rb - rt) / height;
    auto set = Frustum(rb, rt, height, margin);

    EXPECT_FALSE(set.IsPolytopic());

    // Generate test cases.
    std::vector<LocalGeometryTestCase<3>> test_cases;

    std::vector<Vec3r> normals, base_pts;
    std::vector<Real> angs;
    const Vec3r ez = Vec3r::UnitZ();
    int aff_dim, span_dim;

    SupportPatchHull<3> sph;
    NormalConeSpan<3> ncs;
    NormalPair<3> zn;

    angs.clear();
    angs = {Real(0.0), Real(0.5) * kPi, kPi, Real(1.5) * kPi};

    // Case 1: Support patch - edge; normal cone - 2D cone.
    aff_dim = 1;
    span_dim = 2;

    normals.clear();
    for (const auto ang : angs) {
      normals.push_back(Vec3r(std::cos(ang), std::sin(ang), tha).normalized());
    }

    base_pts.clear();
    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      Vec3r sp = (((i % 2 == 0) ? height : Real(0.0)) - set.offset()) * ez;
      sp.head<2>() +=
          ((i % 2 == 0) ? rt : rb) * normals[i].head<2>().normalized();
      base_pts.push_back(sp);
    }

    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      test_cases.push_back(
          {normals[i],
           base_pts[i],
           {DoubleCross(normals[i], ez, normals[i]), aff_dim},
           {DoubleCross(base_pts[i], ez, normals[i]), span_dim}});
      if (((t == 1) && (i % 2 == 0)) || ((t == 2) && (i % 2 == 1))) {
        test_cases.back().ncs_true.span_dim = 3;
      }
    }

    // Case 2: Support patch - edge; normal cone - ray.
    span_dim = 1;

    base_pts.clear();
    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      const Real f = (i + Real(0.5)) / static_cast<Real>(angs.size());
      Vec3r sp = (-set.offset() + (Real(1.0) - f) * height) * ez;
      const Real r = rt + f * (rb - rt);
      sp.head<2>() += f * r * normals[i].head<2>().normalized();
      base_pts.push_back(sp);
    }

    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      test_cases.push_back({normals[i],
                            base_pts[i],
                            {DoubleCross(normals[i], ez, normals[i]), aff_dim},
                            {Vec3r::Zero(), span_dim}});
    }

    // Case 3: Support patch - rim point; normal cone - 2D cone.
    aff_dim = 0;
    span_dim = 2;

    normals.clear();
    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      Real f = (i + Real(0.5)) / static_cast<Real>(angs.size());
      Real sign = Real(2.0) * (((i + 1) % 2) - Real(0.5));
      Vec3r n_edge =
          Vec3r(std::cos(angs[i]), std::sin(angs[i]), tha).normalized();
      normals.push_back((n_edge + sign * f * ez).normalized());
    }

    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      Vec3r sp = SupportPointFunction(&set, normals[i]);
      test_cases.push_back({normals[i],
                            sp,
                            {Vec3r::Zero(), aff_dim},
                            {DoubleCross(sp, ez, normals[i]), span_dim}});
      if (((t == 1) && (i % 2 == 0)) || ((t == 2) && (i % 2 == 1))) {
        test_cases.back().ncs_true.span_dim = 3;
      }
    }

    // Case 4: Support patch - disk; normal cone - 2D cone.
    aff_dim = 2;

    base_pts.clear();
    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      Vec3r sp = (((i % 2 == 0) ? height : Real(0.0)) - set.offset()) * ez;
      sp.head<2>() += ((i % 2 == 0) ? rt : rb) *
                      Vec2r(std::cos(angs[i]), std::sin(angs[i]));
      base_pts.push_back(sp);
    }

    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      Real sign = Real(2.0) * (((i + 1) % 2) - Real(0.5));
      test_cases.push_back({sign * ez,
                            base_pts[i],
                            {Vec3r::Zero(), aff_dim},
                            {DoubleCross(base_pts[i], ez, ez), span_dim}});
      if (((t == 1) && (i % 2 == 0)) || ((t == 2) && (i % 2 == 1))) {
        test_cases.back().sph_true.aff_dim = 0;
        test_cases.back().ncs_true.span_dim = 3;
      }
    }

    // Case 5: Support patch - disk; normal cone - ray.
    span_dim = 1;

    for (int i = 0; i < static_cast<int>(base_pts.size()); ++i) {
      const Real f = (i + Real(0.5)) / static_cast<Real>(base_pts.size());
      base_pts[i].head<2>() *= f;
    }

    for (int i = 0; i < static_cast<int>(angs.size()); ++i) {
      Real sign = Real(2.0) * (((i + 1) % 2) - Real(0.5));
      test_cases.push_back({sign * ez,
                            base_pts[i],
                            {Vec3r::Zero(), aff_dim},
                            {Vec3r::Zero(), span_dim}});
      if (((t == 1) && (i % 2 == 0)) || ((t == 2) && (i % 2 == 1))) {
        test_cases.back().sph_true.aff_dim = 0;
        test_cases.back().ncs_true.span_dim = 3;
      }
    }

    // Run test cases.
    for (const auto& tc : test_cases) {
      zn.n = tc.normal;
      zn.z = tc.base_pt;

      set.ComputeLocalGeometry(zn, sph, ncs);

      EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn,
                   kTolBasis);
      EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
    }

    // Case 6: Positive margin.
    margin = Real(0.5);
    auto set_m = Frustum(rb, rt, height, margin);

    EXPECT_FALSE(set_m.IsPolytopic());

    for (auto& tc : test_cases) {
      tc.base_pt += tc.normal * margin;
      tc.ncs_true.span_dim = 1;
    }

    // Run test cases.
    for (const auto& tc : test_cases) {
      zn.n = tc.normal;
      zn.z = tc.base_pt;

      set_m.ComputeLocalGeometry(zn, sph, ncs);

      EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn,
                   kTolBasis);
      EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
    }
  }
}

//  Mesh test
TEST(MeshTest, LocalGeometry) {
  auto data = CreatePolytopeTestData();

  ASSERT_EQ(data.vertices.size(), 12);
  ASSERT_EQ(data.graph[0], 12);
  ASSERT_EQ(data.graph[1], 20);
  ASSERT_EQ(data.graph.size(), 1 + 1 + 12 + 12 + 3 * 20);

  Real margin = Real(0.0);
  Mesh set(data.vertices, data.graph, data.inradius, margin);

  EXPECT_TRUE(set.IsPolytopic());

  NormalPair<3> zn;
  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs, ncs_true;
  BasePointHint<3> hint;

  // Case 1: Test cases with hints.
  for (size_t i = 0; i < data.test_cases.size(); ++i) {
    const auto& tc = data.test_cases[i];
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    hint.s = &data.s[i];
    hint.bc = &data.bc[i];
    hint.sfh = &data.sfh[i];

    set.ComputeLocalGeometry(zn, sph, ncs, &hint);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 2: Test cases without hints.
  ncs_true.span_dim = 3;
  for (const auto& tc : data.test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, ncs_true, zn, kTolBasis);
  }

  // Case 3: Positive margin, with hints.
  margin = Real(0.3);
  Mesh set_m(std::move(data.vertices), std::move(data.graph), data.inradius,
             margin);

  EXPECT_FALSE(set_m.IsPolytopic());

  for (auto& tc : data.test_cases) {
    tc.base_pt += tc.normal * margin;
    tc.ncs_true.span_dim = 1;
  }

  for (int i = 0; i < static_cast<int>(data.test_cases.size()); ++i) {
    const auto& tc = data.test_cases[i];

    zn.n = tc.normal;
    zn.z = tc.base_pt;

    hint.s = &data.s[i];
    hint.bc = &data.bc[i];
    hint.sfh = &data.sfh[i];

    set_m.ComputeLocalGeometry(zn, sph, ncs, &hint);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 4: Positive margin, without hints.
  for (const auto& tc : data.test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Polytope test
TEST(PolytopeTest, LocalGeometry) {
  auto data = CreatePolytopeTestData();

  Real margin = Real(0.0);
  Polytope set(data.vertices, data.inradius, margin);

  EXPECT_TRUE(set.IsPolytopic());

  NormalPair<3> zn;
  SupportPatchHull<3> sph;
  NormalConeSpan<3> ncs, ncs_true;
  BasePointHint<3> hint;

  // Case 1: Test cases with hints.
  for (size_t i = 0; i < data.test_cases.size(); ++i) {
    const auto& tc = data.test_cases[i];
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    hint.s = &data.s[i];
    hint.bc = &data.bc[i];
    hint.sfh = &data.sfh[i];

    set.ComputeLocalGeometry(zn, sph, ncs, &hint);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 2: Test cases without hints.
  ncs_true.span_dim = 3;
  for (const auto& tc : data.test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, ncs_true, zn, kTolBasis);
  }

  // Case 3: Positive margin, with hints.
  margin = Real(0.3);
  Polytope set_m(std::move(data.vertices), data.inradius, margin);

  EXPECT_FALSE(set_m.IsPolytopic());

  for (auto& tc : data.test_cases) {
    tc.base_pt += tc.normal * margin;
    tc.ncs_true.span_dim = 1;
  }

  for (int i = 0; i < static_cast<int>(data.test_cases.size()); ++i) {
    const auto& tc = data.test_cases[i];

    zn.n = tc.normal;
    zn.z = tc.base_pt;

    hint.s = &data.s[i];
    hint.bc = &data.bc[i];
    hint.sfh = &data.sfh[i];

    set_m.ComputeLocalGeometry(zn, sph, ncs, &hint);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 4: Positive margin, without hints.
  for (const auto& tc : data.test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<3>, sph, tc.sph_true, zn, kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<3>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

// XD convex set tests
struct SetNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    if constexpr (std::is_same_v<T, Stadium>) return "Stadium";
    if constexpr (std::is_same_v<T, Capsule>) return "Capsule";
    if constexpr (std::is_same_v<T, Circle>) return "Circle";
    if constexpr (std::is_same_v<T, Sphere>) return "Sphere";
    return "Unknown";
  }
};

template <class C>
class CapsuleLocalGeometryTest : public testing::Test {
 protected:
  CapsuleLocalGeometryTest() {}
  ~CapsuleLocalGeometryTest() {}
};

using CapsuleTypes = testing::Types<Stadium, Capsule>;
TYPED_TEST_SUITE(CapsuleLocalGeometryTest, CapsuleTypes, SetNameGenerator);

template <class C>
class SphereLocalGeometryTest : public testing::Test {
 protected:
  SphereLocalGeometryTest() {}
  ~SphereLocalGeometryTest() {}
};

using SphereTypes = testing::Types<Circle, Sphere>;
TYPED_TEST_SUITE(SphereLocalGeometryTest, SphereTypes, SetNameGenerator);

//  Capsule test
TYPED_TEST(CapsuleLocalGeometryTest, LocalGeometry) {
  constexpr int dim = TypeParam::dimension();

  Vecr<dim> e2 = Vecr<dim>::UnitY();
  if constexpr (dim == 3) e2 = Vecr<dim>::UnitZ();

  // Generate test cases.
  std::vector<LocalGeometryTestCase<dim>> test_cases;

  std::vector<Vecr<dim>> normals, base_pts;
  int aff_dim, span_dim;

  SupportPatchHull<dim> sph;
  NormalConeSpan<dim> ncs;
  NormalPair<dim> zn;

  Real hlx = Real(2.0), radius = Real(2.0), margin = Real(0.5);
  auto set = TypeParam(hlx, radius, margin);

  EXPECT_FALSE(set.IsPolytopic());

  normals.clear();
  normals = {Vecr<dim>::UnitX(),
             (-Vecr<dim>::UnitX() + Vecr<dim>::UnitY()).normalized(),
             -Vecr<dim>::UnitY(),
             Vecr<dim>::UnitY(),
             -e2,
             e2};

  base_pts.clear();
  base_pts = {
      (hlx + radius + margin) * Vecr<dim>::UnitX(),
      SupportPointFunction(&set, normals[1]),
      -(radius + margin) * Vecr<dim>::UnitY() + hlx * Vecr<dim>::UnitX(),
      (radius + margin) * Vecr<dim>::UnitY() - hlx * Vecr<dim>::UnitX(),
      -(radius + margin) * e2,
      (radius + margin) * e2 + Real(0.99) * hlx * Vecr<dim>::UnitX()};

  // Case 1: Support patch - point, normal cone - ray.
  aff_dim = 0;
  span_dim = 1;

  for (int i = 0; i < 2; ++i) {
    test_cases.push_back({normals[i], base_pts[i], {}, {}});
    test_cases[i].sph_true.aff_dim = aff_dim;
    test_cases[i].ncs_true.span_dim = span_dim;
  }

  // Case 2: Support patch - edge, normal cone - ray.
  aff_dim = 1;

  for (int i = 2; i < static_cast<int>(normals.size()); ++i) {
    test_cases.push_back({normals[i], base_pts[i], {}, {}});
    if constexpr (dim == 3) test_cases[i].sph_true.basis = Vecr<dim>::UnitX();
    test_cases[i].sph_true.aff_dim = aff_dim;
    test_cases[i].ncs_true.span_dim = span_dim;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<dim>, sph, tc.sph_true, zn,
                 kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<dim>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 2: Zero margin.
  radius = Real(0.0);
  margin = Real(0.0);
  auto set_0m = TypeParam(hlx, radius, margin);

  EXPECT_TRUE(set_0m.IsPolytopic());

  base_pts.clear();
  base_pts = {hlx * Vecr<dim>::UnitX(), -hlx * Vecr<dim>::UnitX(),
              hlx * Vecr<dim>::UnitX(), -hlx * Vecr<dim>::UnitX(),
              Vecr<dim>::Zero(),        Real(0.99) * hlx * Vecr<dim>::UnitX()};

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases[i].base_pt = base_pts[i];
    test_cases[i].ncs_true.span_dim = (i < 4) ? dim : dim - 1;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_0m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<dim>, sph, tc.sph_true, zn,
                 kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<dim>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

//  Sphere test
TYPED_TEST(SphereLocalGeometryTest, LocalGeometry) {
  constexpr int dim = TypeParam::dimension();

  Vecr<dim> e2 = Vecr<dim>::UnitY();
  if constexpr (dim == 3) e2 = Vecr<dim>::UnitZ();

  // Generate test cases.
  std::vector<LocalGeometryTestCase<dim>> test_cases;

  std::vector<Vecr<dim>> normals;
  int aff_dim, span_dim;

  SupportPatchHull<dim> sph;
  NormalConeSpan<dim> ncs;
  NormalPair<dim> zn;

  normals.clear();
  normals = {
      Vecr<dim>::UnitX(),
      -Vecr<dim>::UnitY(),
      e2,
      (-Vecr<dim>::UnitX() + Vecr<dim>::UnitY() + e2).normalized(),
  };

  // Case 1: Non-polytopic sphere.
  Real radius = Real(1.5);
  auto set = TypeParam(radius);

  EXPECT_FALSE(set.IsPolytopic());

  aff_dim = 0;
  span_dim = 1;

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases.push_back({normals[i], radius * normals[i], {}, {}});
    test_cases[i].sph_true.aff_dim = aff_dim;
    test_cases[i].ncs_true.span_dim = span_dim;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<dim>, sph, tc.sph_true, zn,
                 kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<dim>, ncs, tc.ncs_true, zn, kTolBasis);
  }

  // Case 2: Zero radius.
  radius = Real(0.0);
  auto set_0m = TypeParam(radius);

  EXPECT_TRUE(set_0m.IsPolytopic());

  span_dim = dim;

  for (int i = 0; i < static_cast<int>(normals.size()); ++i) {
    test_cases[i].base_pt = Vecr<dim>::Zero();
    test_cases[i].ncs_true.span_dim = span_dim;
  }

  // Run test cases.
  for (const auto& tc : test_cases) {
    zn.n = tc.normal;
    zn.z = tc.base_pt;

    set_0m.ComputeLocalGeometry(zn, sph, ncs);

    EXPECT_PRED4(AssertSupportPatchHullEQ<dim>, sph, tc.sph_true, zn,
                 kTolBasis);
    EXPECT_PRED4(AssertNormalConeSpanEQ<dim>, ncs, tc.ncs_true, zn, kTolBasis);
  }
}

}  // namespace
