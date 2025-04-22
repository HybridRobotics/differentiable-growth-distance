#include <gtest/gtest.h>

#include <cmath>
#include <typeinfo>

#include "dgd/data_types.h"
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
#include "dgd/geometry/3d/cone.h"
#include "dgd/geometry/3d/mesh.h"
#include "dgd/graham_scan.h"
// clang-format off
#include "dgd/growth_distance.h"
// clang-format on
#include "dgd/mesh_loader.h"

using namespace dgd;

const Real kTol{kEpsSqrt};

TEST(GrowthDistanceTest, EllipsePolygon) {
  const int nsamples_cold{100};
  const int nsamples_warm{100};

  // Set 1: ellipse.
  const Real hlx{3.0}, hly{2.0}, margin1{0.25};
  auto set1{Ellipse(hlx, hly, margin1)};

  // Set 2: polygon.
  SetDefaultSeed();
  const int npts{100};
  const Real margin2{0.0}, len{2.0};
  std::vector<Vec2f> pts, vert;
  Vec2f vec;
  const int pnorm{6};
  for (int i = 0; i < npts; ++i) {
    vec = Vec2f(Random(1.0), Random(1.0));
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  GrahamScan(pts, vert);
  Real inradius{PolygonInradius(vert, Vec2f::Zero())};
  auto set2{Polygon(vert, margin2, inradius)};

  // Compute growth distance for random transformations.
  Transform2f tf1, tf2;
  SolverSettings settings;
  SolverOutput<2> out;
  const Real dt{0.1};

  for (int i = 0; i < nsamples_cold; ++i) {
    RandomRigidBodyTransform<2>(-2.0, 2.0, tf1);
    RandomRigidBodyTransform<2>(-2.0, 2.0, tf2);
    const Vec2f v(Random(1.0), Random(1.0));
    const Real w{Random(kPi)};
    const Rot2f dR{Eigen::AngleAxis<Real>(w * dt, Vec3f::UnitZ())
                       .matrix()
                       .topLeftCorner<2, 2>()};
    for (int j = 0; j < nsamples_warm; ++j) {
      GrowthDistance(&set1, tf1, &set2, tf2, settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::kOptimal ||
                  out.status == SolutionStatus::kCoincidentCenters);
      const SolutionError err{GetSolutionError(&set1, tf1, &set2, tf2, out)};
      EXPECT_NEAR(err.prim_feas_err, 0.0, kTol);
      EXPECT_NEAR(err.prim_dual_gap, 0.0, kTol);

      tf1.topLeftCorner<2, 2>() *= dR;
      tf1.topRightCorner<2, 1>() += v * dt;
    }
  }
}

TEST(GrowthDistanceTest, ConeMesh) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  const int nsamples_cold{100};
  const int nsamples_warm{100};

  // Set 1: cone.
  const Real ha{kPi / 6.0}, radius{1.0}, margin1{0.25};
  const Real height{radius / std::tan(ha)};
  auto set1{Cone(radius, height, margin1)};

  // Set 2: mesh.
  SetDefaultSeed();
  const int npts{500};
  const Real margin2{0.0}, len{2.0};

  MeshLoader ml{};
  std::vector<Vec3f> pts, vert;
  std::vector<int> graph;
  Vec3f vec;
  const int pnorm{4};
  for (int i = 0; i < npts; ++i) {
    vec = Vec3f(Random(1.0), Random(1.0), Random(1.0));
    vec = vec * len / (vec.lpNorm<pnorm>() + kEps);
    pts.push_back(vec);
  }
  ml.ProcessPoints(pts);
  bool valid{ml.MakeVertexGraph(vert, graph)};
  ASSERT_TRUE(valid);
  const Real inradius{ml.Inradius(vec)};
  for (auto& v : vert) v -= vec;
  auto set2{Mesh(vert, graph, margin2, inradius)};

  // Compute growth distance for random transformations.
  Transform3f tf1, tf2;
  SolverSettings settings;
  SolverOutput<3> out;
  const Real dt{0.1};

  for (int i = 0; i < nsamples_cold; ++i) {
    RandomRigidBodyTransform<3>(-3.0, 3.0, tf1);
    RandomRigidBodyTransform<3>(-3.0, 3.0, tf2);
    const Vec3f v(Random(1.0), Random(1.0), Random(1.0));
    const Vec3f euler(Random(kPi), Random(kPi), Random(kPi));
    Rot3f dR;
    EulerToRotation(dt * euler, dR);
    for (int j = 0; j < nsamples_warm; ++j) {
      GrowthDistance(&set1, tf1, &set2, tf2, settings, out, (j > 0));
      ASSERT_TRUE(out.status == SolutionStatus::kOptimal ||
                  out.status == SolutionStatus::kCoincidentCenters);
      const SolutionError err{GetSolutionError(&set1, tf1, &set2, tf2, out)};
      EXPECT_NEAR(err.prim_feas_err, 0.0, kTol);
      EXPECT_NEAR(err.prim_dual_gap, 0.0, kTol);

      tf1.topLeftCorner<3, 3>() *= dR;
      tf1.topRightCorner<3, 1>() += v * dt;
    }
  }
}
