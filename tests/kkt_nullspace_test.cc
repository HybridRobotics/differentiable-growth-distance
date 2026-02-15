#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "dgd/data_types.h"
#include "dgd/dgd.h"
#include "dgd/geometry/geometry_2d.h"
#include "dgd/geometry/geometry_3d.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/utils/transformations.h"

namespace {

using namespace dgd;

// Checks orthonormality of a basis.
template <int row, int col>
bool IsOrthonormalBasis(const Matr<row, col>& basis, int dim, Real tol) {
  for (int i = 0; i < dim; ++i) {
    if (std::abs(basis.col(i).norm() - Real(1.0)) > tol) return false;
    for (int j = i + 1; j < dim; ++j) {
      if (std::abs(basis.col(i).dot(basis.col(j))) > tol) return false;
    }
  }
  return true;
}

// Assertion functions
const Real kTol = kSqrtEps;

template <int dim>
bool AssertPrimalBasis(const Matr<dim, dim - 1>& basis, const Vecr<dim>& normal,
                       int nullity, Real tol = kTol) {
  Matr<dim, dim> basis_full;
  basis_full.col(0) = normal.normalized();
  basis_full.template rightCols<dim - 1>() = basis;
  if (!IsOrthonormalBasis(basis_full, nullity + 1, tol)) return false;
  return true;
}

template <int dim>
bool AssertDualBasis(const Matr<dim, dim>& basis, const Vecr<dim>& normal,
                     int nullity, Real tol = kTol) {
  if ((basis.col(0) - normal.normalized()).norm() > tol) return false;
  if (!IsOrthonormalBasis(basis, nullity, tol)) return false;
  return true;
}

// KKT nullspace tests
// 2D convex set tests
//  Vertex-vertex test
TEST(KktNullspaceTest, VertexVertex2D) {
  Rectangle rect1(0.5, 0.5);
  Rectangle rect2(0.5, 0.5);

  Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();
  Affine(tf2) = Vec2r(2.0, 2.0);  // Diagonal separation

  Settings settings;
  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&rect1, tf1, &rect2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&rect1, tf1, &rect2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 2);
  EXPECT_EQ(kkt_nullity, 2);

  EXPECT_PRED4(AssertPrimalBasis<2>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<2>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Edge-edge test
TEST(KktNullspaceTest, EdgeEdge2D) {
  Rectangle rect1(1.0, 0.5);
  Rectangle rect2(1.0, 0.5);

  Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();
  Affine(tf2) = Vec2r(0.0, 2.0);  // Vertical separation

  Settings settings;
  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&rect1, tf1, &rect2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&rect1, tf1, &rect2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 1);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 2);

  EXPECT_PRED4(AssertPrimalBasis<2>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<2>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Circle-circle test
TEST(KktNullspaceTest, CircleCircle2D) {
  Ellipse circ1(1.0, 1.0);
  Ellipse circ2(1.0, 1.0);

  Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();
  Affine(tf2) = Vec2r(3.0, 4.0);

  Settings settings;
  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&circ1, tf1, &circ2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&circ1, tf1, &circ2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 1);

  EXPECT_PRED4(AssertPrimalBasis<2>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<2>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Vertex-halfspace test
TEST(KktNullspaceTest, VertexHalfspace2D) {
  Ellipse circ(1.0, 1.0);
  Halfspace<2> hs;

  Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();
  Affine(tf1) = Vec2r(0.0, 2.0);

  Settings settings;
  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&circ, tf1, &hs, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity = ComputeKktNullspace(&circ, tf1, &hs, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 1);

  EXPECT_PRED4(AssertPrimalBasis<2>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<2>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Edge-halfspace test
TEST(KktNullspaceTest, EdgeHalfspace2D) {
  Rectangle rect(1.0, 0.5);
  Halfspace<2> hs;

  Transform2r tf1 = Transform2r::Identity();
  Transform2r tf2 = Transform2r::Identity();
  Affine(tf1) = Vec2r(0.0, 1.5);

  Settings settings;
  Output<2> out;
  DirectionalDerivative<2> dd;
  TotalDerivative<2> td;
  OutputBundle<2> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&rect, tf1, &hs, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity = ComputeKktNullspace(&rect, tf1, &hs, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 1);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 2);

  EXPECT_PRED4(AssertPrimalBasis<2>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<2>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

// 3D convex set tests
//  Vertex-vertex test
TEST(KktNullspaceTest, VertexVertex3D) {
  Cuboid cube1(0.5, 0.5, 0.5);
  Cuboid cube2(0.5, 0.5, 0.5);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = Vec3r(2.0, 2.0, 2.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cube1, tf1, &cube2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 3);
  EXPECT_EQ(kkt_nullity, 3);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Face-face test
TEST(KktNullspaceTest, FaceFace3D) {
  Cuboid cube1(1.0, 1.0, 1.0);
  Cuboid cube2(1.0, 0.25, 0.25);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = Vec3r(3.0, 0.0, 0.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cube1, tf1, &cube2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 2);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 3);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Face-edge test
TEST(KktNullspaceTest, FaceEdge3D) {
  Cuboid cube(1.0, 1.0, 1.0);
  Cylinder cyl(0.5, 2.0);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = Vec3r(0.0, 2.0, 0.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cube, tf1, &cyl, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&cube, tf1, &cyl, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 1);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 2);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Edge-edge (parallel) test
TEST(KktNullspaceTest, EdgeEdgeParallel3D) {
  Cuboid cube1(1.0, 0.5, 0.5);
  Cuboid cube2(1.0, 0.5, 0.5);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Linear(tf2) = EulerToRotation(kPi / Real(2.0) * Vec3r::UnitZ());
  Affine(tf2) = Vec3r(0.0, 1.5, 1.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cube1, tf1, &cube2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);
  EXPECT_GE(dd.z_nullity, 1);
  EXPECT_EQ(dd.n_nullity, 2);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Edge-edge (non-parallel) test
TEST(KktNullspaceTest, EdgeEdgeNonParallel3D) {
  Cuboid cube1(1.0, 0.5, 0.5);
  Cuboid cube2(0.5, 1.0, 0.5);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Linear(tf1) = EulerToRotation(kPi / 4.0 * Vec3r::UnitX());
  Linear(tf2) = EulerToRotation(kPi / 4.0 * Vec3r::UnitY());
  Affine(tf2) = Vec3r(Real(-0.2), Real(0.1), 1.5);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cube1, tf1, &cube2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 1);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Sphere-sphere test
TEST(KktNullspaceTest, SphereSphere3D) {
  Sphere sph1(1.0);
  Sphere sph2(1.0);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = Vec3r(3.0, 4.0, 5.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&sph1, tf1, &sph2, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&sph1, tf1, &sph2, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 1);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

//  Vertex-Face test
TEST(KktNullspaceTest, VertexFace3D) {
  Sphere sph(1.0);
  Cuboid cube(1.0, 1.0, 1.0);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = Vec3r(3.0, 0.0, 0.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&sph, tf1, &cube, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity =
      ComputeKktNullspace(&sph, tf1, &cube, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 1);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

// Vertex-halfspace test
TEST(KktNullspaceTest, VertexHalfspace3D) {
  Sphere sph(1.0);
  Halfspace<3> hs;

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf1) = Vec3r(0.0, 0.0, 2.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&sph, tf1, &hs, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity = ComputeKktNullspace(&sph, tf1, &hs, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 1);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

// Edge-halfspace test
TEST(KktNullspaceTest, EdgeHalfspaceTest) {
  Cylinder cyl(1.0, 2.0);
  Halfspace<3> hs;

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf1) = Vec3r(0.0, 0.0, 2.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cyl, tf1, &hs, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity = ComputeKktNullspace(&cyl, tf1, &hs, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 1);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_LE(kkt_nullity, 2);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

// Face-halfspace test
TEST(KktNullspaceTest, FaceHalfspaceTest) {
  Cuboid cube(1.0, 1.0, 1.0);
  Halfspace<3> hs;

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf1) = Vec3r(0.0, 0.0, 2.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  Real gd = GrowthDistance(&cube, tf1, &hs, tf2, settings, out);
  EXPECT_GT(gd, 0.0);
  EXPECT_EQ(out.status, SolutionStatus::Optimal);

  int kkt_nullity = ComputeKktNullspace(&cube, tf1, &hs, tf2, settings, bundle);
  EXPECT_EQ(dd.z_nullity, 2);
  EXPECT_EQ(dd.n_nullity, 1);
  EXPECT_EQ(kkt_nullity, 3);

  EXPECT_PRED4(AssertPrimalBasis<3>, dd.z_nullspace, out.normal, dd.z_nullity,
               kTol);
  EXPECT_PRED4(AssertDualBasis<3>, dd.n_nullspace, out.normal, dd.n_nullity,
               kTol);
}

// Non-optimal status tests
TEST(KktNullspaceTest, NonOptimalStatus3D) {
  Cuboid cube1(1.0, 1.0, 1.0);
  Cuboid cube2(1.0, 1.0, 1.0);

  Transform3r tf1 = Transform3r::Identity();
  Transform3r tf2 = Transform3r::Identity();
  Affine(tf2) = Vec3r(3.0, 0.0, 0.0);

  Settings settings;
  Output<3> out;
  DirectionalDerivative<3> dd;
  TotalDerivative<3> td;
  OutputBundle<3> bundle{&out, &dd, &td};

  GrowthDistance(&cube1, tf1, &cube2, tf2, settings, out);

  out.status = SolutionStatus::CoincidentCenters;
  int kkt_nullity =
      ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);

  EXPECT_EQ(dd.z_nullity, 0);
  EXPECT_EQ(dd.n_nullity, 0);
  EXPECT_EQ(kkt_nullity, 0);

  out.status = SolutionStatus::IllConditionedInputs;
  kkt_nullity = ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);
  EXPECT_EQ(kkt_nullity, 0);

  out.status = SolutionStatus::MaxIterReached;
  kkt_nullity = ComputeKktNullspace(&cube1, tf1, &cube2, tf2, settings, bundle);
  EXPECT_EQ(kkt_nullity, 0);
}

}  // namespace
