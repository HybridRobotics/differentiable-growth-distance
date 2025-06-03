#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "dgd/data_types.h"
// clang-format off
#include "dgd/mesh_loader.h"
// clang-format on
#include "dgd/utils.h"

namespace {

using namespace dgd;

const Real kTol{kSqrtEps};

TEST(MeshLoaderTest, StringParse) {
  std::string obj =
      "v 0.000000 2.000000 2.000000\n"
      "v 0.000000 0.000000 2.000000\n"
      "v 2.000000 0.000000 2.000000\n"
      "v 2.000000 2.000000 2.000000\n"
      "v 0.000000 2.000000 0.000000\n"
      "v 0.000000 0.000000 0.000000\n"
      "v 2.000000 0.000000 0.000000\n"
      "v 2.000000 2.000000 0.000000\n"
      "# 8 vertices\n"
      "\n"
      "g front cube\n"
      "f 1 2 3 4\n"
      "g back cube\n"
      "f 8 7 6 5\n"
      "g right cube\n"
      "f 4 3 7 8\n"
      "g top cube\n"
      "f 5 1 4 8\n"
      "g left cube\n"
      "f 5 6 2 1\n"
      "g bottom cube\n"
      "f 2 6 7 3\n"
      "# 6 elements";

  MeshLoader ml{};
  ml.LoadObj(obj, false);

  const int nvert{8}, nface{12};
  ASSERT_EQ(ml.npts(), nvert);

  std::vector<Vec3r> vert;
  std::vector<int> graph;
  bool valid{ml.MakeVertexGraph(vert, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(vert.size(), nvert);
  ASSERT_EQ(vert.size(), graph[0]);
  ASSERT_EQ(graph.size(), 2 + 2 * nvert + 3 * nface);
}

TEST(MeshLoaderTest, InputFile) {
  GTEST_SKIP();
  std::string file = "../tinyobjloader/models/cube.obj";

  MeshLoader ml{};
  ml.LoadObj(file);

  const int nvert{8}, nface{12};
  ASSERT_EQ(ml.npts(), nvert);

  std::vector<Vec3r> vert;
  std::vector<int> graph;
  bool valid{ml.MakeVertexGraph(vert, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(vert.size(), nvert);
  ASSERT_EQ(vert.size(), graph[0]);
  ASSERT_EQ(graph.size(), 2 + 2 * nvert + 3 * nface);
}

TEST(MeshLoaderTest, MakeVertexGraph) {
  std::vector<Vec3r> pts;
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 1.0});
  pts.push_back({0.0, 1.0, 0.0});
  pts.push_back({0.0, 1.0, 1.0});
  pts.push_back({1.0, 1.0, 1.0});

  MeshLoader ml{};
  ml.ProcessPoints(pts);

  const int nvert{5}, nface{6};
  ASSERT_EQ(ml.npts(), nvert);

  std::vector<Vec3r> vert;
  std::vector<int> graph;
  bool valid{ml.MakeVertexGraph(vert, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(vert.size(), nvert);
  ASSERT_EQ(vert.size(), graph[0]);
  ASSERT_EQ(graph.size(), 2 + 2 * nvert + 3 * nface);
}

TEST(MeshLoaderTest, MakeFacetGraph) {
  std::vector<Vec3r> pts;
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 1.0});
  pts.push_back({0.0, 1.0, 0.0});
  pts.push_back({0.0, 1.0, 1.0});
  pts.push_back({1.0, 1.0, 1.0});

  MeshLoader ml{};
  ml.ProcessPoints(pts);

  const int nvert{5}, nfacet{5};
  const int nridge{nfacet + nvert - 2};
  ASSERT_EQ(ml.npts(), nvert);

  std::vector<Vec3r> normal;
  std::vector<Real> offset;
  std::vector<int> graph;
  Vec3r interior_point;
  bool valid{ml.MakeFacetGraph(normal, offset, graph, interior_point)};
  Real inradius{ml.ComputeInradius(normal, offset, interior_point)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(normal.size(), nfacet);
  ASSERT_EQ(offset.size(), nfacet);
  ASSERT_EQ(graph[0], nfacet);
  ASSERT_EQ(graph[1], nridge);
  ASSERT_EQ(graph.size(), 2 + 2 * nfacet + 2 * nridge);

  Real eqn, eqnr, max, maxr{-kInf};
  for (int i = 0; i < nfacet; ++i) {
    eqnr = normal[i].dot(interior_point) + offset[i];
    EXPECT_LE(eqnr, -inradius);
    maxr = std::max(maxr, eqnr + inradius);

    max = -kInf;
    for (int j = 0; j < nvert; ++j) {
      eqn = normal[i].dot(pts[j]) + offset[i];
      EXPECT_LE(eqn, 0.0);
      max = std::max(max, eqn);
    }
    ASSERT_NEAR(max, 0.0, kTol);
  }
  ASSERT_NEAR(maxr, 0.0, kTol);
}

TEST(MeshLoaderTest, SupportFunction) {
  // Qhull computations can be unstable with float.
  if (typeid(Real) == typeid(float)) GTEST_SKIP();

  SetDefaultSeed();
  const int nruns{10};
  const int ndir_xy{100}, ndir_z{10};
  const int npts{1000};
  const Real side_len{10.0};

  auto support = [](const std::vector<Vec3r>& p, const Vec3r& n,
                    Vec3r& sp) -> bool {
    int idx{0};
    Real s{0.0}, sv{n.dot(p[0])};
    bool multiple{false};
    for (int i = 1; i < static_cast<int>(p.size()); ++i) {
      s = n.dot(p[i]);
      if (s > sv) {
        idx = i;
        sv = s;
        multiple = false;
      } else if (s == sv)
        multiple = true;
    }
    sp = p[idx];
    return multiple;
  };

  MeshLoader ml{};
  std::vector<Vec3r> pts(npts), vert;
  std::vector<int> graph;
  Vec3r sp, spt, dir;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j)
      pts[j] = Vec3r(Random(side_len), Random(side_len), Random(side_len));
    ml.ProcessPoints(pts);
    bool valid{ml.MakeVertexGraph(vert, graph)};

    ASSERT_TRUE(valid);

    // Support function test.
    for (int kxy = 0; kxy < ndir_xy; ++kxy) {
      Real ang_xy{Real(2 * kxy) * kPi / ndir_xy};
      for (int kz = 0; kz < ndir_z; ++kz) {
        Real ang_z{Real(2 * kz) * kPi / ndir_z};
        dir = Vec3r(std::cos(ang_z) * std::cos(ang_xy),
                    std::cos(ang_z) * std::sin(ang_xy), std::sin(ang_z));
        bool multiple{support(pts, dir, spt)};
        if (multiple) continue;
        support(vert, dir, sp);
        EXPECT_NEAR(dir.dot(sp), dir.dot(spt), kTol);
        ASSERT_NEAR((spt - sp).lpNorm<Eigen::Infinity>(), 0.0, kTol);
      }
    }
  }
}

}  // namespace
