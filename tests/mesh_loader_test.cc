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

const Real kTol{kEpsSqrt};

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
  ml.LoadOBJ(obj, false);

  const int nvert_{8}, nface_{12};
  ASSERT_EQ(ml.npts(), nvert_);

  std::vector<Vec3f> vert;
  std::vector<int> graph;
  bool valid{ml.MakeVertexGraph(vert, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(vert.size(), nvert_);
  ASSERT_EQ(vert.size(), graph[0]);
  ASSERT_EQ(graph.size(), 2 + 2 * nvert_ + 3 * nface_);
}

TEST(MeshLoaderTest, InputFile) {
  std::string file = "../tinyobjloader/models/cube.obj";

  MeshLoader ml{};
  ml.LoadOBJ(file);

  const int nvert_{8}, nface_{12};
  ASSERT_EQ(ml.npts(), nvert_);

  std::vector<Vec3f> vert;
  std::vector<int> graph;
  bool valid{ml.MakeVertexGraph(vert, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(vert.size(), nvert_);
  ASSERT_EQ(vert.size(), graph[0]);
  ASSERT_EQ(graph.size(), 2 + 2 * nvert_ + 3 * nface_);
}

TEST(MeshLoaderTest, MakeVertexGraph) {
  std::vector<Vec3f> pts;
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 1.0});
  pts.push_back({0.0, 1.0, 0.0});
  pts.push_back({0.0, 1.0, 1.0});
  pts.push_back({1.0, 1.0, 1.0});

  MeshLoader ml{};
  ml.ProcessPoints(pts);

  const int nvert_{5}, nface_{6};
  ASSERT_EQ(ml.npts(), nvert_);

  std::vector<Vec3f> vert;
  std::vector<int> graph;
  bool valid{ml.MakeVertexGraph(vert, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(vert.size(), nvert_);
  ASSERT_EQ(vert.size(), graph[0]);
  ASSERT_EQ(graph.size(), 2 + 2 * nvert_ + 3 * nface_);
}

TEST(MeshLoaderTest, MakeFacetGraph) {
  std::vector<Vec3f> pts;
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 0.0});
  pts.push_back({0.0, 0.0, 1.0});
  pts.push_back({0.0, 1.0, 0.0});
  pts.push_back({0.0, 1.0, 1.0});
  pts.push_back({1.0, 1.0, 1.0});

  MeshLoader ml{};
  ml.ProcessPoints(pts);

  const int nvert_{5}, nfacet_{5};
  const int nridge_{nfacet_ + nvert_ - 2};
  ASSERT_EQ(ml.npts(), nvert_);

  std::vector<Vec3f> normal;
  std::vector<Real> offset;
  std::vector<int> graph;
  bool valid{ml.MakeFacetGraph(normal, offset, graph)};

  ASSERT_TRUE(valid);
  ASSERT_EQ(normal.size(), nfacet_);
  ASSERT_EQ(offset.size(), nfacet_);
  ASSERT_EQ(graph[0], nfacet_);
  ASSERT_EQ(graph[1], nridge_);
  ASSERT_EQ(graph.size(), 2 + 2 * nfacet_ + 2 * nridge_);

  Real eqn, max;
  for (int i = 0; i < nfacet_; ++i) {
    max = -kInf;
    for (int j = 0; j < nvert_; ++j) {
      eqn = normal[i].dot(pts[j]) + offset[i];
      EXPECT_LE(eqn, 0.0);
      if (eqn > max) max = eqn;
    }
    ASSERT_NEAR(max, 0.0, kTol);
  }
}

TEST(MeshLoaderTest, SupportFunction) {
  SetDefaultSeed();
  const int nruns{10};
  const int ndir_xy{100}, ndir_z{10};
  const int npts{1000};
  const Real side_len{10.0};

  auto support = [](const std::vector<Vec3f>& p, const Vec3f& n,
                    Vec3f& sp) -> bool {
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
  std::vector<Vec3f> pts(npts), vert;
  std::vector<int> graph;
  Vec3f sp, sp_, dir;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j)
      pts[j] = Vec3f(Random(side_len), Random(side_len), Random(side_len));
    ml.ProcessPoints(pts);
    bool valid{ml.MakeVertexGraph(vert, graph)};

    ASSERT_TRUE(valid);

    // Support function test.
    for (int kxy = 0; kxy < ndir_xy; ++kxy) {
      Real ang_xy{Real(2 * kxy) * kPi / ndir_xy};
      for (int kz = 0; kz < ndir_z; ++kz) {
        Real ang_z{Real(2 * kz) * kPi / ndir_z};
        dir = Vec3f(std::cos(ang_z) * std::cos(ang_xy),
                    std::cos(ang_z) * std::sin(ang_xy), std::sin(ang_z));
        bool multiple{support(pts, dir, sp_)};
        if (multiple) continue;
        support(vert, dir, sp);
        EXPECT_NEAR(dir.dot(sp), dir.dot(sp_), kTol);
        ASSERT_NEAR((sp_ - sp).lpNorm<Eigen::Infinity>(), 0.0, kTol);
      }
    }
  }
}

}  // namespace
