#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dgd/data_types.h"
// clang-format off
#include "dgd/graham_scan.h"
// clang-format on
#include "dgd/utils.h"

namespace {

using namespace dgd;

const Real kTol{kSqrtEps};

TEST(GrahamScanTest, ZeroDim) {
  std::vector<Vec2r> pts, vert;
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 1);
  EXPECT_EQ(vert[0], Vec2r(1.0, 1.0));

  // Duplicate points.
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 1);
  EXPECT_EQ(vert[0], Vec2r(1.0, 1.0));
}

TEST(GrahamScanTest, OneDim) {
  std::vector<Vec2r> pts, vert;
  pts.push_back({2.0, 3.0});
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2r(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2r(2.0, 3.0));

  // Duplicate points.
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2r(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2r(2.0, 3.0));

  // Collinear points.
  pts.push_back({3.0, 5.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2r(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2r(3.0, 5.0));
}

TEST(GrahamScanTest, TwoDim) {
  // CW-sorted points.
  std::vector<Vec2r> pts, vert;
  pts.push_back({3.0, 2.0});
  pts.push_back({1.0, 0.5});
  pts.push_back({0.0, 0.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2r(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2r(1.0, 0.5));
  EXPECT_EQ(vert[2], Vec2r(3.0, 2.0));

  // CCW-sorted points.
  pts[1] = Vec2r(2.0, 2.0);
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2r(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2r(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2r(2.0, 2.0));

  // Unsorted points with duplicates and p0-collinearity.
  pts.push_back({1.5, 1.0});
  pts.push_back({3.0, 2.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2r(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2r(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2r(2.0, 2.0));

  // Unsorted points with duplicates and p0- and non p0-collinearity.
  // Note: For hill-climbing to work correctly, there must not be any
  // collinear points.
  pts.push_back({2.5, 2.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2r(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2r(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2r(2.0, 2.0));

  // Unsorted points with duplicates, p0- and non p0-collinearity, and right
  // turns.
  pts.push_back({0.0, 4.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2r(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2r(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2r(0.0, 4.0));

  // Check inradius.
  pts.push_back({6.0, 4.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2r(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2r(6.0, 4.0));
  EXPECT_EQ(vert[2], Vec2r(0.0, 4.0));

  Real inradius{ComputePolygonInradius(vert, Vec2r(3.0, 3.0))};
  EXPECT_NEAR(inradius, std::sqrt(9.0 / 13.0), kTol);
}

TEST(GrahamScanTest, CcwOrientation) {
  SetDefaultSeed();
  const int nruns{100};
  const int npts{1000};
  const Real side_len{10.0};

  auto ccw = [](const Vec2r& u, const Vec2r& v, const Vec2r& w) -> Real {
    return (v - u).cross(w - u);
  };

  std::vector<Vec2r> pts(npts), vert;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j)
      pts[j] = Vec2r(Random(side_len), Random(side_len));
    const int nvert{GrahamScan(pts, vert)};

    // Orientation test.
    if (nvert > 2) {
      for (int j = 0; j < nvert - 2; ++j)
        ASSERT_GT(ccw(vert[j], vert[j + 1], vert[j + 2]), 0.0);
      ASSERT_GT(ccw(vert.end()[-2], vert.end()[-1], vert[0]), 0.0);
      ASSERT_GT(ccw(vert.end()[-1], vert[0], vert[1]), 0.0);
    }
  }
}

// Support functions of a set and its convex hull are the same.
TEST(GrahamScanTest, SupportFunction) {
  SetDefaultSeed();
  const int nruns{100};
  const int ndir{100};
  const int npts{1000};
  const Real side_len{10.0};

  auto support = [](const std::vector<Vec2r>& p, const Vec2r& n,
                    Vec2r& sp) -> bool {
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

  std::vector<Vec2r> pts(npts), vert;
  Vec2r sp, spt, dir;
  for (int i = 0; i < nruns; ++i) {
    for (int j = 0; j < npts; ++j)
      pts[j] = Vec2r(Random(side_len), Random(side_len));
    GrahamScan(pts, vert);

    // Support function test.
    for (int k = 0; k < ndir; ++k) {
      Real ang{Real(2 * k) * kPi / ndir};
      dir = Vec2r(std::cos(ang), std::sin(ang));
      bool multiple{support(pts, dir, spt)};
      if (multiple) continue;
      support(vert, dir, sp);
      EXPECT_NEAR(dir.dot(spt), dir.dot(sp), kTol);
      ASSERT_NEAR((spt - sp).lpNorm<Eigen::Infinity>(), 0.0, kTol);
    }
  }
}

}  // namespace
