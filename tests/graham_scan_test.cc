#include "dgd/graham_scan.h"

#include <gtest/gtest.h>

#include "dgd/data_types.h"

namespace {

using namespace dgd;

TEST(GrahamScanTest, ZeroDim) {
  std::vector<Vec2f> pts, vert;
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 1);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));

  // Duplicate points.
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 1);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
}

TEST(GrahamScanTest, OneDim) {
  std::vector<Vec2f> pts, vert;
  pts.push_back({2.0, 3.0});
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2f(2.0, 3.0));

  // Duplicate points.
  pts.push_back({1.0, 1.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2f(2.0, 3.0));

  // Collinear points.
  pts.push_back({3.0, 5.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 2);
  EXPECT_EQ(vert[0], Vec2f(1.0, 1.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 5.0));
}

TEST(GrahamScanTest, TwoDim) {
  // CW-sorted points.
  std::vector<Vec2f> pts, vert;
  pts.push_back({3.0, 2.0});
  pts.push_back({1.0, 0.5});
  pts.push_back({0.0, 0.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(1.0, 0.5));
  EXPECT_EQ(vert[2], Vec2f(3.0, 2.0));

  // CCW-sorted points.
  pts[1] = Vec2f(2.0, 2.0);
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(2.0, 2.0));

  // Unsorted points with duplicates and p0-collinearity.
  pts.push_back({1.5, 1.0});
  pts.push_back({3.0, 2.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(2.0, 2.0));

  // Unsorted points with duplicates, p0- and non p0-collinearity, and right
  // turns.
  pts.push_back({2.5, 2.0});
  pts.push_back({0.0, 4.0});
  GrahamScan(pts, vert);
  ASSERT_EQ(vert.size(), 3);
  EXPECT_EQ(vert[0], Vec2f(0.0, 0.0));
  EXPECT_EQ(vert[1], Vec2f(3.0, 2.0));
  EXPECT_EQ(vert[2], Vec2f(0.0, 4.0));
}

}  // namespace
