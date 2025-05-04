// Copyright 2021 DeepMind Technologies Limited
// Copyright 2025 Akshay Thirugnanam
//
// Copied and adapted from MuJoCo
// (https://github.com/google-deepmind/mujoco/blob/main/src/user/user_mesh.cc)
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
 * @file mesh_loader.cc
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-04-01
 * @brief 3D mesh loader class implementation.
 */

#include "dgd/mesh_loader.h"

#include <cmath>
#include <csetjmp>

#define TINYOBJLOADER_IMPLEMENTATION  // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust triangulation.
// Requires C++11
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tinyobjloader/tiny_obj_loader.h"

extern "C" {
#include "libqhull_r/qhull_ra.h"
}

namespace dgd {

// load OBJ mesh
void MeshLoader::LoadOBJ(const std::string& input, bool is_file) {
  tinyobj::ObjReader objReader;

  if (is_file) {
    objReader.ParseFromFile(input);
  } else {
    objReader.ParseFromString(input, std::string());
  }

  if (!objReader.Valid()) {
    std::string err = "could not parse OBJ file";
    if (is_file) err += std::string(" ") + input;
    throw std::runtime_error(err);
  }

  const auto& attrib = objReader.GetAttrib();
  normal_ = attrib.normals;
  facenormal_.clear();

  if (!objReader.GetShapes().empty()) {
    const auto& mesh = objReader.GetShapes()[0].mesh;

    // iterate over mesh faces
    std::vector<tinyobj::index_t> face_indices;
    for (int face = 0, idx = 0; idx < static_cast<int>(mesh.indices.size());) {
      int nfacevert = mesh.num_face_vertices[face];
      if (nfacevert < 3 || nfacevert > 4) {
        throw std::runtime_error(
            "only tri or quad meshes are supported for OBJ");
      }

      face_indices.push_back(mesh.indices[idx]);
      face_indices.push_back(mesh.indices[idx + 1]);
      face_indices.push_back(mesh.indices[idx + 2]);

      if (nfacevert == 4) {
        face_indices.push_back(mesh.indices[idx]);
        face_indices.push_back(mesh.indices[idx + 2]);
        face_indices.push_back(mesh.indices[idx + 3]);
      }
      idx += nfacevert;
      ++face;
    }

    // for each vertex, store index and normal
    for (const auto& mesh_index : face_indices) {
      face_.push_back(mesh_index.vertex_index);

      if (!normal_.empty()) {
        facenormal_.push_back(mesh_index.normal_index);
      }
    }
  }

  // copy vertex (point) data
  ProcessPoints(attrib.vertices);
}

// make vertex graph describing convex hull
bool MeshLoader::MakeVertexGraph(std::vector<Vec3f>& vert,
                                 std::vector<int>& graph) {
  vert.clear();
  graph.clear();

  int adr, ok, curlong, totlong, exitcode;
  facetT *facet, **facetp;
  vertexT *vertex, *vertex1, **vertex1p;

  std::string qhopt = "qhull Qt";
  if (maxhullvert_ > -1) {
    // qhull "TA" actually means "number of vertices added after the initial
    // simplex"
    qhopt += " TA" + std::to_string(maxhullvert_ - 4);
  }

  // graph not needed for small meshes
  if (npts() < 4) {
    return false;
  }

  qhT qh_qh;
  qhT* qh = &qh_qh;
  qh_zero(qh, stderr);

  // qhull basic init
  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);

  // install longjmp error handler
  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = false;
  if (!exitcode) {
    // actual init
    qh_initflags(qh, const_cast<char*>(qhopt.c_str()));
    double* data = this->pts_.data();
    qh_init_B(qh, data, npts(), 3, False);

    // construct convex hull
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);

    // allocate graph:
    //  nvert, nface, vert_edgeadr[nvert],
    //  edge_localid[nvert + 3*nface]
    int nvert = qh->num_vertices;
    int nface = qh->num_facets;
    int szgraph = 2 + 2 * nvert + 3 * nface;
    vert.resize(nvert);
    graph.resize(szgraph);
    graph[0] = nvert;
    graph[1] = nface;

    std::vector<int> vert_globalid(nvert);

    // indices for convenience
    int vert_edgeadr = 2;
    int edge_localid = 2 + nvert;

    // fill in graph data
    int i = adr = 0;
    ok = 1;
    FORALLvertices {
      // point id of this vertex, check
      int pid = qh_pointid(qh, vertex->point);
      if (pid < 0 || pid >= npts()) {
        ok = 0;
        break;
      }

      // save edge address and global id of this vertex
      graph[vert_edgeadr + i] = adr;
      vert_globalid[i] = pid;
      vert[i] = Vec3f(Real(pts_[3 * pid]), Real(pts_[3 * pid + 1]),
                      Real(pts_[3 * pid + 2]));

      // process neighboring faces and their vertices
      int start = adr;
      FOREACHsetelement_(facetT, vertex->neighbors, facet) {
        int cnt = 0;
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          ++cnt;

          // point id of face vertex, check
          int pid1 = qh_pointid(qh, vertex1->point);
          if (pid1 < 0 || pid1 >= npts()) {
            ok = 0;
            break;
          }

          // if different from vertex id, try to insert
          if (pid != pid1) {
            // check for previous record
            int j;
            for (j = start; j < adr; ++j)
              if (pid1 == graph[edge_localid + j]) {
                break;
              }

            // not found: insert
            if (j >= adr) {
              graph[edge_localid + adr] = pid1;
              ++adr;
            }
          }
        }

        // make sure we have triangle: SHOULD NOT OCCUR
        if (cnt != 3) {
          throw std::runtime_error("Qhull did not return triangle");
        }
      }

      // insert separator, advance to next vertex
      graph[edge_localid + adr] = -1;
      ++adr;
      ++i;
    }

    // size check: SHOULD NOT OCCUR
    if (adr != nvert + 3 * nface) {
      throw std::runtime_error("Wrong size in convex hull graph");
    }

    // free all
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);

    // bad graph: delete
    if (!ok) {
      vert.clear();
      graph.clear();
      return false;
    }

    // replace global ids with local ids in edge data
    for (int i = 0; i < nvert + 3 * nface; ++i) {
      if (graph[edge_localid + i] >= 0) {
        // search vert_globalid for match
        int adr;
        for (adr = 0; adr < nvert; ++adr) {
          if (vert_globalid[adr] == graph[edge_localid + i]) {
            graph[edge_localid + i] = adr;
            break;
          }
        }

        // make sure we found a match: SHOULD NOT OCCUR
        if (adr >= nvert) {
          throw std::runtime_error("Vertex id not found in convex hull");
        }
      }
    }
  }

  // longjmp error handler
  else {
    // free all
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    vert.clear();
    graph.clear();

    throw std::runtime_error("qhull error");
  }

  return true;
}

// make facet graph describing convex hull
bool MeshLoader::MakeFacetGraph(std::vector<Vec3f>& normal,
                                std::vector<Real>& offset,
                                std::vector<int>& graph,
                                Vec3f& interior_point) {
  normal.clear();
  offset.clear();
  graph.clear();

  int adr, ok, curlong, totlong, exitcode;
  facetT *facet, *facet1, **facet1p;

  std::string qhopt = "qhull Qt";
  if (maxhullvert_ > -1) {
    // qhull "TA" actually means "number of vertices added after the initial
    // simplex"
    qhopt += " TA" + std::to_string(maxhullvert_ - 4);
  }

  // graph not needed for small meshes
  if (npts() < 4) {
    return false;
  }

  // linear algebra functions
  auto eqn = [](facetT* facet, double* pt) -> double {
    return (facet->normal[0] * pt[0] + facet->normal[1] * pt[1] +
            facet->normal[2] * pt[2] + facet->offset);
  };

  qhT qh_qh;
  qhT* qh = &qh_qh;
  qh_zero(qh, stderr);

  // qhull basic init
  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);

  // install longjmp error handler
  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = false;
  if (!exitcode) {
    // actual init
    qh_initflags(qh, const_cast<char*>(qhopt.c_str()));
    double* data = this->pts_.data();
    qh_init_B(qh, data, npts(), 3, False);

    // construct convex hull
    qh_qhull(qh);

    // allocate graph:
    //  nfacet, nridge, facet_ridgeadr[nfacet],
    //  ridge_localid[nfacet + 2*nridge]
    int nvert = qh->num_vertices;
    int nfacet = qh->num_facets;
    int nridge = nfacet + nvert - 2;
    int szgraph = 2 + 2 * nfacet + 2 * nridge;
    normal.resize(nfacet);
    offset.resize(nfacet);
    graph.resize(szgraph);
    graph[0] = nfacet;
    graph[1] = nridge;
    for (int i = 0; i < 3; ++i) interior_point(i) = Real(qh->interior_point[i]);

    std::vector<int> facet_globalid(nfacet);

    // indices for convenience
    int facet_ridgeadr = 2;
    int ridge_localid = 2 + nfacet;

    // fill in graph data
    Real norm;
    int i = adr = 0;
    ok = 1;
    FORALLfacets {
      facet_globalid[i] = facet->id;

      // check constraint at facet
      if (eqn(facet, SETfirstt_(facet->vertices, vertexT)->point) > kEps) {
        throw std::runtime_error("Incorrect normal and offset at facet");
      }

      // set normal and offset
      Real sign = eqn(facet, qh->interior_point) <= 0 ? 1.0 : -1.0;
      for (int k = 0; k < 3; ++k) {
        normal[i](k) = sign * Real(facet->normal[k]);
      }
      offset[i] = sign * Real(facet->offset);
      norm = normal[i].norm();
      if (norm < kEps) {
        std::runtime_error("Zero normal vector at facet");
      }
      normal[i] = normal[i] / norm;
      offset[i] = offset[i] / norm;

      // save ridge address of this facet
      graph[facet_ridgeadr + i] = adr;

      // set facet neighbours
      int cnt = 0;
      FOREACHsetelement_(facetT, facet->neighbors, facet1) {
        // insert facet id
        graph[ridge_localid + adr] = facet1->id;
        ++cnt;
        ++adr;
      }
      if (cnt < 3) {
        throw std::runtime_error("Facet with less than three neighbours");
      }

      // insert separator, advance to next facet
      graph[ridge_localid + adr] = -1;
      ++adr;
      ++i;
    }

    // size check: SHOULD NOT OCCUR
    if (i != nfacet || adr != nfacet + 2 * nridge) {
      throw std::runtime_error("Wrong size in convex hull graph");
    }

    // replace global ids with local ids in facet data
    for (int i = 0; i < nfacet + 2 * nridge; ++i) {
      if (graph[ridge_localid + i] >= 0) {
        // search facet_globalid for match
        int adr;
        for (adr = 0; adr < nfacet; ++adr) {
          if (facet_globalid[adr] == graph[ridge_localid + i]) {
            graph[ridge_localid + i] = adr;
            break;
          }
        }

        // make sure we found a match: SHOULD NOT OCCUR
        if (adr >= nfacet) {
          throw std::runtime_error("Facet id not found in convex hull");
        }
      }
    }

    // reorder facet neighbours according to CCW orientation
    int nn;
    Vec3f n, n1, t1, t2;
    for (int i = 0; i < nfacet; ++i) {
      n = normal[i];
      adr = ridge_localid + graph[facet_ridgeadr + i];

      // number of neighbours
      if (i < nfacet - 1) {
        nn = graph[facet_ridgeadr + i + 1] - graph[facet_ridgeadr + i] - 1;
      } else {
        nn = nfacet + 2 * nridge - graph[facet_ridgeadr + i] - 1;
      }

      // check for adjacent parallel facets
      for (int j = 0; j < nn; ++j) {
        n1 = normal[graph[adr + j]];
        if (std::abs(n1.dot(n)) > Real(1.0) - kEps) {
          throw std::runtime_error("Adjacent facets are (anti)parallel");
        }
      }

      // compute tangent vectors at facet
      t1 = n.cross(normal[graph[adr]]).normalized();
      t2 = n.cross(t1);

      // sort neighbours
      auto comp = [&normal, &t1, &t2](int j, int k) -> bool {
        double aj = std::atan2(normal[j].dot(t2), normal[j].dot(t1));
        double ak = std::atan2(normal[k].dot(t2), normal[k].dot(t1));
        return aj < ak;
      };

      std::sort(graph.begin() + adr, graph.begin() + adr + nn, comp);
    }

    // free all
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);

    // bad graph: delete
    if (!ok) {
      normal.clear();
      offset.clear();
      graph.clear();
      return false;
    }
  }

  // longjmp error handler
  else {
    // free all
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    normal.clear();
    offset.clear();
    graph.clear();

    throw std::runtime_error("qhull error");
  }

  return true;
}

Real MeshLoader::ComputeInradius(const std::vector<Vec3f>& normal,
                                 const std::vector<Real>& offset,
                                 const Vec3f& interior_point) const {
  Real max{-kInf}, eqn;
  for (int i = 0; i < static_cast<int>(normal.size()); ++i) {
    eqn = normal[i].dot(interior_point) + offset[i];
    if (eqn >= 0.0)
      throw std::runtime_error("Point is not in the polytope interior");
    max = std::max(max, eqn);
  }

  return -max;
}

Real MeshLoader::ComputeInradius(Vec3f& interior_point, bool use_given_ip) {
  std::vector<Vec3f> normal;
  std::vector<Real> offset;
  std::vector<int> graph;
  Vec3f ip_;
  bool valid{MakeFacetGraph(normal, offset, graph, ip_)};
  if (!valid) throw std::runtime_error("Qhull error");

  if (!use_given_ip) interior_point = ip_;
  return ComputeInradius(normal, offset, interior_point);
}

}  // namespace dgd
