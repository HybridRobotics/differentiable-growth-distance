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
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

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

  // copy vertex data
  ProcessVertices(attrib.vertices);
}

// make graph describing convex hull
bool MeshLoader::MakeGraph(std::vector<Vec3f>& vert, std::vector<int>& graph) {
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
  if (nvert() < 4) {
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
    double* data = this->vert_.data();
    qh_init_B(qh, data, nvert(), 3, False);

    // construct convex hull
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);

    // allocate graph:
    //  numvert, numface, vert_edgeadr[numvert],
    //  edge_localid[numvert+3*numface]
    int numvert = qh->num_vertices;
    int numface = qh->num_facets;
    int szgraph = 2 + 2 * numvert + 3 * numface;
    vert.reserve(numvert);
    graph.reserve(szgraph);
    graph[0] = numvert;
    graph[1] = numface;

    std::vector<int> vert_globalid(numvert);

    // indices for convenience
    int vert_edgeadr = 2;
    int edge_localid = 2 + numvert;

    // fill in graph data
    int i = adr = 0;
    ok = 1;
    FORALLvertices {
      // point id of this vertex, check
      int pid = qh_pointid(qh, vertex->point);
      if (pid < 0 || pid >= nvert()) {
        ok = 0;
        break;
      }

      // save edge address and global id of this vertex
      graph[vert_edgeadr + i] = adr;
      vert_globalid[i] = pid;
      vert[i] = Vec3f(vert_[3 * pid], vert_[3 * pid + 1], vert_[3 * pid + 2]);

      // process neighboring faces and their vertices
      int start = adr;
      FOREACHsetelement_(facetT, vertex->neighbors, facet) {
        int cnt = 0;
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          ++cnt;

          // point id of face vertex, check
          int pid1 = qh_pointid(qh, vertex1->point);
          if (pid1 < 0 || pid1 >= nvert()) {
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
    if (adr != numvert + 3 * numface) {
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
    for (int i = 0; i < numvert + 3 * numface; ++i) {
      if (graph[edge_localid + i] >= 0) {
        // search vert_globalid for match
        int adr;
        for (adr = 0; adr < numvert; ++adr) {
          if (vert_globalid[adr] == graph[edge_localid + i]) {
            graph[edge_localid + i] = adr;
            break;
          }
        }

        // make sure we found a match: SHOULD NOT OCCUR
        if (adr >= numvert) {
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

}  // namespace dgd
