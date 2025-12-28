#ifndef DGD_BENCHMARKS_INTERNAL_HELPERS_FILESYSTEM_UTILS_H_
#define DGD_BENCHMARKS_INTERNAL_HELPERS_FILESYSTEM_UTILS_H_

#include <filesystem>
#include <iostream>
#include <vector>

namespace dgd {

namespace bench {

// Checks if the given folder path is a valid directory.
inline bool IsValidDirectory(const std::string& path) {
  if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
    std::cerr << "Error: The specified folder path is not a valid directory"
              << std::endl;
    return false;
  }
  return true;
}

// Gets the .obj file names from the given folder path.
inline std::vector<std::string> GetObjFileNames(const std::string& path) {
  std::vector<std::string> filenames;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".obj") {
      filenames.push_back(path + entry.path().filename().string());
    }
  }
  return filenames;
}

}  // namespace bench

}  // namespace dgd

#endif  // DGD_BENCHMARKS_INTERNAL_HELPERS_FILESYSTEM_UTILS_H_
