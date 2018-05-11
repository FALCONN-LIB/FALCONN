#include <falconn/experimental/code_generation.h>

#include <serialize.h>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: ./pipe_from_json path_json" << std::endl;
    return 1;
  }
  const std::string filename(argv[1]);
  std::ifstream fin(filename);
  std::cout << falconn::experimental::generate_pipeline_from_json<ir::Point>(
      fin);
  return 0;
}