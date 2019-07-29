#include <falconn/experimental/pipes.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

namespace py = pybind11;

using falconn::experimental::HashProducer;
using ir::Point;

class Multiprobe {
 public:
  Multiprobe(int32_t dimension, int32_t num_bits, uint64_t seed)
      : dimension(dimension) {
    hp = std::make_unique<HashProducer<Point>>(1, dimension, num_bits, 1, -1, 2,
                                               seed);
  }

  std::vector<uint32_t> query(py::array_t<float, py::array::c_style> query) {
    auto buf = query.request();
    if (buf.ndim != 1) {
      std::cerr << "One-dimensional array expected" << std::endl;
      exit(1);
    }
    if (buf.shape[0] != dimension) {
      std::cerr << "Invalid dimension" << std::endl;
      exit(1);
    }
    Eigen::Map<Point> p(static_cast<float*>(buf.ptr), dimension);
    float n = p.norm();
    if (fabs(n - 1.0) > 1e-4) {
      std::cerr << "Not a unit vector: " << n << std::endl;
      exit(1);
    }
    hp->load_query(0, p);
    std::vector<uint32_t> res;
    auto it = hp->run(0);
    while (it.is_valid()) {
      res.push_back(it.get().first);
      ++it;
    }
    return res;
  }

 private:
  int32_t dimension;
  std::unique_ptr<HashProducer<Point>> hp;
};

PYBIND11_MODULE(_multiprobe, m) {
  py::class_<Multiprobe>(m, "Multiprobe")
      .def(py::init<int32_t, int32_t, uint64_t>())
      .def("query", &Multiprobe::query);
}
