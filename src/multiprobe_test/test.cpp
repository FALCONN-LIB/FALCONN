#include <falconn/experimental/pipes.h>

#include <iostream>
#include <random>

using falconn::experimental::HashProducer;
using ir::Point;

const int32_t d = 100;
const int32_t log_num_parts = 8;

int32_t main() {
    HashProducer<Point> hp(1, d, log_num_parts, 1);
    Point p(d);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::normal_distribution<float> g(0.0, 1.0);
    for (size_t i = 0; i < d; ++i) {
        p[i] = g(gen);
    }
    p /= p.norm();
    hp.load_query(0, p);
    auto it = hp.run(0);
    while (it.is_valid()) {
        auto x = it.get();
        std::cout << x.first << " ";
        ++it;
    }
    std::cout << std::endl;
    return 0;
}
