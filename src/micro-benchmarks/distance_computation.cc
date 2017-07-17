#include <Eigen/Dense>
#include <iostream>

#include <chrono>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cstdlib>

using std::cout;
using std::endl;
using std::mt19937_64;
using std::normal_distribution;
using std::random_device;
using std::runtime_error;
using std::thread;
using std::uniform_int_distribution;
using std::vector;

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

using Eigen::VectorXf;
using Eigen::Map;

void worker(float *dataset, float *query, const vector<int> &candidates, int d,
            int u, int v, float *dummy) {
  *dummy = 0.0;
  for (int i = u; i < v; ++i) {
    *dummy += Map<VectorXf>(dataset + candidates[i] * d, d)
                  .dot(Map<VectorXf>(query, d));
  }
}

int main() {
  const int N = 1200000;
  const int D = 104;
  const int Q = 100000000;
  cout << N << " points" << endl;
  cout << D << " dimensions" << endl;
  cout << "retrieving " << Q << " candidates" << endl;
  int max_num_threads = thread::hardware_concurrency();
  cout << max_num_threads << " threads are supported" << endl;
  float *dataset;
  if (posix_memalign((void **)&dataset, 32, sizeof(float) * N * D)) {
    throw runtime_error("can't allocate memory for dataset");
  }
  random_device rd;
  mt19937_64 gen(rd());
  normal_distribution<float> g(0.0, 1.0);
  for (int i = 0; i < N * D; ++i) {
    dataset[i] = g(gen);
  }
  vector<int> candidates(Q);
  uniform_int_distribution<> u(0, N - 1);
  for (int i = 0; i < Q; ++i) {
    candidates[i] = u(gen);
  }
  for (int num_threads = 1; num_threads <= max_num_threads; ++num_threads) {
    float *queries;
    if (posix_memalign((void **)&queries, 32,
                       sizeof(float) * D * num_threads)) {
      throw runtime_error("can't allocate memory for queries");
    }
    for (int i = 0; i < num_threads * D; ++i) {
      queries[i] = g(gen);
    }
    vector<int> start(num_threads + 1);
    start[0] = 0;
    for (int i = 0; i < num_threads; ++i) {
      int cnt = Q / num_threads;
      int rem = Q % num_threads;
      if (i < rem) {
        ++cnt;
      }
      start[i + 1] = start[i] + cnt;
    }
    vector<float> dummy(num_threads, 0.0);
    vector<thread> threads;
    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
      threads.push_back(thread(worker, dataset, queries + i * D, candidates, D,
                               start[i], start[i + 1], &dummy[i]));
    }
    for (int i = 0; i < num_threads; ++i) {
      threads[i].join();
    }
    auto t2 = high_resolution_clock::now();
    cout << num_threads << " thread" << ((num_threads > 1) ? "s" : "") << ": "
         << duration_cast<duration<double>>(t2 - t1).count() << " seconds"
         << endl;
    free(queries);
  }
  free(dataset);
}
