#include <falconn/experimental/code_generation.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <vector>

using falconn::core::RandomProjectionSketches;
using falconn::experimental::DeduplicationPipe;
using falconn::experimental::DistanceScorer;
using falconn::experimental::ExhaustiveProducer;
using falconn::experimental::TopKPipe;

using falconn::DenseVector;

// hash -> table -> deduplication -> topk w/sketches -> topk w/distance
class Pipeline1 {
 public:
  Pipeline1(
      int32_t num_workers, std::vector<falconn::DenseVector<float>>& dataset,
      const std::map<std::string, std::string>& deserialization_filenames = {})
      : producer_(num_workers, 1, 16, 10, -1, 2, 4057218),
        num_workers_(num_workers),
        step_1_(num_workers, dataset, producer_, 2,
                deserialization_filenames.find("step_1") !=
                        deserialization_filenames.end()
                    ? deserialization_filenames.find("step_1")->second
                    : ""),
        step_2_(num_workers, dataset.size()),
        step_3_(num_workers, 20, false, 1),
        step_4_(num_workers, 5, false, 1),
        scorer_step_3_(num_workers, dataset, 2, 4057218),
        scorer_step_4_(num_workers, dataset) {}

  auto execute_query(int32_t worker_id,
                     const falconn::DenseVector<float>& query) {
    if (worker_id < 0 || worker_id >= num_workers_) {
      throw falconn::experimental::PipelineError(
          "The worker id should be between 0 and num_workers - 1");
    }
    // load query
    producer_.load_query(worker_id, query);
    scorer_step_3_.load_query(worker_id, query);
    scorer_step_4_.load_query(worker_id, query);

    // run pipe
    auto it0 = producer_.run(worker_id);
    auto it1 = step_1_.run(worker_id, it0);
    auto it2 = step_2_.run(worker_id, it1);
    auto it3 = step_3_.run(worker_id, it2, scorer_step_3_);
    auto it4 = step_4_.run(worker_id, it3, scorer_step_4_);
    return it4;
  }
  // getters
  falconn::experimental::TablePipe<falconn::DenseVector<float>>* get_step_1() {
    return &step_1_;
  }
  falconn::experimental::DeduplicationPipe<falconn::DenseVector<float>>*
  get_step_2() {
    return &step_2_;
  }
  falconn::experimental::TopKPipe<
      falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>>*
  get_step_3() {
    return &step_3_;
  }
  falconn::experimental::TopKPipe<
      falconn::experimental::DistanceScorer<falconn::DenseVector<float>>>*
  get_step_4() {
    return &step_4_;
  }
  falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>*
  get_scorer_step_3() {
    return &scorer_step_3_;
  }
  falconn::experimental::DistanceScorer<falconn::DenseVector<float>>*
  get_scorer_step_4() {
    return &scorer_step_4_;
  }
  falconn::experimental::HashProducer<falconn::DenseVector<float>>*
  get_producer() {
    return &producer_;
  }

 private:
  falconn::experimental::HashProducer<falconn::DenseVector<float>> producer_;
  int32_t num_workers_;
  falconn::experimental::TablePipe<falconn::DenseVector<float>> step_1_;
  falconn::experimental::DeduplicationPipe<falconn::DenseVector<float>> step_2_;
  falconn::experimental::TopKPipe<
      falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>>
      step_3_;
  falconn::experimental::TopKPipe<
      falconn::experimental::DistanceScorer<falconn::DenseVector<float>>>
      step_4_;
  falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>
      scorer_step_3_;
  falconn::experimental::DistanceScorer<falconn::DenseVector<float>>
      scorer_step_4_;
};

// exhaustive -> topk w/sketches -> topk w/distance
class Pipeline2 {
 public:
  Pipeline2(int32_t num_workers,
            std::vector<falconn::DenseVector<float>>& dataset)
      : producer_(num_workers, dataset.size()),
        num_workers_(num_workers),
        step_1_(num_workers, 1024, false, 1),
        step_2_(num_workers, 5, false, 1),
        scorer_step_1_(num_workers, dataset, 2, 41231238),
        scorer_step_2_(num_workers, dataset) {}

  auto execute_query(int32_t worker_id,
                     const falconn::DenseVector<float>& query) {
    if (worker_id < 0 || worker_id >= num_workers_) {
      throw falconn::experimental::PipelineError(
          "The worker id should be between 0 and num_workers - 1");
    }
    // load query
    scorer_step_1_.load_query(worker_id, query);
    scorer_step_2_.load_query(worker_id, query);

    // run pipe
    auto it0 = producer_.run(worker_id);
    auto it1 = step_1_.run(worker_id, it0, scorer_step_1_);
    auto it2 = step_2_.run(worker_id, it1, scorer_step_2_);
    return it2;
  }
  // getters
  falconn::experimental::TopKPipe<
      falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>>*
  get_step_1() {
    return &step_1_;
  }
  falconn::experimental::TopKPipe<
      falconn::experimental::DistanceScorer<falconn::DenseVector<float>>>*
  get_step_2() {
    return &step_2_;
  }
  falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>*
  get_scorer_step_1() {
    return &scorer_step_1_;
  }
  falconn::experimental::DistanceScorer<falconn::DenseVector<float>>*
  get_scorer_step_2() {
    return &scorer_step_2_;
  }
  falconn::experimental::ExhaustiveProducer* get_producer() {
    return &producer_;
  }

 private:
  falconn::experimental::ExhaustiveProducer producer_;
  int32_t num_workers_;
  falconn::experimental::TopKPipe<
      falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>>
      step_1_;
  falconn::experimental::TopKPipe<
      falconn::experimental::DistanceScorer<falconn::DenseVector<float>>>
      step_2_;
  falconn::core::RandomProjectionSketches<falconn::DenseVector<float>>
      scorer_step_1_;
  falconn::experimental::DistanceScorer<falconn::DenseVector<float>>
      scorer_step_2_;
};

std::vector<DenseVector<float>> get_dummy_dataset(int32_t n) {
  std::vector<DenseVector<float>> dataset;
  for (int i = 0; i < n; ++i) {
    DenseVector<float> p(1);
    p[0] = i;
    dataset.push_back(p);
  }
  return dataset;
}

TEST(Pipeline1Test, TestRunQuerySimple) {
  const int32_t n = 3000;
  auto dataset = get_dummy_dataset(n);

  Pipeline1 pipe(1, dataset);
  auto it = pipe.execute_query(0, dataset[0]);
  int32_t total = 0;
  std::vector<int32_t> ans;
  while (it.is_valid()) {
    ans.push_back(it.get());
    ++it;
    total++;
  }
  ASSERT_EQ(total, 5);
  ASSERT_THAT(ans, ::testing::Contains(0));
}

TEST(Pipeline1Test, TestRunQueryWorkers) {
  const int32_t n = 3000;
  const int32_t num_workers = 4;
  auto dataset = get_dummy_dataset(n);

  Pipeline1 pipe(num_workers, dataset);
  for (int32_t worker_id = 0; worker_id < num_workers; worker_id++) {
    auto it = pipe.execute_query(worker_id, dataset[0]);
    int32_t total = 0;
    std::vector<int32_t> ans;
    while (it.is_valid()) {
      ans.push_back(it.get());
      ++it;
      total++;
    }
    ASSERT_EQ(total, 5);
    ASSERT_THAT(ans, ::testing::Contains(0));
  }
  ASSERT_THROW(pipe.execute_query(num_workers, dataset[0]),
               falconn::experimental::PipelineError);
}

TEST(Pipeline2Test, TestRunQuerySimple) {
  const int32_t n = 3000;
  auto dataset = get_dummy_dataset(n);

  Pipeline2 pipe(1, dataset);
  auto it = pipe.execute_query(0, dataset[0]);
  int32_t total = 0;
  std::vector<int32_t> ans;
  while (it.is_valid()) {
    ans.push_back(it.get());
    ++it;
    total++;
  }
  ASSERT_EQ(total, 5);
  ASSERT_THAT(ans, ::testing::Contains(0));
}

TEST(Pipeline2Test, TestRunQueryWorkers) {
  const int32_t n = 3000;
  const int32_t num_workers = 4;
  auto dataset = get_dummy_dataset(n);

  Pipeline2 pipe(num_workers, dataset);
  for (int32_t worker_id = 0; worker_id < num_workers; worker_id++) {
    auto it = pipe.execute_query(worker_id, dataset[0]);
    int32_t total = 0;
    std::vector<int32_t> ans;
    while (it.is_valid()) {
      ans.push_back(it.get());
      ++it;
      total++;
    }
    ASSERT_EQ(total, 5);
    ASSERT_THAT(ans, ::testing::Contains(0));
  }
  ASSERT_THROW(pipe.execute_query(num_workers, dataset[0]),
               falconn::experimental::PipelineError);
}

TEST(JsonTest, TestCorrectFormat1) {
  const std::string json_string = R"(
  {
    "producer": {
        "type": "HashProducer",
        "dimension": 128,
        "num_hash_bits": 16,
        "num_tables": 10,
        "num_probes": 15,
        "num_rotations": 2,
        "seed": 998123
    },
    "step_1": {
        "type": "TablePipe",
        "num_setup_threads": 0
    },
    "step_2": {
        "type": "DeduplicationPipe"
    },
    "step_3": {
        "type": "TopKPipe",
        "k": 100,
        "scorer": {
            "type": "RandomProjectionSketches",
            "num_chunks": 2,
            "seed": 123123
        },
        "sort": true,
        "look_ahead": 2
    },
    "step_4": {
        "type": "TopKPipe",
        "k": 100,
        "scorer": {
            "type": "DistanceScorer"
        },
        "sort": true,
        "look_ahead": 2
    }
  })";
  std::istringstream string_stream(json_string);
  std::string code =
      falconn::experimental::generate_pipeline_from_json<ir::Point>(
          string_stream);
  ASSERT_TRUE(code.size() > 0);
}

TEST(JsonTest, TestCorrectFormat2) {
  const std::string json_string = R"(
  {
    "producer": {
        "type": "ExhaustiveProducer"
    },
    "step_1": {
        "type": "TopKPipe",
        "k": 100,
        "scorer": {
            "type": "RandomProjectionSketches",
            "num_chunks": 2,
            "seed": 123123
        },
        "sort": true,
        "look_ahead": 2
    }
  })";
  std::istringstream string_stream(json_string);
  std::string code =
      falconn::experimental::generate_pipeline_from_json<ir::Point>(
          string_stream);
  ASSERT_TRUE(code.size() > 0);
}

TEST(JsonTest, TestIncorrectFormat1) {
  // typo in type
  const std::string json_string = R"(
  {
    "producer": {
        "type": "ExhaustiveProducer"
    },
    "step_1": {
        "type": "TopKPipe",
        "k": 100,
        "scorer": {
            "type": "RandomTypo",
            "num_chunks": 2,
            "seed": 123123
        },
        "sort": true,
        "look_ahead": 2
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat2) {
  // no scorer
  const std::string json_string = R"(
  {
    "producer": {
        "type": "ExhaustiveProducer"
    },
    "step_1": {
        "type": "TopKPipe",
        "k": 100,
        "sort": true,
        "look_ahead": 2
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat3) {
  // extra comma
  const std::string json_string = R"(
  {
    "producer": {
        "type": "HashProducer",
        "dimension": 128,
        "num_hash_bits": 16,
        "num_tables": 10,
        "num_probes": 15,
        "num_rotations": 2,
        "seed": 998123
    },
    "step_1": {
        "type": "TablePipe",
        "num_setup_threads": 0
    },
    "step_2": {
        "type": "DeduplicationPipe"
    },
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat4) {
  // missing comma
  const std::string json_string = R"(
  {
    "producer": {
        "type": "HashProducer",
        "dimension": 128,
        "num_hash_bits": 16
        "num_tables": 10,
        "num_probes": 15,
        "num_rotations": 2,
        "seed": 998123
    },
    "step_1": {
        "type": "TablePipe",
        "num_setup_threads": 0
    },
    "step_2": {
        "type": "DeduplicationPipe"
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat5) {
  // only producer
  const std::string json_string = R"(
  {
    "producer": {
        "type": "HashProducer",
        "dimension": 128,
        "num_hash_bits": 16,
        "num_tables": 10,
        "num_probes": 15,
        "num_rotations": 2,
        "seed": 998123
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat6) {
  // no producer
  const std::string json_string = R"(
  {
    "step_1": {
        "type": "TablePipe",
        "num_setup_threads": 0
    },
    "step_2": {
        "type": "DeduplicationPipe"
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat7) {
  // missing step number
  const std::string json_string = R"(
  {
    "producer": {
        "type": "HashProducer",
        "dimension": 128,
        "num_hash_bits": 16,
        "num_tables": 10,
        "num_probes": 15,
        "num_rotations": 2,
        "seed": 998123
    },
    "step_1": {
        "type": "TablePipe",
        "num_setup_threads": 0
    },
    "step_2": {
        "type": "DeduplicationPipe"
    },
    "step_4": {
        "type": "TopKPipe",
        "k": 100,
        "scorer": {
            "type": "DistanceScorer"
        },
        "sort": true,
        "look_ahead": 2
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}

TEST(JsonTest, TestIncorrectFormat8) {
  // incorrect step name
  const std::string json_string = R"(
  {
    "producer": {
        "type": "HashProducer",
        "dimension": 128,
        "num_hash_bits": 16,
        "num_tables": 10,
        "num_probes": 15,
        "num_rotations": 2,
        "seed": 998123
    },
    "step_1": {
        "type": "TablePipe",
        "num_setup_threads": 0
    },
    "step_2": {
        "type": "DeduplicationPipe"
    },
    "step_x": {
        "type": "TopKPipe",
        "k": 100,
        "scorer": {
            "type": "DistanceScorer"
        },
        "sort": true,
        "look_ahead": 2
    }
  })";
  std::istringstream string_stream(json_string);
  ASSERT_THROW(falconn::experimental::generate_pipeline_from_json<ir::Point>(
                   string_stream),
               falconn::experimental::PipelineGenerationError);
}