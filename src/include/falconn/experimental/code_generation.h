#ifndef __CODE_GENERATION__
#define __CODE_GENERATION__

#define UNUSED(x) (void)(x)

#include "../falconn_global.h"
#include "pipes.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace falconn {
namespace experimental {

class PipelineGenerationError : public FalconnError {
 public:
  PipelineGenerationError(const char *msg) : FalconnError(msg) {}
};

template <typename PointType>
struct PointTypeName;

template <>
struct PointTypeName<DenseVector<float>> {
  static const std::string get_type_name() {
    return "falconn::DenseVector<float>";
  }
};

template <>
struct PointTypeName<DenseVector<double>> {
  static const std::string get_type_name() {
    return "falconn::DenseVector<double>";
  }
};

enum class Producer {
  ExhaustiveProducer,
  HashProducer,
  Unknown,
};

enum class Scorer {
  RandomProjectionSketchesScorer,
  DistanceScorer,
  Unknown,
};

enum class Pipe {
  TablePipe,
  DeduplicationPipe,
  TopKPipe,
  Unknown,
};

struct ProducerParameters {
  ProducerParameters(Producer p) : producer_type_(p) {}
  virtual ~ProducerParameters() {}
  virtual std::string get_parameters() const = 0;

  const Producer producer_type_;
};

struct HashProducerParameters : ProducerParameters {
  HashProducerParameters(int32_t dimension, int32_t num_hash_bits,
                         int32_t num_tables, int32_t num_probes = -1,
                         int32_t num_rotations = 2,
                         uint_fast64_t seed = 4057218)
      : ProducerParameters(Producer::HashProducer),
        dimension_(dimension),
        num_hash_bits_(num_hash_bits),
        num_tables_(num_tables),
        num_probes_(num_probes),
        num_rotations_(num_rotations),
        seed_(seed) {}

  std::string get_parameters() const {
    return std::to_string(dimension_) + ", " + std::to_string(num_hash_bits_) +
           ", " + std::to_string(num_tables_) + ", " +
           std::to_string(num_probes_) + ", " + std::to_string(num_rotations_) +
           ", " + std::to_string(seed_);
  }

  int32_t dimension_;
  int32_t num_hash_bits_;
  int32_t num_tables_;
  int32_t num_probes_;
  int32_t num_rotations_;
  uint_fast64_t seed_;
};

struct ExhaustiveProducerParameters : ProducerParameters {
  ExhaustiveProducerParameters()
      : ProducerParameters(Producer::ExhaustiveProducer) {}

  std::string get_parameters() const { return "dataset.size()"; }
};

struct ScorerParameters {
  ScorerParameters(Scorer s, bool is_serializable)
      : scorer_type_(s), is_serializable_(is_serializable) {}
  virtual ~ScorerParameters() {}
  virtual std::string get_parameters(const std::string &step_name) const = 0;

  const Scorer scorer_type_;
  const bool is_serializable_;
};

struct RandomProjectionSketchesScorerParameters : ScorerParameters {
  RandomProjectionSketchesScorerParameters(int32_t num_chunks,
                                           uint_fast64_t seed)
      : ScorerParameters(Scorer::RandomProjectionSketchesScorer, false),
        num_chunks_(num_chunks),
        seed_(seed) {}

  std::string get_parameters(const std::string &step_name) const {
    UNUSED(step_name);
    return "dataset, " + std::to_string(num_chunks_) + ", " +
           std::to_string(seed_);
  }

  int32_t num_chunks_;
  uint_fast64_t seed_;
};

struct DistanceScorerParameters : ScorerParameters {
  DistanceScorerParameters()
      : ScorerParameters(Scorer::DistanceScorer, false) {}

  std::string get_parameters(const std::string &step_name) const {
    UNUSED(step_name);
    return "dataset";
  }
};

struct PipeParameters {
  PipeParameters(Pipe p, bool is_serializable)
      : pipe_type_(p), is_serializable_(is_serializable) {}
  virtual ~PipeParameters() {}
  virtual std::string get_parameters(const std::string &step_name) const = 0;

  const Pipe pipe_type_;
  const bool is_serializable_;
};

struct TopKPipeParameters : PipeParameters {
  TopKPipeParameters(int32_t k, ScorerParameters *scorer, bool sort = false,
                     int32_t look_ahead = 1)
      : PipeParameters(Pipe::TopKPipe, false),
        k_(k),
        scorer_(scorer),
        sort_(sort),
        look_ahead_(look_ahead) {}

  std::string get_parameters(const std::string &step_name) const {
    UNUSED(step_name);
    return std::to_string(k_) + ", " + (sort_ ? "true" : "false") + ", " +
           std::to_string(look_ahead_);
  }

  int32_t k_;
  ScorerParameters *scorer_;
  bool sort_;
  int32_t look_ahead_;
};

struct TablePipeParameters : PipeParameters {
  TablePipeParameters(int_fast32_t num_setup_threads = 0)
      : PipeParameters(Pipe::TablePipe, true),
        num_setup_threads_(num_setup_threads) {}

  std::string get_parameters(const std::string &pipe_name) const {
    // trim trailing _ from member name
    const std::string lookup_key = pipe_name.substr(0, pipe_name.size() - 1);
    return "dataset, producer_, " + std::to_string(num_setup_threads_) +
           ", "
           "deserialization_filenames.find(\"" +
           lookup_key +
           "\") "
           "!= deserialization_filenames.end() ? "
           "deserialization_filenames.find(\"" +
           lookup_key + "\")->second : \"\"";
  }

  int_fast32_t num_setup_threads_;
};

struct DeduplicationPipeParameters : PipeParameters {
  DeduplicationPipeParameters()
      : PipeParameters(Pipe::DeduplicationPipe, false) {}

  std::string get_parameters(const std::string &step_name) const {
    UNUSED(step_name);
    return "dataset.size()";
  }
};

struct PipeElement {
  std::string type;
  std::string name;
  std::string scorer_name;
};

std::string get_scorer_type_definition(const ScorerParameters *s,
                                       const std::string &template_parameter) {
  std::string class_name;
  switch (s->scorer_type_) {
    case Scorer::RandomProjectionSketchesScorer:
      class_name = "falconn::core::RandomProjectionSketches";
      break;
    case Scorer::DistanceScorer:
      class_name = "falconn::experimental::DistanceScorer";
      break;
    default:
      break;
  }
  return class_name + "<" + template_parameter + ">";
}

std::string get_pipe_type_definition(PipeParameters *p,
                                     const std::string &template_parameter) {
  std::string class_name;
  switch (p->pipe_type_) {
    case Pipe::TablePipe:
      class_name = "falconn::experimental::TablePipe";
      break;
    case Pipe::DeduplicationPipe:
      class_name = "falconn::experimental::DeduplicationPipe";
      break;
    case Pipe::TopKPipe:
      class_name = "falconn::experimental::TopKPipe";
      break;
    default:
      break;
  }
  return class_name + "<" + template_parameter + ">";
}

std::tuple<std::vector<PipeElement>, std::vector<PipeElement>>
gen_types_and_names(const std::string &point_type,
                    const std::vector<PipeParameters *> &pipe_params) {
  std::vector<PipeElement> pipes, scorers;

  for (int32_t i = 0; i < static_cast<int32_t>(pipe_params.size()); ++i) {
    auto p = pipe_params[i];
    std::string pipe_type;
    const std::string pipe_name = "step_" + std::to_string(i + 1) + "_";
    std::string scorer_name;
    if (p->pipe_type_ == Pipe::TopKPipe) {
      auto tk = dynamic_cast<TopKPipeParameters *>(p);
      if (tk == nullptr) {
        throw PipelineGenerationError("The parameter for TopKPipe is invalid.");
      }
      const ScorerParameters *scorer = tk->scorer_;
      scorer_name = "scorer_step_" + std::to_string(i + 1) + "_";
      const std::string scorer_type =
          get_scorer_type_definition(scorer, point_type);
      scorers.push_back({scorer_type, scorer_name, ""});
      pipe_type = get_pipe_type_definition(p, scorer_type);
    } else {
      pipe_type = get_pipe_type_definition(p, point_type);
    }
    pipes.push_back({pipe_type, pipe_name, scorer_name});
  }
  return std::make_tuple(pipes, scorers);
}

std::string gen_load_query(const std::vector<PipeElement> &scorers,
                           const PipeElement &producer) {
  // only hash producer loads the query
  std::string gen = producer.type.find("HashProducer") == std::string::npos
                        ? ""
                        : (producer.name + ".load_query(worker_id, query);\n");
  for (auto s : scorers) {
    gen.append(s.name + ".load_query(worker_id, query);\n");
  }
  return gen;
}

std::string gen_getters(const std::vector<PipeElement> &pipes,
                        const std::vector<PipeElement> &scorers,
                        PipeElement &producer) {
  std::string gen;
  auto format_getter = [](std::string &type, std::string &name) {
    const std::string method_name = name.substr(0, name.size() - 1);
    return type + "* get_" + method_name + "() { return &" + name + "; }\n";
  };
  for (auto p : pipes) {
    gen.append(format_getter(p.type, p.name));
  }
  for (auto s : scorers) {
    gen.append(format_getter(s.type, s.name));
  }
  gen.append(format_getter(producer.type, producer.name));
  return gen;
}

std::string gen_query_steps(const std::vector<PipeElement> &pipes) {
  std::string gen;
  for (int32_t i = 0; i < static_cast<int32_t>(pipes.size()); ++i) {
    auto p = pipes[i];
    const std::string ans_it = "auto it" + std::to_string(i + 1);
    const std::string prev_it = "it" + std::to_string(i);
    const std::string fn_call =
        ".run(worker_id, " + prev_it +
        (p.scorer_name.empty() ? "" : (", " + p.scorer_name)) + ");\n";
    gen.append(ans_it + " = " + p.name + fn_call);
  }
  return gen + "return it" + std::to_string(pipes.size()) + ";";
}

std::string gen_init_list(ProducerParameters *producer,
                          const std::vector<PipeParameters *> &pipe_params,
                          const std::vector<PipeElement> &pipes,
                          const std::vector<PipeElement> &scorers) {
  std::string gen_pipes, gen_scorers;
  for (int32_t i = 0, j = 0; i < static_cast<int32_t>(pipe_params.size());
       ++i) {
    auto p = pipe_params[i];
    std::string pipe_init;

    if (p->pipe_type_ == Pipe::TopKPipe) {
      auto tk = dynamic_cast<TopKPipeParameters *>(p);
      if (tk == nullptr) {
        throw PipelineGenerationError("The parameter for TopKPipe is invalid.");
      }
      const ScorerParameters *scorer = tk->scorer_;
      pipe_init = pipes[i].name + "(num_workers, " +
                  tk->get_parameters(pipes[i].name) + ")";
      const std::string scorer_init = scorers[j].name + "(num_workers, " +
                                      scorer->get_parameters(scorers[j].name) +
                                      ")";
      j++;
      gen_scorers.append(",\n" + scorer_init);
    } else {
      pipe_init = pipes[i].name + "(num_workers, " +
                  p->get_parameters(pipes[i].name) + ")";
    }
    gen_pipes.append(",\n" + pipe_init);
  }
  return "producer_(num_workers, " + producer->get_parameters() +
         "),\n num_workers_(num_workers)" + gen_pipes + gen_scorers;
}

template <typename PointType>
std::string generate(const std::vector<PointType> &dataset,
                     ProducerParameters *producer_params,
                     const std::vector<PipeParameters *> &pipe_params) {
  const std::string base_template =
      R"(
    #include <falconn/experimental/pipes.h>
      
    class Pipeline {
     public:
      Pipeline(int32_t num_workers,
               std::vector<%s>& dataset
               %s)
        : %s {}

      auto execute_query(int32_t worker_id, const %s& query) {
        if (worker_id < 0 || worker_id >= num_workers_) {
          throw falconn::experimental::PipelineError(
          "The worker id should be between 0 and num_workers - 1");
        }
        // load query
        %s
        // run pipe
        auto it0 = producer_.run(worker_id);
        %s
      }
      // getters
      %s
     private:
      %s
    };
  )";
  const std::string point_type = PointTypeName<PointType>::get_type_name();
  // we just use it to obtain the point_type above.
  UNUSED(dataset);

  const std::string producer_type =
      (producer_params->producer_type_ == Producer::HashProducer
           ? "falconn::experimental::HashProducer<" + point_type + ">"
           : "falconn::experimental::ExhaustiveProducer");
  PipeElement producer({producer_type, "producer_", ""});

  std::vector<PipeElement> pipes, scorers;
  std::tie(pipes, scorers) = gen_types_and_names(point_type, pipe_params);
  const std::string load_query = gen_load_query(scorers, producer);
  const std::string query_steps = gen_query_steps(pipes);
  const std::string getters = gen_getters(pipes, scorers, producer);
  const std::string init_list =
      gen_init_list(producer_params, pipe_params, pipes, scorers);
  std::string member_declaration =
      producer.type + " " + producer.name + ";\nint32_t num_workers_;\n";
  for (auto p : pipes) {
    member_declaration.append(p.type + " " + p.name + ";\n");
  }
  for (auto s : scorers) {
    member_declaration.append(s.type + " " + s.name + ";\n");
  }

  auto format_code = [](const std::string &base_template,
                        const std::string &point_type,
                        const std::string &deserialization_filenames,
                        const std::string &init_list,
                        const std::string &load_query,
                        const std::string &query_steps,
                        const std::string &getters,
                        const std::string &member_declaration) {
    // https://stackoverflow.com/a/26221725
    size_t size =
        snprintf(nullptr, 0, base_template.c_str(), point_type.c_str(),
                 deserialization_filenames.c_str(), init_list.c_str(),
                 point_type.c_str(), load_query.c_str(), query_steps.c_str(),
                 getters.c_str(), member_declaration.c_str()) +
        1;
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, base_template.c_str(), point_type.c_str(),
                  deserialization_filenames.c_str(), init_list.c_str(),
                  point_type.c_str(), load_query.c_str(), query_steps.c_str(),
                  getters.c_str(), member_declaration.c_str());
    return std::string(buf.get(), buf.get() + size - 1);
  };
  auto is_there_serializable =
      [](const std::vector<PipeParameters *> &pipe_params) {
        bool is_serializable = false;
        for (auto p : pipe_params) {
          is_serializable |= p->is_serializable_;
        }
        return is_serializable;
      };
  const std::string deserialization_filenames =
      is_there_serializable(pipe_params)
          ? ",\nconst std::map<std::string, std::string>& "
            "deserialization_filenames = {}"
          : "";
  return format_code(base_template, point_type, deserialization_filenames,
                     init_list, load_query, query_steps, getters,
                     member_declaration);
}

}  // namespace experimental
}  // namespace falconn

#endif /* __CODE_GENERATION__ */
