#ifndef _SERIALIZE_H_
#define _SERIALIZE_H_

#include "common.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cstdint>

namespace ir {

using std::runtime_error;
using std::string;
using std::vector;

class SerializeError : public std::logic_error {
 public:
  SerializeError(const char *msg) : std::logic_error(msg) {}
};

template <typename T>
struct IsElementary : std::false_type {};

template <>
struct IsElementary<float> : std::true_type {};
template <>
struct IsElementary<double> : std::true_type {};
template <>
struct IsElementary<uint8_t> : std::true_type {};
template <>
struct IsElementary<uint16_t> : std::true_type {};
template <>
struct IsElementary<uint32_t> : std::true_type {};
template <>
struct IsElementary<uint64_t> : std::true_type {};
template <>
struct IsElementary<int8_t> : std::true_type {};
template <>
struct IsElementary<int16_t> : std::true_type {};
template <>
struct IsElementary<int32_t> : std::true_type {};
template <>
struct IsElementary<int64_t> : std::true_type {};

template <typename T, typename dummy = void>
struct Serializer;

template <typename T>
struct Serializer<T,
                  typename std::enable_if<IsElementary<T>::value, void>::type> {
  static void serialize(FILE *output, T entity) {
    if (fwrite(&entity, sizeof(T), 1, output) != 1) {
      throw SerializeError("can't write");
    }
  }

  static void deserialize(FILE *input, T *entity) {
    if (fread(entity, sizeof(T), 1, input) != 1) {
      throw SerializeError("can't read");
    }
  }
};

template <typename S>
struct Serializer<vector<S>,
                  typename std::enable_if<IsElementary<S>::value, void>::type> {
  static void serialize(FILE *output, const vector<S> &entity) {
    Serializer<size_t>::serialize(output, entity.size());
    if (fwrite(&entity[0], sizeof(S), entity.size(), output) != entity.size()) {
      throw SerializeError("can't write");
    }
  }

  static void deserialize(FILE *input, vector<S> *entity) {
    size_t s;
    Serializer<size_t>::deserialize(input, &s);
    entity->resize(s);
    if (fread(&(*entity)[0], sizeof(S), s, input) != s) {
      throw SerializeError("can't read");
    }
  }
};

template <>
struct Serializer<Point> {
  static void serialize(FILE *output, const Point &entity) {
    vector<float> aux(entity.size());
    for (size_t i = 0; i < aux.size(); ++i) {
      aux[i] = entity[i];
    }
    Serializer<vector<float>>::serialize(output, aux);
  }

  static void deserialize(FILE *input, Point *entity) {
    vector<float> aux;
    Serializer<vector<float>>::deserialize(input, &aux);
    entity->resize(aux.size());
    for (size_t i = 0; i < aux.size(); ++i) {
      (*entity)[i] = aux[i];
    }
  }
};

template <typename S>
struct Serializer<
    vector<S>, typename std::enable_if<!IsElementary<S>::value, void>::type> {
  static void serialize(FILE *output, const vector<S> &entity) {
    Serializer<size_t>::serialize(output, entity.size());
    for (auto &element : entity) {
      Serializer<S>::serialize(output, element);
    }
  }

  static void deserialize(FILE *input, vector<S> *entity) {
    size_t s;
    Serializer<size_t>::deserialize(input, &s);
    entity->resize(s);
    for (size_t i = 0; i < s; ++i) {
      Serializer<S>::deserialize(input, &(*entity)[i]);
    }
  }
};

template <typename T>
void serialize(FILE *output, const T &entity) {
  Serializer<T>::serialize(output, entity);
}

template <typename T>
void serialize(string file_name, const T &entity) {
  FILE *output = fopen(file_name.c_str(), "wb");
  if (!output) {
    throw SerializeError("can't open file for writing");
  }
  serialize(output, entity);
  if (fclose(output)) {
    throw SerializeError("can't close");
  }
}

template <typename T>
void deserialize(FILE *input, T *entity) {
  Serializer<T>::deserialize(input, entity);
}

template <typename T>
void deserialize(string file_name, T *entity) {
  FILE *input = fopen(file_name.c_str(), "rb");
  if (!input) {
    throw SerializeError("can't open file for reading");
  }
  deserialize(input, entity);
  if (fclose(input)) {
    throw SerializeError("can't close");
  }
}

};  // namespace ir

#endif
