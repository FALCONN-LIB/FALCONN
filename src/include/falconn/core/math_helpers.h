#ifndef __MATH_HELPERS_H__
#define __MATH_HELPERS_H__

namespace falconn {
namespace core {

template<typename Point>
struct NormalizationHelper {
  static void normalize(Point*) {
    static_assert(FalseStruct<Point>::value, "Point type not supported.");
  }
  template<typename T> struct FalseStruct : std::false_type {};
};

template<typename CoordinateType>
struct NormalizationHelper<DenseVector<CoordinateType>> {
  static void normalize(DenseVector<CoordinateType>* p) {
    p->normalize();
  }
};

template<typename Point>
void normalize(Point* p) {
  NormalizationHelper<Point>::normalize(p);
}

}  // namsepace core
}  // namespace falconn

#endif
