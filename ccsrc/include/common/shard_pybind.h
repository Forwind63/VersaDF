

#ifndef MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_COMMON_SHARD_PYBIND_H_
#define MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_COMMON_SHARD_PYBIND_H_

#include <string>
#include <vector>
#include "minddata/versadf/include/common/shard_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace nlohmann {
template <>
struct adl_serializer<py::object> {
  py::object FromJson(const json &j);

  void ToJson(json *j, const py::object &obj);
};

namespace detail {
py::object FromJsonImpl(const json &j);

json ToJsonImpl(const py::handle &obj);
}  // namespace detail
}  // namespace nlohmann
#endif  // MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_COMMON_SHARD_PYBIND_H_
