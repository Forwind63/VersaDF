

#pragma once
#ifndef MINDSPORE_CCSRC_MINDDATA_versadf_STATISTICS_H
#define MINDSPORE_CCSRC_MINDDATA_versadf_STATISTICS_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "minddata/versadf/include/common/log_adapter.h"
#include "minddata/versadf/include/common/shard_pybind.h"
#include "minddata/versadf/include/common/shard_utils.h"
#include "minddata/versadf/include/versadf_macro.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace versadf {
class versadf_API Statistics {
 public:
  /// \brief save the statistic and its description
  /// \param[in] desc the statistic's description
  /// \param[in] statistics the statistic needs to be saved
  static std::shared_ptr<Statistics> Build(std::string desc, const json &statistics);

  ~Statistics() = default;

  /// \brief compare two statistics to judge if they are equal
  /// \param b another statistics to be judged
  /// \return true if they are equal,false if not
  bool operator==(const Statistics &b) const;

  /// \brief get the description
  /// \return the description
  std::string GetDesc() const;

  /// \brief get the statistic
  /// \return json format of the statistic
  json GetStatistics() const;

  /// \brief decode the bson statistics to json
  /// \param[in] encodedStatistics the bson type of statistics
  /// \return json type of statistic
  void SetStatisticsID(int64_t id);

  /// \brief get the statistics id
  /// \return the int64 statistics id
  int64_t GetStatisticsID() const;

 private:
  /// \brief validate the statistic
  /// \return true / false
  static bool Validate(const json &statistics);

  static bool LevelRecursive(json level);

  Statistics() = default;

  std::string desc_;
  json statistics_;
  int64_t statistics_id_ = -1;
};
}  // namespace versadf
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_versadf_STATISTICS_H
