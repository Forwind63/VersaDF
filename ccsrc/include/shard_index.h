

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INDEX_H
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INDEX_H
#pragma once

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "minddata/mindrecord/include/common/log_adapter.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/mindrecord_macro.h"
#include "minddata/mindrecord/include/shard_schema.h"

namespace mindspore {
namespace mindrecord {
using std::cin;
using std::endl;
using std::pair;
using std::string;
using std::vector;

class MINDRECORD_API Index {
 public:
  Index();

  ~Index() {}

  /// \brief Add field which from schema according to schemaId
  /// \param[in] schemaId the id of schema to be added
  /// \param[in] field the field need to be added
  ///
  /// add the field to the fields_ vector
  void AddIndexField(const int64_t &schemaId, const std::string &field);

  /// \brief get stored fields
  /// \return fields stored
  std::vector<std::pair<uint64_t, std::string> > GetFields();

 private:
  std::vector<std::pair<uint64_t, std::string> > fields_;
  string database_name_;
  string table_name_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INDEX_H
