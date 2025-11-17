

#include "minddata/versadf/include/shard_index.h"

namespace mindspore {
namespace versadf {
// table name for index
const char TABLENAME[] = "index_table";

Index::Index() : database_name_(""), table_name_(TABLENAME) {}

void Index::AddIndexField(const int64_t &schemaId, const std::string &field) {
  (void)fields_.emplace_back(pair<int64_t, string>(schemaId, field));
}

// Get attribute list
std::vector<std::pair<uint64_t, std::string>> Index::GetFields() { return fields_; }
}  // namespace versadf
}  // namespace mindspore
