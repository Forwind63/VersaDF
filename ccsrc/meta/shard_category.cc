

#include "minddata/mindrecord/include/shard_category.h"

namespace mindspore {
namespace mindrecord {
ShardCategory::ShardCategory(const std::vector<std::pair<std::string, std::string>> &categories, int64_t num_elements,
                             bool replacement)
    : categories_(categories),
      category_field_(""),
      num_elements_(num_elements),
      num_categories_(0),
      replacement_(replacement) {}

ShardCategory::ShardCategory(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             bool replacement)
    : categories_({}),
      category_field_(category_field),
      num_elements_(num_elements),
      num_categories_(num_categories),
      replacement_(replacement) {}

Status ShardCategory::Execute(ShardTaskList &tasks) { return Status::OK(); }

int64_t ShardCategory::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (dataset_size == 0) {
    return dataset_size;
  }
  if (dataset_size > 0 && num_classes > 0 && num_categories_ > 0 && num_elements_ > 0) {
    num_classes = std::min(num_categories_, num_classes);
    if (num_classes == 0) {
      return 0;
    }
    if (num_elements_ > std::numeric_limits<int64_t>::max() / num_classes) {
      return -1;
    }
    return num_classes * num_elements_;
  }
  return 0;
}
}  // namespace mindrecord
}  // namespace mindspore
