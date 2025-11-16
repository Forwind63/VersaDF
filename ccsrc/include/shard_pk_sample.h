

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "minddata/mindrecord/include/shard_category.h"

namespace mindspore {
namespace mindrecord {
class MINDRECORD_API ShardPkSample : public ShardCategory {
 public:
  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_samples);

  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories, int64_t num_samples);

  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories, uint32_t seed,
                int64_t num_samples);

  ~ShardPkSample() override{};

  std::string Name() override { return "ShardPkSample"; }

  Status SufExecute(ShardTaskList &tasks) override;

  int64_t GetNumSamples() const { return num_samples_; }

 private:
  bool shuffle_;
  std::shared_ptr<ShardShuffle> shuffle_op_;
  int64_t num_samples_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_
