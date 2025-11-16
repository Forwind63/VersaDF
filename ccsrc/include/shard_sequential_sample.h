

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SEQUENTIAL_SAMPLE_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SEQUENTIAL_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_sample.h"

namespace mindspore {
namespace mindrecord {
class MINDRECORD_API ShardSequentialSample : public ShardSample {
 public:
  ShardSequentialSample(int64_t n, int64_t offset);

  ShardSequentialSample(float per, float per_offset);

  ~ShardSequentialSample() override{};

  std::string Name() override { return "ShardSequentialSample"; }

  Status Execute(ShardTaskList &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  int64_t offset_;
  float per_;
  float per_offset_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SEQUENTIAL_SAMPLE_H_
