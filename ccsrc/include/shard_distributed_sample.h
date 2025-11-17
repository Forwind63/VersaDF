

#ifndef MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_SHARD_DISTRIBUTED_SAMPLE_H_
#define MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_SHARD_DISTRIBUTED_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/versadf/include/shard_shuffle.h"
#include "minddata/versadf/include/shard_sample.h"

namespace mindspore {
namespace versadf {
class versadf_API ShardDistributedSample : public ShardSample {
 public:
  ShardDistributedSample(int num_shards, int shard_id, int64_t no_of_padded_samples, dataset::ShuffleMode shuffle_mode,
                         uint32_t seed, int64_t no_of_samples = 0, int64_t offset = -1);

  ShardDistributedSample(int num_shards, int shard_id, dataset::ShuffleMode shuffle_mode, uint32_t seed,
                         int64_t no_of_samples = 0, int64_t offset = -1);

  void SetNumPaddedSamples(int64_t no_of_padded_samples) { no_of_padded_samples_ = no_of_padded_samples; }

  ~ShardDistributedSample() override{};

  std::string Name() override { return "ShardDistributedSample"; }

  virtual void UpdateShuffleMode(dataset::ShuffleMode shuffle_mode) {
    ShardSample::UpdateShuffleMode(shuffle_mode);
    if (shuffle_op_) {
      shuffle_op_->UpdateShuffleMode(shuffle_mode);
    }
  }

  Status PreExecute(ShardTaskList &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  int64_t no_of_padded_samples_;
  bool first_epoch_;    // check (num_sample + num_padded) % num_shards == 0 in first epoch
  ShardTaskList task_;  // maintain the input tasks in first epoch
};
}  // namespace versadf
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_versadf_INCLUDE_SHARD_DISTRIBUTED_SAMPLE_H_
