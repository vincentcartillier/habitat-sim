// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "Recorder.h"

#include "esp/assets/RenderAssetInstanceCreationInfo.h"
#include "esp/io/json.h"
#include "esp/scene/SceneNode.h"

namespace esp {
namespace gfx {
namespace replay {

/**
 * @brief Helper class to get notified when a SceneNode is about to be
 * destroyed.
 */
class NodeDeletionHelper : public Magnum::SceneGraph::AbstractFeature3D {
 public:
  NodeDeletionHelper(scene::SceneNode& node_, Recorder* writer)
      : Magnum::SceneGraph::AbstractFeature3D(node_),
        node(&node_),
        recorder_(writer) {}

  ~NodeDeletionHelper() override {
    recorder_->onDeleteRenderAssetInstance(node);
  }

 private:
  Recorder* recorder_ = nullptr;
  const scene::SceneNode* node = nullptr;
};

Recorder::~Recorder() {
  // Delete NodeDeletionHelpers. This is important because they hold raw
  // pointers to this Recorder and these pointers would become dangling
  // (invalid) after this Recorder is destroyed.
  for (auto& instanceRecord : instanceRecords_) {
    delete instanceRecord.deletionHelper;
  }
}

void Recorder::onLoadRenderAsset(const esp::assets::AssetInfo& assetInfo) {
  getKeyframe().loads.push_back(assetInfo);
}

void Recorder::onCreateRenderAssetInstance(
    scene::SceneNode* node,
    const esp::assets::RenderAssetInstanceCreationInfo& creation) {
  ASSERT(node);
  ASSERT(findInstance(node) == ID_UNDEFINED);

  RenderAssetInstanceKey instanceKey = getNewInstanceKey();

  getKeyframe().creations.emplace_back(std::make_pair(instanceKey, creation));

  // Constructing NodeDeletionHelper here is equivalent to calling
  // node->addFeature. We keep a pointer to deletionHelper so we can delete it
  // manually later if necessary.
  NodeDeletionHelper* deletionHelper = new NodeDeletionHelper{*node, this};

  instanceRecords_.emplace_back(InstanceRecord{
      node, instanceKey, Corrade::Containers::NullOpt, deletionHelper});
}

void Recorder::saveKeyframe() {
  updateInstanceStates();
  advanceKeyframe();
}

void Recorder::addUserTransformToKeyframe(const std::string& name,
                                          const Magnum::Vector3& translation,
                                          const Magnum::Quaternion& rotation) {
  getKeyframe().userTransforms[name] = Transform{translation, rotation};
}

void Recorder::addLoadsCreationsDeletions(KeyframeIterator begin,
                                          KeyframeIterator end,
                                          Keyframe* dest) {
  ASSERT(dest);
  for (KeyframeIterator curr = begin; curr != end; curr++) {
    const auto& keyframe = *curr;
    dest->loads.insert(dest->loads.end(), keyframe.loads.begin(),
                       keyframe.loads.end());
    dest->creations.insert(dest->creations.end(), keyframe.creations.begin(),
                           keyframe.creations.end());
    for (const auto& deletionInstanceKey : keyframe.deletions) {
      checkAndAddDeletion(dest, deletionInstanceKey);
    }
  }
}

void Recorder::checkAndAddDeletion(Keyframe* keyframe,
                                   RenderAssetInstanceKey instanceKey) {
  auto it =
      std::find_if(keyframe->creations.begin(), keyframe->creations.end(),
                   [&](const auto& pair) { return pair.first == instanceKey; });
  if (it != keyframe->creations.end()) {
    // this deletion just cancels out with an earlier creation
    keyframe->creations.erase(it);
  } else {
    // This deletion has no matching creation so it can't be canceled out.
    // Include it in the keyframe.
    keyframe->deletions.push_back(instanceKey);
  }
}

void Recorder::onDeleteRenderAssetInstance(const scene::SceneNode* node) {
  int index = findInstance(node);
  ASSERT(index != ID_UNDEFINED);

  auto instanceKey = instanceRecords_[index].instanceKey;

  checkAndAddDeletion(&getKeyframe(), instanceKey);

  instanceRecords_.erase(instanceRecords_.begin() + index);
}

Keyframe& Recorder::getKeyframe() {
  return currKeyframe_;
}

RenderAssetInstanceKey Recorder::getNewInstanceKey() {
  return nextInstanceKey_++;
}

int Recorder::findInstance(const scene::SceneNode* queryNode) {
  auto it = std::find_if(instanceRecords_.begin(), instanceRecords_.end(),
                         [&queryNode](const InstanceRecord& record) {
                           return record.node == queryNode;
                         });

  return it == instanceRecords_.end() ? ID_UNDEFINED
                                      : int(it - instanceRecords_.begin());
}

RenderAssetInstanceState Recorder::getInstanceState(
    const scene::SceneNode* node) {
  const auto absTransformMat = node->absoluteTransformation();
  Transform absTransform{
      absTransformMat.translation(),
      Magnum::Quaternion::fromMatrix(absTransformMat.rotationShear())};

  return RenderAssetInstanceState{absTransform, node->getSemanticId()};
}

void Recorder::updateInstanceStates() {
  for (auto& instanceRecord : instanceRecords_) {
    auto state = getInstanceState(instanceRecord.node);
    if (!instanceRecord.recentState || state != instanceRecord.recentState) {
      getKeyframe().stateUpdates.push_back(
          std::make_pair(instanceRecord.instanceKey, state));
      instanceRecord.recentState = state;
    }
  }
}

void Recorder::advanceKeyframe() {
  savedKeyframes_.emplace_back(std::move(currKeyframe_));
  currKeyframe_ = Keyframe{};
}

void Recorder::writeSavedKeyframesToFile(const std::string& filepath) {
  auto document = writeKeyframesToJsonDocument();
  esp::io::writeJsonToFile(document, filepath);

  // consolidate saved keyframes into current keyframe
  addLoadsCreationsDeletions(savedKeyframes_.begin(), savedKeyframes_.end(),
                             &getKeyframe());
  savedKeyframes_.clear();
}

rapidjson::Document Recorder::writeKeyframesToJsonDocument() {
  LOG(WARNING) << "Recorder::writeKeyframesToJsonDocument: not implemented";
  return rapidjson::Document();
#if 0  // coming soon
  if (savedKeyframes_.empty()) {
    LOG(WARNING) << "Recorder::writeKeyframesToJsonDocument: no saved "
                    "keyframes to write";
    return rapidjson::Document();
  }

  Document d(kObjectType);
  Document::AllocatorType& allocator = d.GetAllocator();
  esp::io::AddMember(d, "keyframes", savedKeyframes_, allocator);
  return d;
#endif
}

}  // namespace replay
}  // namespace gfx
}  // namespace esp
