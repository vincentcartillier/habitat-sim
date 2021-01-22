// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_METADATA_ATTRIBUTES_OBJECTATTRIBUTES_H_
#define ESP_METADATA_ATTRIBUTES_OBJECTATTRIBUTES_H_

#include "AttributesBase.h"

#include "esp/assets/Asset.h"

namespace esp {
namespace metadata {
namespace attributes {

/**
 * @brief base attributes object holding attributes shared by all
 * @ref esp::metadata::attributes::ObjectAttributes and @ref
 * esp::metadata::attributes::StageAttributes objects; Should be treated as
 * abstract - should never be instanced directly
 */
class AbstractObjectAttributes : public AbstractAttributes {
 public:
  /**
   * @brief Constant static map to provide mappings from string tags to @ref
   * esp::assets::AssetType values.  This will be used to map values set in json
   * for mesh type to @ref esp::assets::AssetType.  Keys must be lowercase.
   */
  static const std::map<std::string, esp::assets::AssetType> AssetTypeNamesMap;
  AbstractObjectAttributes(const std::string& classKey,
                           const std::string& handle);

  virtual ~AbstractObjectAttributes() = default;

  /**
   * @brief Scale of the ojbect
   */
  void setScale(const Magnum::Vector3& scale) { setVec3("scale", scale); }
  Magnum::Vector3 getScale() const { return getVec3("scale"); }

  /**
   * @brief collision shape inflation margin
   */
  void setMargin(double margin) { setDouble("margin", margin); }
  double getMargin() const { return getDouble("margin"); }

  /**
   * @brief set default up orientation for object/stage mesh
   */
  void setOrientUp(const Magnum::Vector3& orientUp) {
    setVec3("orient_up", orientUp);
  }
  /**
   * @brief get default up orientation for object/stage mesh
   */
  Magnum::Vector3 getOrientUp() const { return getVec3("orient_up"); }
  /**
   * @brief set default forwardd orientation for object/stage mesh
   */
  void setOrientFront(const Magnum::Vector3& orientFront) {
    setVec3("orient_front", orientFront);
  }
  /**
   * @brief get default forwardd orientation for object/stage mesh
   */
  Magnum::Vector3 getOrientFront() const { return getVec3("orient_front"); }

  // units to meters mapping
  void setUnitsToMeters(double unitsToMeters) {
    setDouble("units_to_meters", unitsToMeters);
  }
  double getUnitsToMeters() const { return getDouble("units_to_meters"); }

  void setFrictionCoefficient(double frictionCoefficient) {
    setDouble("friction_coefficient", frictionCoefficient);
  }
  double getFrictionCoefficient() const {
    return getDouble("friction_coefficient");
  }

  void setRestitutionCoefficient(double restitutionCoefficient) {
    setDouble("restitution_coefficient", restitutionCoefficient);
  }
  double getRestitutionCoefficient() const {
    return getDouble("restitution_coefficient");
  }
  void setRenderAssetType(int renderAssetType) {
    setInt("render_asset_type", renderAssetType);
  }
  int getRenderAssetType() { return getInt("render_asset_type"); }

  void setRenderAssetHandle(const std::string& renderAssetHandle) {
    setString("render_asset", renderAssetHandle);
    setIsDirty();
  }
  std::string getRenderAssetHandle() const { return getString("render_asset"); }

  /**
   * @brief Sets whether this object uses file-based mesh render object or
   * primitive render shapes
   * @param renderAssetIsPrimitive whether this object's render asset is a
   * primitive or not
   */
  void setRenderAssetIsPrimitive(bool renderAssetIsPrimitive) {
    setBool("renderAssetIsPrimitive", renderAssetIsPrimitive);
  }

  bool getRenderAssetIsPrimitive() const {
    return getBool("renderAssetIsPrimitive");
  }

  void setCollisionAssetHandle(const std::string& collisionAssetHandle) {
    setString("collision_asset", collisionAssetHandle);
    setIsDirty();
  }
  std::string getCollisionAssetHandle() const {
    return getString("collision_asset");
  }

  void setCollisionAssetType(int collisionAssetType) {
    setInt("collision_asset_type", collisionAssetType);
  }
  int getCollisionAssetType() { return getInt("collision_asset_type"); }

  void setCollisionAssetSize(const Magnum::Vector3& collisionAssetSize) {
    setVec3("collision_asset_size", collisionAssetSize);
  }
  Magnum::Vector3 getCollisionAssetSize() const {
    return getVec3("collision_asset_size");
  }

  /**
   * @brief Sets whether this object uses file-based mesh collision object or
   * primitive(implicit) collision shapes
   * @param collisionAssetIsPrimitive whether this object's collision asset is a
   * primitive (implicitly calculated) or a mesh
   */
  void setCollisionAssetIsPrimitive(bool collisionAssetIsPrimitive) {
    setBool("collisionAssetIsPrimitive", collisionAssetIsPrimitive);
  }

  bool getCollisionAssetIsPrimitive() const {
    return getBool("collisionAssetIsPrimitive");
  }

  /**
   * @brief whether this object uses mesh collision or primitive(implicit)
   * collision calculation.
   */
  void setUseMeshCollision(bool useMeshCollision) {
    setBool("useMeshCollision", useMeshCollision);
  }

  bool getUseMeshCollision() const { return getBool("useMeshCollision"); }

  // if true use phong illumination model instead of flat shading
  void setRequiresLighting(bool requiresLighting) {
    setBool("requires_lighting", requiresLighting);
  }
  bool getRequiresLighting() const { return getBool("requires_lighting"); }

  bool getIsDirty() const { return getBool("__isDirty"); }
  void setIsClean() { setBool("__isDirty", false); }

 protected:
  void setIsDirty() { setBool("__isDirty", true); }

 public:
  ESP_SMART_POINTERS(AbstractObjectAttributes)

};  // class AbstractObjectAttributes

/**
 * @brief Specific Attributes instance describing an object, constructed with a
 * default set of object-specific required attributes
 */
class ObjectAttributes : public AbstractObjectAttributes {
 public:
  ObjectAttributes(const std::string& handle = "");
  // center of mass (COM)
  void setCOM(const Magnum::Vector3& com) { setVec3("COM", com); }
  Magnum::Vector3 getCOM() const { return getVec3("COM"); }

  // whether com is provided or not
  void setComputeCOMFromShape(bool computeCOMFromShape) {
    setBool("computeCOMFromShape", computeCOMFromShape);
  }
  bool getComputeCOMFromShape() const { return getBool("computeCOMFromShape"); }

  void setMass(double mass) { setDouble("mass", mass); }
  double getMass() const { return getDouble("mass"); }

  // inertia diagonal
  void setInertia(const Magnum::Vector3& inertia) {
    setVec3("inertia", inertia);
  }
  Magnum::Vector3 getInertia() const { return getVec3("inertia"); }

  void setLinearDamping(double linearDamping) {
    setDouble("linear_damping", linearDamping);
  }
  double getLinearDamping() const { return getDouble("linear_damping"); }

  void setAngularDamping(double angularDamping) {
    setDouble("angular_damping", angularDamping);
  }
  double getAngularDamping() const { return getDouble("angular_damping"); }

  // if true override other settings and use render mesh bounding box as
  // collision object
  void setBoundingBoxCollisions(bool useBoundingBoxForCollision) {
    setBool("use_bounding_box_for_collision", useBoundingBoxForCollision);
  }
  bool getBoundingBoxCollisions() const {
    return getBool("use_bounding_box_for_collision");
  }

  // if true join all mesh components of an asset into a unified collision
  // object
  void setJoinCollisionMeshes(bool joinCollisionMeshes) {
    setBool("join_collision_meshes", joinCollisionMeshes);
  }
  bool getJoinCollisionMeshes() const {
    return getBool("join_collision_meshes");
  }

  /**
   * @brief If not visible can add dynamic non-rendered object into a scene
   * object.  If is not visible then should not add object to drawables.
   */
  void setIsVisible(bool isVisible) { setBool("isVisible", isVisible); }
  bool getIsVisible() const { return getBool("isVisible"); }

  //void setSemanticId(uint32_t semanticId) { setInt("semanticId", semanticId); }
  void setSemanticId(uint32_t semanticId) { setInt("semantic_id", semanticId); }

  //uint32_t getSemanticId() const { return getInt("semanticId"); }
  uint32_t getSemanticId() const { return getInt("semantic_id"); }

  // if object should be checked for collisions - if other objects can collide
  // with this object
  void setIsCollidable(bool isCollidable) {
    setBool("isCollidable", isCollidable);
  }
  bool getIsCollidable() { return getBool("isCollidable"); }

 public:
  ESP_SMART_POINTERS(ObjectAttributes)

};  // class ObjectAttributes

///////////////////////////////////////
// stage attributes

/**
 * @brief Specific Attributes instance describing a stage, constructed with a
 * default set of stage-specific required attributes
 */
class StageAttributes : public AbstractObjectAttributes {
 public:
  StageAttributes(const std::string& handle = "");

  void setOrigin(const Magnum::Vector3& origin) { setVec3("origin", origin); }
  Magnum::Vector3 getOrigin() const { return getVec3("origin"); }

  void setGravity(const Magnum::Vector3& gravity) {
    setVec3("gravity", gravity);
  }
  Magnum::Vector3 getGravity() const { return getVec3("gravity"); }
  void setHouseFilename(const std::string& houseFilename) {
    setString("houseFilename", houseFilename);
    setIsDirty();
  }
  std::string getHouseFilename() const { return getString("houseFilename"); }
  void setSemanticAssetHandle(const std::string& semanticAssetHandle) {
    setString("semanticAssetHandle", semanticAssetHandle);
    setIsDirty();
  }
  std::string getSemanticAssetHandle() const {
    return getString("semanticAssetHandle");
  }
  void setSemanticAssetType(int semanticAssetType) {
    setInt("semanticAssetType", semanticAssetType);
  }
  int getSemanticAssetType() { return getInt("semanticAssetType"); }

  void setLoadSemanticMesh(bool loadSemanticMesh) {
    setBool("loadSemanticMesh", loadSemanticMesh);
  }
  bool getLoadSemanticMesh() { return getBool("loadSemanticMesh"); }

  void setNavmeshAssetHandle(const std::string& navmeshAssetHandle) {
    setString("navmeshAssetHandle", navmeshAssetHandle);
    setIsDirty();
  }
  std::string getNavmeshAssetHandle() const {
    return getString("navmeshAssetHandle");
  }

  /**
   * @brief set lighting setup for scene.  Default value comes from
   * @ref esp::sim::SimulatorConfiguration, is overridden by any value set in
   * json, if exists.
   */
  void setLightSetup(const std::string& lightSetup) {
    setString("lightSetup", lightSetup);
  }
  std::string getLightSetup() { return getString("lightSetup"); }

  void setFrustrumCulling(bool frustrumCulling) {
    setBool("frustrumCulling", frustrumCulling);
  }
  bool getFrustrumCulling() const { return getBool("frustrumCulling"); }

 public:
  ESP_SMART_POINTERS(StageAttributes)

};  // class StageAttributes

}  // namespace attributes
}  // namespace metadata
}  // namespace esp

#endif  // ESP_METADATA_ATTRIBUTES_OBJECTATTRIBUTES_H_
