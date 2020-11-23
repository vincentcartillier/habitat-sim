// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

uniform samplerCube textureData;

out vec4 fragmentColor;

void main(void) {
  fragmentColor = texture(textureData, vec3(1.0, 1.0, 1.0));
}