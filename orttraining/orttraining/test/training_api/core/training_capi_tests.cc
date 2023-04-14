// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_training_cxx_api.h"

#include "orttraining/training_api/checkpoint.h"

#include "orttraining/test/training_api/core/data_utils.h"

namespace onnxruntime::training::test {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

TEST(TrainingCApiTest, GetState) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(Ort::SessionOptions(), checkpoint_state, model_uri);

  Ort::CheckpointState state = training_session.GetState(false);

  // Check if the retrieved state and the original checkpoint state are equal
  OrtCheckpointState* checkpoint_state_c = checkpoint_state;
  OrtCheckpointState* state_c = state;
  auto checkpoint_state_internal = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state_c);
  auto state_internal = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(state_c);

  ASSERT_FALSE(checkpoint_state_internal->module_checkpoint_state.named_parameters.empty());
  ASSERT_EQ(checkpoint_state_internal->module_checkpoint_state.named_parameters.size(),
            state_internal->module_checkpoint_state.named_parameters.size());
  for (auto& [key, value] : checkpoint_state_internal->module_checkpoint_state.named_parameters) {
    ASSERT_TRUE(state_internal->module_checkpoint_state.named_parameters.count(key));
  }
}

TEST(TrainingCApiTest, AddIntProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  int64_t value = 365 * 24;

  checkpoint_state.AddProperty("hours in a year", value);

  auto property = checkpoint_state.GetProperty("hours in a year");

  ASSERT_EQ(std::get<int64_t>(property), value);
}

TEST(TrainingCApiTest, AddFloatProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  float value = 3.14f;

  checkpoint_state.AddProperty("pi", value);

  auto property = checkpoint_state.GetProperty("pi");

  ASSERT_EQ(std::get<float>(property), value);
}

TEST(TrainingCApiTest, AddStringProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  std::string value("onnxruntime");

  checkpoint_state.AddProperty("framework", value);

  auto property = checkpoint_state.GetProperty("framework");

  ASSERT_EQ(std::get<std::string>(property), value);
}

TEST(TrainingCApiTest, InputNames) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(Ort::SessionOptions(), checkpoint_state, model_uri);

  const auto input_names = training_session.InputNames(true);
  ASSERT_EQ(input_names.size(), 2U);
  ASSERT_EQ(input_names.front(), "input-0");
  ASSERT_EQ(input_names.back(), "labels");
}

TEST(TrainingCApiTest, OutputNames) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(Ort::SessionOptions(), checkpoint_state, model_uri);

  const auto output_names = training_session.OutputNames(true);
  ASSERT_EQ(output_names.size(), 1U);
  ASSERT_EQ(output_names.front(), "onnx::loss::21273");
}

TEST(TrainingCApiTest, ToBuffer) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(Ort::SessionOptions(), checkpoint_state, model_uri);

  Ort::Value buffer = training_session.ToBuffer(true);

  ASSERT_TRUE(buffer.IsTensor());
  auto tensor_info = buffer.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  ASSERT_EQ(shape.size(), 1U);
  ASSERT_EQ(shape.front(), static_cast<int64_t>(397510));

  buffer = training_session.ToBuffer(false);

  ASSERT_TRUE(buffer.IsTensor());
  tensor_info = buffer.GetTensorTypeAndShapeInfo();
  shape = tensor_info.GetShape();
  ASSERT_EQ(shape.size(), 1U);
  ASSERT_EQ(shape.front(), static_cast<int64_t>(397510));
}

TEST(TrainingCApiTest, FromBuffer) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(Ort::SessionOptions(), checkpoint_state, model_uri);

  OrtValue* buffer_impl = std::make_unique<OrtValue>().release();
  GenerateRandomInput(std::array<int64_t, 1>{397510}, *buffer_impl);

  Ort::Value buffer(buffer_impl);

  training_session.FromBuffer(buffer);
}

}  // namespace onnxruntime::training::test
