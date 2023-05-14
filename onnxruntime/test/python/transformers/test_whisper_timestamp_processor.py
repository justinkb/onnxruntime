0~# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest
import pytest
import onnx
import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions

class TestTimestampProcessor(unittest.TestCase):
    def generate_model(self, arguments: str):
        from onnxruntime.transformers.models.whisper.convert_to_onnx import main as whisper_to_onnx
        whisper_to_onnx(arguments.split())

    def generate_dataset(self):
        from transformers import AutoProcessor
        from datasets import load_dataset
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        print("ds: ", ds)
        inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        input_features = inputs.input_features
        print("input_features shape: ", input_features.shape)
        print("input_features: ", input_features)
        return [input_features, processor]

    def run_timestamp(self, provider: str):
        generate_model(f"-m openai/whisper-tiny --optimize_onnx --precision fp32 --use_external_data_format")
        [input_features, processor] = generate_dataset()
        min_length = 0
        max_length = 128
        beam_size = 1
        NUM_RETURN_SEQUENCES = 1
        repetition_penalty = 1.0
        model_path = "./onnx_models/openai/whisper-tiny_beamsearch.onnx"
        sess_options = SessionOptions()
        sess_options.log_severity_level = 4
        sess = InferenceSession(model_path, sess_options, providers=[provider])
        input_data = input_features.repeat(1, 1, 1)
        ort_inputs = {
            "input_features": np.float32(input_data.cpu().numpy()),
            "max_length": np.array([128], dtype=np.int32),
            "min_length": np.array([0], dtype=np.int32),
            "num_beams": np.array([1], dtype=np.int32),
            "num_return_sequences": np.array([1], dtype=np.int32),
            "length_penalty": np.array([1.0], dtype=np.float32),
            "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
            "attention_mask": np.zeros(input_data.shape).astype(np.int32),
            "timestamp_enable": np.array([True], dtype=bool),
        }
        ort_out = sess.run(None, ort_inputs)
        ort_out_tensor = torch.from_numpy(ort_out[0])
        print("ort_out_tensor: ", ort_out_tensor)
        ort_transcription = processor.batch_decode(ort_out_tensor[0][0].view(1, -1), skip_special_tokens=True, output_offsets=True)##[0]
        print("whisper: ort_transcription: ", ort_transcription)
        expected_transcription = [{'text': ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.', 'offsets': [{'text': ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.', 'timestamp': (0.0, 5.44)}]}]
        self.assertEqual(ort_transcription, expected_transcription)

    @pytest.mark.slow
    def test_timestamp_cpu(self):
        provider = "CPUExecutionProvider"
        run_timestamp(provider)


if __name__ == "__main__":
    unittest.main()
