import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import onnx
import onnxruntime as ort
from typing import Dict, Any
import argparse
import os
import shutil
import numpy as np
import onnxsim
import onnx
import MNN
import MNN.numpy as mnp
import subprocess



class RobertaModel(nn.Module):
    """Wrapper class that combines BERT tokenizer preprocessing and model inference."""

    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).half()

    def forward(self, text_input: torch.Tensor):
        """Forward pass that includes tokenization and model inference."""
        # Note: For ONNX export, we'll work with pre-tokenized input_ids
        # In practice, text tokenization needs to happen outside ONNX
        input_ids = text_input.long()

        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        return torch.cat(outputs["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

def export_bert_to_onnx(
    model_name: str = "bert-base-uncased",
    output_dir: str = "bert_exported"
):
    """Export BERT model to ONNX format and copy tokenizer files."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    combined_model = RobertaModel(model_name)
    combined_model.eval()

    # Create dummy inputs for ONNX export (pre-tokenized input_ids)
    batch_size = 1
    dummy_input_ids = torch.randint(0, combined_model.tokenizer.vocab_size, (batch_size, 512))

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "chinese-roberta-wwm-ext-large.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        combined_model,
        dummy_input_ids,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'logits_length'}
        }
    )
    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Simplify the model
    model_simplified, _ = onnxsim.simplify(model)
    # Save the simplified model
    onnx.save(model_simplified, onnx_path)

    # Export to MNN format
    mnn_path = os.path.join(output_dir, "chinese-roberta-wwm-ext-large.mnn")
    print(f"Exporting to MNN: {mnn_path}")
    mnn_command = [
        "mnnconvert",
        "--f", "ONNX",
        "--modelFile", onnx_path,
        "--optimizeLevel", "2",
        "--optimizePrefer", "2",
        "--MNNModel", mnn_path,
        "--weightQuantBits", "8"
    ]

    try:
        subprocess.run(mnn_command, check=True, capture_output=True, text=True)
        print(f"Successfully exported to MNN: {mnn_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting to MNN: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

    # Copy tokenizer.json if it exists
    tokenizer_cache_dir = combined_model.tokenizer.name_or_path
    if os.path.isdir(tokenizer_cache_dir):
        tokenizer_json_path = os.path.join(tokenizer_cache_dir, "tokenizer.json")
    else:
        # For models from HuggingFace cache
        from transformers import cached_path
        try:
            tokenizer_json_path = combined_model.tokenizer._tokenizer.model_path
        except:
            # Alternative approach to find tokenizer.json in cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
            tokenizer_json_path = None
            for root, dirs, files in os.walk(cache_dir):
                if "tokenizer.json" in files and model_name.replace("/", "--") in root:
                    tokenizer_json_path = os.path.join(root, "tokenizer.json")
                    break

    if tokenizer_json_path and os.path.exists(tokenizer_json_path):
        dest_tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        shutil.copy2(tokenizer_json_path, dest_tokenizer_path)
        print(f"Copied tokenizer.json to: {dest_tokenizer_path}")
    else:
        print("Warning: tokenizer.json not found")

    # Copy config.json if it exists
    if tokenizer_cache_dir and os.path.isdir(tokenizer_cache_dir):
        config_json_path = os.path.join(tokenizer_cache_dir, "config.json")
    else:
        # For models from HuggingFace cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        config_json_path = None
        for root, dirs, files in os.walk(cache_dir):
            if "config.json" in files and model_name.replace("/", "--") in root:
                config_json_path = os.path.join(root, "config.json")
                break

    if config_json_path and os.path.exists(config_json_path):
        dest_config_path = os.path.join(output_dir, "config.json")
        shutil.copy2(config_json_path, dest_config_path)
        print(f"Copied config.json to: {dest_config_path}")
    else:
        print("Warning: config.json not found")

    print(f"Model exported successfully to: {output_dir}")
    return combined_model, onnx_path, mnn_path

def test_model_equivalence(original_model, onnx_path: str, mnn_path: str = None):
    """Test if the original PyTorch model, ONNX model, and MNN model produce the same outputs."""

    print("Testing model equivalence...")

    # Create test input
    input_ids = original_model.tokenizer.encode("原神，启动！", return_tensors="pt")


    # Get PyTorch output
    original_model.eval()
    with torch.no_grad():
        pytorch_output = original_model(input_ids).numpy()

    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {"input_ids": input_ids.numpy()})[0]


    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")

    if mean_diff < 1e-3:
        print("✅ PyTorch and ONNX models are numerically equivalent!")
    else:
        print("❌ PyTorch and ONNX models have significant differences!")

    # Test MNN model if path is provided
    if mnn_path and os.path.exists(mnn_path):
        print("\nTesting MNN model...")
        try:
            mnn_config = {}
            mnn_config['backend'] = 'CPU'
            mnn_config['thread'] = 12
            mnn_rt = MNN.nn.create_runtime_manager((mnn_config,))

            roberta_mnn = MNN.nn.load_module_from_file(mnn_path,
                                                      ['input_ids'], ['logits'], runtime_manager=mnn_rt)
            mnn_output = np.array(roberta_mnn(mnp.array(input_ids.numpy()).astype(mnp.int32)).read())

            print(f"MNN output shape: {mnn_output.shape}")

            # Compare MNN with PyTorch
            max_diff_mnn_pt = np.max(np.abs(pytorch_output - mnn_output))
            mean_diff_mnn_pt = np.mean(np.abs(pytorch_output - mnn_output))

            print(f"MNN vs PyTorch - Maximum absolute difference: {max_diff_mnn_pt}")
            print(f"MNN vs PyTorch - Mean absolute difference: {mean_diff_mnn_pt}")

            if mean_diff_mnn_pt < 1e-2:
                print("✅ MNN and PyTorch models are numerically equivalent!")
                return True
            else:
                print("❌ MNN and PyTorch models have significant differences!")
                return False

        except Exception as e:
            print(f"❌ Error testing MNN model: {e}")
            return False
    else:
        print("MNN model path not provided or file does not exist, skipping MNN test")
        return mean_diff < 1e-3

def main():
    parser = argparse.ArgumentParser(description="Export BERT model to ONNX")
    parser.add_argument("--model_name", type=str, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", 
                       help="Pretrained BERT model name")
    parser.add_argument("--output_dir", type=str, default="onnx/chinese-roberta-wwm-ext-large",
                       help="Output directory path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Export model
    original_model, onnx_path, mnn_path = export_bert_to_onnx(
        model_name=args.model_name,
        output_dir=args.output_dir
    )

    # Test equivalence
    test_model_equivalence(
        original_model=original_model,
        onnx_path=onnx_path,
        mnn_path=mnn_path
    )

if __name__ == "__main__":
    main()