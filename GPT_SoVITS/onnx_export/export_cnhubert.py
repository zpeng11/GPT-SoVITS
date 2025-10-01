import os
import tempfile
import urllib.request
import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import argparse
import subprocess
import shutil
from transformers import HubertModel, HubertConfig


# Constants
TEST_AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/4.flac"
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 160000  # 10 seconds
ZERO_PADDING_LENGTH = 9600  # 3200 * 0.3


def download_audio_from_url(url):
    """Download audio file from URL to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.flac') as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        return tmp_file.name


def load_and_preprocess_audio(audio_path):
    """Load and preprocess audio file"""
    # Handle URL by downloading first
    if audio_path.startswith('http'):
        audio_path = download_audio_from_url(audio_path)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]

    # Limit length for testing
    if waveform.shape[1] > MAX_AUDIO_LENGTH:
        waveform = waveform[:, :MAX_AUDIO_LENGTH]

    # Add zero padding (3200 * 0.3 = 9600 samples)
    zero_padding = torch.zeros((1, ZERO_PADDING_LENGTH), dtype=torch.float16)
    waveform = torch.cat([waveform, zero_padding], dim=1)

    print(f"Processed waveform shape: {waveform.shape}, zero padding shape: {zero_padding.shape}")

    return waveform


def export_hubert_to_onnx(
    model_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    output_dir: str = "onnx/chinese-hubert-base"
):
    """Export HuBERT model to ONNX format (half-precision)"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    onnx_path = os.path.join(output_dir, "chinese-hubert-base.onnx")
    mnn_path = os.path.join(output_dir, "chinese-hubert-base.mnn")

    print(f"Loading HuBERT model from: {model_path}")

    # Configure for better ONNX compatibility
    config = HubertConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"  # Use standard attention
    config.apply_spec_augment = False      # Disable masking for inference
    config.layerdrop = 0.0                 # Disable layer dropout

    # Load the model in half precision
    model = HubertModel.from_pretrained(
        model_path,
        config=config,
        local_files_only=True
    ).half()
    model.eval()

    # Create dummy input (1 second at 16kHz) - use float16 for half-precision export
    dummy_input = torch.rand(1, 16000, dtype=torch.float16) - 0.5

    # Export to ONNX
    print(f"Exporting half-precision model to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['audio16k'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'audio16k': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
        }
    )

    print(f"Half-precision model exported successfully to: {onnx_path}")

    # Export to MNN format
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
        mnn_path = None

    return model, onnx_path, mnn_path


def test_model_equivalence(original_model, onnx_path: str, mnn_path: str = None, test_audio_url: str = TEST_AUDIO_URL):
    """Test if the original PyTorch model, ONNX model, and MNN model produce the same outputs"""

    print("Testing model equivalence (half-precision)...")
    print(f"Using test audio: {test_audio_url}")

    # Load and preprocess audio
    waveform = load_and_preprocess_audio(test_audio_url)

    # Get PyTorch output (use half precision)
    print("Running PyTorch inference...")
    original_model.eval()
    with torch.no_grad():
        waveform_half = waveform.half()
        torch_output = original_model(waveform_half)
        torch_hidden_states = torch_output.last_hidden_state

    # Get ONNX output
    print("Running ONNX inference...")
    ort_session = ort.InferenceSession(onnx_path)
    input_values = waveform.numpy().astype(np.float16)
    ort_inputs = {ort_session.get_inputs()[0].name: input_values}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_hidden_states = ort_outputs[0]

    # Compare outputs (convert to float32 for comparison)
    torch_numpy = torch_hidden_states.float().numpy()
    onnx_float32 = onnx_hidden_states.astype(np.float32)
    mean_diff = np.abs(torch_numpy - onnx_float32).mean()
    max_diff = np.abs(torch_numpy - onnx_float32).max()

    print(f"PyTorch output shape: {torch_numpy.shape}")
    print(f"ONNX output shape: {onnx_hidden_states.shape}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")

    success = mean_diff < 1e-3  # Higher tolerance for half precision
    if success:
        print("✅ PyTorch and ONNX models are numerically equivalent!")
    else:
        print("❌ PyTorch and ONNX models have significant differences!")

    # Test MNN model if path is provided
    if mnn_path and os.path.exists(mnn_path):
        print("\nTesting MNN model...")
        try:
            import MNN
            import MNN.numpy as mnp

            mnn_config = {}
            mnn_config['backend'] = 'CPU'
            mnn_config['thread'] = 12
            mnn_rt = MNN.nn.create_runtime_manager((mnn_config,))

            hubert_mnn = MNN.nn.load_module_from_file(mnn_path,
                                                      ['audio16k'], ['last_hidden_state'], runtime_manager=mnn_rt)
            mnn_output = np.array(hubert_mnn(mnp.array(input_values.astype(np.float32))).read())

            print(f"MNN output shape: {mnn_output.shape}")

            # Compare MNN with PyTorch
            max_diff_mnn_pt = np.max(np.abs(torch_numpy - mnn_output.astype(np.float32)))
            mean_diff_mnn_pt = np.mean(np.abs(torch_numpy - mnn_output.astype(np.float32)))

            print(f"MNN vs PyTorch - Maximum absolute difference: {max_diff_mnn_pt:.6f}")
            print(f"MNN vs PyTorch - Mean absolute difference: {mean_diff_mnn_pt:.6f}")

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
        return success


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Export HuBERT model to ONNX format with automatic testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
        help="Path to the HuBERT model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx/chinese-hubert-base",
        help="Output directory for the exported models"
    )
    parser.add_argument(
        "--test_audio_url",
        type=str,
        default=TEST_AUDIO_URL,
        help="Test audio URL for validation"
    )
    args = parser.parse_args()

    try:
        # Export model
        original_model, onnx_path, mnn_path = export_hubert_to_onnx(
            model_path=args.model_path,
            output_dir=args.output_dir
        )

        # Test equivalence
        test_model_equivalence(
            original_model=original_model,
            onnx_path=onnx_path,
            mnn_path=mnn_path,
            test_audio_url=args.test_audio_url
        )

        print("\n✨ All done! Your half-precision ONNX and MNN models are ready to use.")
        return 0

    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        return 1


if __name__ == "__main__":
    main()