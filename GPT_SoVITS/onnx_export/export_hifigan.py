import sys, os
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import argparse
import subprocess
from io import BytesIO

@contextmanager
def temp_sys_path(path):
    old_sys_path = list(sys.path)
    sys.path.insert(0, path)  # insert(0, ...) 保证优先级最高
    try:
        yield
    finally:
        sys.path = old_sys_path

def denorm_spec(x: torch.Tensor) -> torch.Tensor:
    spec_min = -12
    spec_max = 2
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


class HifiGANVocoder(nn.Module):
    def __init__(self, model_path: str = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth"):
        super().__init__()
        with temp_sys_path(os.path.join(os.path.dirname(__file__), '..' )):
            from module.models_onnx import Generator
            self.hifigan = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
        self.hifigan.eval()
        self.hifigan.remove_weight_norm()
        self.hifigan.half()
        state_dict_g = torch.load(model_path, map_location="cpu")
        print(f"Loading vocoder from {model_path}", self.hifigan.load_state_dict(state_dict_g))

    def forward(self, x):
        x = denorm_spec(x)
        return self.hifigan(x)


def export_hifigan_to_onnx(
    model_path: str = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth",
    output_dir: str = "onnx/hifigan"
):
    """Export HifiGAN vocoder to ONNX format"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    onnx_path = os.path.join(output_dir, "sovits_hifigan.onnx")
    mnn_path = os.path.join(output_dir, "sovits_hifigan.mnn")

    # Load model
    model = HifiGANVocoder(model_path)

    # Create dummy input
    cmf_res_rand = torch.randn(1, 100, 32).to(torch.float16)

    # Export to ONNX
    print(f"Exporting HifiGAN to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (cmf_res_rand,),
        onnx_path,
        input_names=["cfm_res"],
        output_names=["audio48k"],
        dynamic_axes={
            "cfm_res": {2: "cfm_length"},
            "audio48k": {2: "audio_length"},
        },
        opset_version=17,
        verbose=False,
    )

    print(f"HifiGAN model exported successfully to: {onnx_path}")

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


def test_model_equivalence(original_model, onnx_path: str, mnn_path: str = None, num_tests: int = 5):
    """Test if the original PyTorch model, ONNX model, and MNN model produce similar outputs"""

    print("Testing HifiGAN model equivalence...")

    success = True

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")

        # Generate random input
        batch_size = 1
        seq_len = np.random.randint(16, 64)  # Variable length
        cmf_res_rand = torch.randn(batch_size, 100, seq_len).to(torch.float16)

        # Get PyTorch output
        print("Running PyTorch inference...")
        original_model.eval()
        with torch.no_grad():
            torch_output = original_model(cmf_res_rand)

        # Get ONNX output
        print("Running ONNX inference...")
        ort_session = ort.InferenceSession(onnx_path)
        input_values = cmf_res_rand.numpy().astype(np.float16)
        ort_inputs = {ort_session.get_inputs()[0].name: input_values}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = ort_outputs[0]

        # Compare outputs (convert to float32 for comparison)
        torch_numpy = torch_output.float().numpy()
        onnx_float32 = onnx_output.astype(np.float32)
        mean_diff = np.abs(torch_numpy - onnx_float32).mean()
        max_diff = np.abs(torch_numpy - onnx_float32).max()

        print(f"PyTorch output shape: {torch_numpy.shape}")
        print(f"ONNX output shape: {onnx_output.shape}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")

        # Success criteria for this test
        test_success = mean_diff < 1e-3  # Higher tolerance for half precision
        if test_success:
            print("✅ PyTorch and ONNX outputs are numerically equivalent!")
        else:
            print("❌ PyTorch and ONNX outputs have significant differences!")
            success = False

        # Test MNN model if path is provided
        if mnn_path and os.path.exists(mnn_path):
            print("Testing MNN model...")
            try:
                import MNN
                import MNN.numpy as mnp

                mnn_config = {}
                mnn_config['backend'] = 'CPU'
                mnn_config['thread'] = 12
                mnn_rt = MNN.nn.create_runtime_manager((mnn_config,))

                hifigan_mnn = MNN.nn.load_module_from_file(
                    mnn_path,
                    ['cfm_res'], ['audio48k'],
                    runtime_manager=mnn_rt
                )
                mnn_output = np.array(hifigan_mnn(mnp.array(input_values.astype(np.float32))).read())

                print(f"MNN output shape: {mnn_output.shape}")

                # Compare MNN with PyTorch
                max_diff_mnn_pt = np.max(np.abs(torch_numpy - mnn_output.astype(np.float32)))
                mean_diff_mnn_pt = np.mean(np.abs(torch_numpy - mnn_output.astype(np.float32)))

                print(f"MNN vs PyTorch - Maximum absolute difference: {max_diff_mnn_pt:.6f}")
                print(f"MNN vs PyTorch - Mean absolute difference: {mean_diff_mnn_pt:.6f}")

                if mean_diff_mnn_pt < 1e-2:
                    print("✅ MNN and PyTorch models are numerically equivalent!")
                else:
                    print("❌ MNN and PyTorch models have significant differences!")
                    success = False

            except Exception as e:
                print(f"❌ Error testing MNN model: {e}")
                success = False
        else:
            print("MNN model path not provided or file does not exist, skipping MNN test")

    return success


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Export HifiGAN vocoder to ONNX format with automatic testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth",
        help="Path to the HifiGAN vocoder model file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx/hifigan",
        help="Output directory for the exported models"
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=1,
        help="Number of random tests to perform for model equivalence"
    )
    args = parser.parse_args()

    try:
        # Export model
        original_model, onnx_path, mnn_path = export_hifigan_to_onnx(
            model_path=args.model_path,
            output_dir=args.output_dir
        )

        # Test equivalence
        success = test_model_equivalence(
            original_model=original_model,
            onnx_path=onnx_path,
            mnn_path=mnn_path,
            num_tests=args.num_tests
        )

        if success:
            print("\n✨ All tests passed! Your ONNX and MNN models are ready to use.")
            return 0
        else:
            print("\n⚠️  Some tests failed. Please check the outputs above.")
            return 1

    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        return 1


if __name__ == "__main__":
    exit(main())