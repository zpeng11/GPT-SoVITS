import torch
import torch.nn as nn
from io import BytesIO
import sys, os
from contextlib import contextmanager
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import subprocess
import argparse

@contextmanager
def temp_sys_path(path):
    old_sys_path = list(sys.path)
    sys.path.insert(0, path)  # insert(0, ...) 保证优先级最高
    try:
        yield
    finally:
        sys.path = old_sys_path

def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class VitsV1V2Model(nn.Module):
    def __init__(self, vits_path, version:str = 'v2'):
        super().__init__()
        dict_s2 = load_sovits_new(vits_path)
        self.hps = dict_s2["config"]
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = version

        self.is_v2p = version.lower() in ['v2pro', 'v2proplus']

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        with temp_sys_path(os.path.join(os.path.dirname(__file__), '..' )):
            from module.models_onnx import SynthesizerTrn
            self.vq_model:SynthesizerTrn = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model,
            )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        # self.vq_model.half()
        # print(f"filter_length:{self.hps.data.filter_length} sampling_rate:{self.hps.data.sampling_rate} hop_length:{self.hps.data.hop_length} win_length:{self.hps.data.win_length}")
        #v2 filter_length: 2048 sampling_rate: 32000 hop_length: 640 win_length: 2048
    def forward(self, text_seq, pred_semantic, spectrum, sv_emb):
        if self.is_v2p:
            return self.vq_model(pred_semantic, text_seq, spectrum, sv_emb=sv_emb)[0, 0]
        else:
            return self.vq_model(pred_semantic, text_seq, spectrum)[0, 0]
        

def export_sovits_v1v2_to_onnx(
    vits_path: str = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    output_dir: str = "onnx/sovits_v2",
    version: str = 'v2'
):
    """Export SoVITS v1/v2/v2p/v2pp to ONNX format"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    sovits_onnx_path = os.path.join(output_dir, "sovits_v1v2.onnx")
    sovits_mnn_path = os.path.join(output_dir, "sovits_v1v2.mnn")

    # Load model
    model = VitsV1V2Model(vits_path, version=version)

    # Create dummy input
    text_seq_rand = torch.randint(0, 100, (1, 20), dtype=torch.int64)
    pred_semantic_rand = torch.randint(1, 100, (1, 1, 30)).to(torch.int64)
    spectrum_rand = torch.randn(1, 1025, 30).to(torch.float32)
    sv_emb_rand = torch.randn(1, 20480).to(torch.float32)

    # Export to ONNX
    print(f"Exporting SoVITS v1/v2/v2p/v2pp to ONNX: {sovits_onnx_path}")
    torch.onnx.export(
        model,
        (text_seq_rand, pred_semantic_rand, spectrum_rand, sv_emb_rand),
        sovits_onnx_path,
        input_names=["input_text_phones", "pred_semantic", "spectrum", "sv_emb"],
        output_names=["audio32k"],
        dynamic_axes={
            "input_text_phones": {1: "text_length"},
            "pred_semantic": {2: "pred_length"},
            "spectrum": {2: "spectrum_length"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"SoVITS v1/v2/v2p/v2pp model exported successfully to: {sovits_onnx_path}")

    print(f"Simplifying ONNX model: {sovits_onnx_path}")
    try:
        # Load the exported ONNX model
        onnx_model = onnx.load(sovits_onnx_path)

        # Simplify the model
        simplified_model, check = onnxsim.simplify(onnx_model)

        if check:
            # Save the simplified model, replacing the original
            onnx.save(simplified_model, sovits_onnx_path)
            print(f"ONNX model simplified and saved in-place: {sovits_onnx_path}")
        else:
            print("ONNX simplification check failed, keeping original model")
    except Exception as e:
        print(f"Error during ONNX simplification: {e}")
        print("Keeping original ONNX model without simplification")

    # Export to MNN format
    print(f"Exporting to MNN: {sovits_mnn_path}")
    mnn_command = [
        "mnnconvert",
        "--f", "ONNX",
        "--modelFile", sovits_onnx_path,
        "--optimizeLevel", "2",
        "--optimizePrefer", "2",
        "--MNNModel", sovits_mnn_path,
        "--weightQuantBits", "8",
        "--fp16"
    ]

    try:
        subprocess.run(mnn_command, check=True, capture_output=True, text=True)
        print(f"Successfully exported to MNN: {sovits_mnn_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting to MNN: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sovits_mnn_path = None

    return model, sovits_onnx_path, sovits_mnn_path


def test_model(original_model, onnx_path: str, mnn_path: str = None):
    """Test if the original PyTorch model, ONNX model, and MNN model produce similar outputs"""

    print("Testing SoVITS v1/v2/v2p/v2pp model equivalence...")

    # Generate random input with variable dimensions
    batch_size = 1
    text_length = 20  # Variable text length
    pred_length = 30  # Variable pred semantic length
    spectrum_length = 30  # Variable spectrum length

    text_seq_rand = torch.randint(0, 100, (batch_size, text_length), dtype=torch.int64)
    pred_semantic_rand = torch.randint(1, 100, (batch_size, 1, pred_length)).to(torch.int64)
    spectrum_rand = torch.randn(batch_size, 1025, spectrum_length).to(torch.float32)
    sv_emb_rand = torch.randn(batch_size, 20480).to(torch.float32)

    # Get PyTorch output
    print("Running PyTorch inference...")
    original_model.eval()
    with torch.no_grad():
        torch_output = original_model(text_seq_rand, pred_semantic_rand, spectrum_rand, sv_emb_rand)

    # Get ONNX output
    print("Running ONNX inference...")
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        ort_session.get_inputs()[0].name: text_seq_rand.numpy(),
        ort_session.get_inputs()[1].name: pred_semantic_rand.numpy(),
        ort_session.get_inputs()[2].name: spectrum_rand.numpy().astype(np.float32),
        ort_session.get_inputs()[3].name: sv_emb_rand.numpy().astype(np.float32),
    }
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

            sovits_mnn = MNN.nn.load_module_from_file(
                mnn_path,
                ["input_text_phones", "pred_semantic", "spectrum", "sv_emb"],
                ['audio32k'],
                runtime_manager=mnn_rt
            )

            # Prepare inputs for MNN (convert to float32 for MNN)
            mnn_inputs = [
                mnp.array(text_seq_rand.numpy().astype(np.int64)),
                mnp.array(pred_semantic_rand.numpy().astype(np.int64)),
                mnp.array(spectrum_rand.numpy().astype(np.float32)),
                mnp.array(sv_emb_rand.numpy().astype(np.float32))
            ]

            mnn_result = sovits_mnn(mnn_inputs)
            mnn_output = np.array(mnn_result[0].read())

            # Compare MNN with PyTorch
            max_diff_mnn_pt = np.max(np.abs(torch_numpy - mnn_output.astype(np.float32)))
            mean_diff_mnn_pt = np.mean(np.abs(torch_numpy - mnn_output.astype(np.float32)))

            print(f"MNN vs PyTorch - Maximum absolute difference: {max_diff_mnn_pt:.6f}")
            print(f"MNN vs PyTorch - Mean absolute difference: {mean_diff_mnn_pt:.6f}")


        except Exception as e:
            print(f"❌ Error testing MNN model: {e}")
    else:
        print("MNN model path not provided or file does not exist, skipping MNN test")

    return True


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Export SoVITS v1/v2/v2p/v2pp model to ONNX format with automatic testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--vits_path",
        type=str,
        default="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        help="Path to the SoVITS v1/v2/v2p/v2pp model file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx/sovits_v1v2",
        help="Output directory for the exported models"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2ProPlus",
        choices=["v1", "v2", "v2Pro", "v2ProPlus"],
        help="SoVITS model version"
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
        original_model, onnx_path, mnn_path = export_sovits_v1v2_to_onnx(
            vits_path=args.vits_path,
            output_dir=args.output_dir,
            version=args.version
        )

        # Test equivalence
        success = test_model(
            original_model=original_model,
            onnx_path=onnx_path,
            mnn_path=mnn_path
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