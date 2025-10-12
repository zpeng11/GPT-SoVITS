import argparse
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from io import BytesIO
from typing import Optional, Tuple
from io import BytesIO
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MNN_OPTIMIZE_LEVEL = "2"
DEFAULT_MNN_OPTIMIZE_PREFER = "2"
DEFAULT_MNN_WEIGHT_BITS = "8"
DEFAULT_MNN_WEIGHT_BLOCK = "32"
DEFAULT_TEXT_LENGTH = 20
DEFAULT_PRED_LENGTH = 30
DEFAULT_SPECTRUM_LENGTH = 30
DEFAULT_TEXT_VOCAB_SIZE = 100
DEFAULT_SEMANTIC_VOCAB_SIZE = 100
SPECTRUM_SIZE = 1025
SV_EMB_SIZE = 20480

@contextmanager
def temp_sys_path(path: str):
    """
    Context manager for temporarily adding a path to sys.path with highest priority

    Args:
        path: Path to temporarily add to sys.path
    """
    old_sys_path = list(sys.path)
    sys.path.insert(0, path)  # insert(0, ...) ensures highest priority
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
    """
    Dictionary subclass that provides attribute access to dictionary items recursively

    This class allows accessing nested dictionary items as attributes, making
    the code more readable when working with configuration dictionaries.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize recursive attribute dictionary

        Args:
            input_dict: Dictionary to convert to attribute-accessible format
        """
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item: str):
        """
        Get dictionary item as attribute

        Args:
            item: Key to retrieve

        Returns:
            Value associated with the key

        Raises:
            AttributeError: If key is not found
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key: str, value):
        """
        Set dictionary item as attribute

        Args:
            key: Key to set
            value: Value to associate with key
        """
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item: str):
        """
        Delete dictionary item as attribute

        Args:
            item: Key to delete

        Raises:
            AttributeError: If key is not found
        """
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class VitsV1V2Model(nn.Module):
    """
    SoVITS v1/v2/v2Pro/v2ProPlus model wrapper for ONNX export

    This class wraps the SoVITS model and provides a unified interface
    for different model versions while preparing them for ONNX export.
    """

    def __init__(self, vits_path: str, version: str = 'v2'):
        """
        Initialize SoVITS model wrapper

        Args:
            vits_path: Path to the SoVITS model file
            version: Model version ('v1', 'v2', 'v2Pro', 'v2ProPlus')
        """
        super().__init__()

        # Load model data
        dict_s2 = load_sovits_new(vits_path)
        self.hps = dict_s2["config"]

        # Auto-detect version based on text embedding size
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
            logger.info("Auto-detected model version: v1")
        else:
            self.hps["model"]["version"] = version
            logger.info(f"Using specified model version: {version}")

        # Check if this is a v2Pro model
        self.is_v2p = version.lower() in ['v2pro', 'v2proplus']
        if self.is_v2p:
            logger.info("Initializing v2Pro model with speaker embedding support")

        # Convert configuration to attribute-accessible format
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"

        # Load the synthesizer model
        self._load_synthesizer_model(dict_s2["weight"])

    def _load_synthesizer_model(self, weights: dict) -> None:
        """
        Load the synthesizer model with proper configuration

        Args:
            weights: Model weights dictionary
        """
        logger.info("Loading synthesizer model...")

        with temp_sys_path(os.path.join(os.path.dirname(__file__), '..')):
            from module.models_onnx import SynthesizerTrn

            self.vq_model: SynthesizerTrn = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model,
            )

        self.vq_model.eval()
        self.vq_model.load_state_dict(weights, strict=False)

        logger.info(f"Model configuration - filter_length: {self.hps.data.filter_length}, "
                   f"sampling_rate: {self.hps.data.sampling_rate}, "
                   f"hop_length: {self.hps.data.hop_length}, "
                   f"win_length: {self.hps.data.win_length}")

    def forward(self, text_seq: torch.Tensor, pred_semantic: torch.Tensor,
                spectrum: torch.Tensor, sv_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the SoVITS model

        Args:
            text_seq: Text sequence tensor
            pred_semantic: Predicted semantic tensor
            spectrum: Spectrogram tensor
            sv_emb: Speaker embedding tensor (required for v2Pro models)

        Returns:
            Generated audio tensor
        """
        if self.is_v2p:
            if sv_emb is None:
                raise ValueError("Speaker embedding (sv_emb) is required for v2Pro models")
            return self.vq_model(pred_semantic, text_seq, spectrum, sv_emb=sv_emb)[0, 0]
        else:
            return self.vq_model(pred_semantic, text_seq, spectrum)[0, 0]


def create_dummy_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create dummy input tensors for ONNX export

    Returns:
        Tuple of dummy tensors (text_seq, pred_semantic, spectrum, sv_emb)
    """
    logger.info("Creating dummy input tensors for ONNX export")

    text_seq_rand = torch.randint(0, DEFAULT_TEXT_VOCAB_SIZE, (1, DEFAULT_TEXT_LENGTH), dtype=torch.int64)
    pred_semantic_rand = torch.randint(1, DEFAULT_SEMANTIC_VOCAB_SIZE, (1, 1, DEFAULT_PRED_LENGTH)).to(torch.int64)
    spectrum_rand = torch.randn(1, SPECTRUM_SIZE, DEFAULT_SPECTRUM_LENGTH).to(torch.float32)
    sv_emb_rand = torch.randn(1, SV_EMB_SIZE).to(torch.float32)

    return text_seq_rand, pred_semantic_rand, spectrum_rand, sv_emb_rand


def export_to_onnx(model: VitsV1V2Model, output_path: str,
                   dummy_inputs: Tuple[torch.Tensor, ...]) -> None:
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        dummy_inputs: Dummy input tensors for tracing
    """
    logger.info(f"Exporting SoVITS model to ONNX: {output_path}")

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
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

    logger.info(f"SoVITS model exported successfully to: {output_path}")


def simplify_onnx_model(onnx_path: str) -> bool:
    """
    Simplify ONNX model to optimize performance

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        True if simplification was successful, False otherwise
    """
    logger.info(f"Simplifying ONNX model: {onnx_path}")

    try:
        # Load the exported ONNX model
        onnx_model = onnx.load(onnx_path)

        # Simplify the model
        simplified_model, check = onnxsim.simplify(onnx_model)

        if check:
            # Save the simplified model, replacing the original
            onnx.save(simplified_model, onnx_path)
            logger.info(f"ONNX model simplified and saved in-place: {onnx_path}")
            return True
        else:
            logger.warning("ONNX simplification check failed, keeping original model")
            return False

    except Exception as e:
        logger.error(f"Error during ONNX simplification: {e}")
        logger.warning("Keeping original ONNX model without simplification")
        return False


def export_to_mnn(onnx_path: str, mnn_path: str) -> Optional[str]:
    """
    Export ONNX model to MNN format with quantization

    Args:
        onnx_path: Path to input ONNX model
        mnn_path: Path to save MNN model

    Returns:
        Path to MNN model if successful, None otherwise
    """
    logger.info(f"Exporting ONNX to MNN: {onnx_path} -> {mnn_path}")

    mnn_command = [
        "mnnconvert",
        "--f", "ONNX",
        "--modelFile", onnx_path,
        "--optimizeLevel", DEFAULT_MNN_OPTIMIZE_LEVEL,
        "--optimizePrefer", DEFAULT_MNN_OPTIMIZE_PREFER,
        "--MNNModel", mnn_path,
        "--weightQuantBits", DEFAULT_MNN_WEIGHT_BITS,
        "--weightQuantBlock", DEFAULT_MNN_WEIGHT_BLOCK,
    ]

    try:
        subprocess.run(mnn_command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully exported to MNN: {mnn_path}")
        return mnn_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error exporting to MNN: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None


def export_sovits_v1v2_to_onnx(
    vits_path: str = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    output_dir: str = "onnx/sovits_v2",
    version: str = 'v2'
) -> Tuple[VitsV1V2Model, str, Optional[str]]:
    """
    Export SoVITS v1/v2/v2Pro/v2ProPlus model to ONNX and MNN formats

    Args:
        vits_path: Path to the SoVITS model file
        output_dir: Output directory for exported models
        version: SoVITS model version

    Returns:
        Tuple of (model, onnx_path, mnn_path)
    """
    logger.info(f"Starting SoVITS {version} export pipeline")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    sovits_onnx_path = os.path.join(output_dir, "sovits_v1v2.onnx")
    sovits_mnn_path = os.path.join(output_dir, "sovits_v1v2.mnn")

    # Load model
    model = VitsV1V2Model(vits_path, version=version)

    # Create dummy inputs
    dummy_inputs = create_dummy_inputs()

    # Export to ONNX
    export_to_onnx(model, sovits_onnx_path, dummy_inputs)

    # Simplify ONNX model
    simplify_onnx_model(sovits_onnx_path)

    # Export to MNN format
    mnn_path = export_to_mnn(sovits_onnx_path, sovits_mnn_path)

    logger.info("SoVITS export pipeline completed successfully")
    return model, sovits_onnx_path, mnn_path


def create_test_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test input tensors for model validation

    Returns:
        Tuple of test tensors (text_seq, pred_semantic, spectrum, sv_emb)
    """
    logger.info("Creating test input tensors for model validation")

    text_seq_rand = torch.randint(0, DEFAULT_TEXT_VOCAB_SIZE, (1, DEFAULT_TEXT_LENGTH), dtype=torch.int64)
    pred_semantic_rand = torch.randint(1, DEFAULT_SEMANTIC_VOCAB_SIZE, (1, 1, DEFAULT_PRED_LENGTH)).to(torch.int64)
    spectrum_rand = torch.randn(1, SPECTRUM_SIZE, DEFAULT_SPECTRUM_LENGTH).to(torch.float32)
    sv_emb_rand = torch.randn(1, SV_EMB_SIZE).to(torch.float32)

    return text_seq_rand, pred_semantic_rand, spectrum_rand, sv_emb_rand


def run_pytorch_inference(model: VitsV1V2Model, inputs: Tuple[torch.Tensor, ...]) -> np.ndarray:
    """
    Run inference on PyTorch model

    Args:
        model: PyTorch model
        inputs: Input tensors

    Returns:
        Model output as numpy array
    """
    logger.info("Running PyTorch inference...")
    model.eval()
    with torch.no_grad():
        torch_output = model(*inputs)
    return torch_output.float().numpy()


def run_onnx_inference(onnx_path: str, inputs: Tuple[torch.Tensor, ...]) -> np.ndarray:
    """
    Run inference on ONNX model

    Args:
        onnx_path: Path to ONNX model
        inputs: Input tensors

    Returns:
        Model output as numpy array
    """
    logger.info("Running ONNX inference...")
    ort_session = ort.InferenceSession(onnx_path)

    ort_inputs = {
        ort_session.get_inputs()[0].name: inputs[0].numpy(),
        ort_session.get_inputs()[1].name: inputs[1].numpy(),
        ort_session.get_inputs()[2].name: inputs[2].numpy().astype(np.float32),
        ort_session.get_inputs()[3].name: inputs[3].numpy().astype(np.float32),
    }

    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs[0]


def run_mnn_inference(mnn_path: str, inputs: Tuple[torch.Tensor, ...]) -> Optional[np.ndarray]:
    """
    Run inference on MNN model

    Args:
        mnn_path: Path to MNN model
        inputs: Input tensors

    Returns:
        Model output as numpy array if successful, None otherwise
    """
    if not mnn_path or not os.path.exists(mnn_path):
        logger.info("MNN model path not provided or file does not exist, skipping MNN test")
        return None

    logger.info("Running MNN inference...")

    try:
        import MNN
        import MNN.numpy as mnp

        mnn_config = {
            'backend': 'CPU',
            'thread': 12
        }
        mnn_rt = MNN.nn.create_runtime_manager((mnn_config,))

        sovits_mnn = MNN.nn.load_module_from_file(
            mnn_path,
            ["input_text_phones", "pred_semantic", "spectrum", "sv_emb"],
            ['audio32k'],
            runtime_manager=mnn_rt
        )

        # Prepare inputs for MNN
        mnn_inputs = [
            mnp.array(inputs[0].numpy().astype(np.int64)),
            mnp.array(inputs[1].numpy().astype(np.int64)),
            mnp.array(inputs[2].numpy().astype(np.float32)),
            mnp.array(inputs[3].numpy().astype(np.float32))
        ]

        mnn_result = sovits_mnn(mnn_inputs)
        mnn_output = np.array(mnn_result[0].read())

        logger.info("MNN inference completed successfully")
        return mnn_output

    except Exception as e:
        logger.error(f"❌ Error testing MNN model: {e}")
        return None


def compare_outputs(torch_output: np.ndarray, onnx_output: np.ndarray,
                    mnn_output: Optional[np.ndarray] = None) -> bool:
    """
    Compare outputs from different model formats

    Args:
        torch_output: PyTorch model output
        onnx_output: ONNX model output
        mnn_output: MNN model output (optional)

    Returns:
        True if outputs are similar enough, False otherwise
    """
    # Convert to float32 for comparison
    onnx_float32 = onnx_output.astype(np.float32)

    # Compare PyTorch and ONNX outputs
    mean_diff_onnx = np.abs(torch_output - onnx_float32).mean()
    max_diff_onnx = np.abs(torch_output - onnx_float32).max()

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(f"ONNX output shape: {onnx_output.shape}")
    logger.info(f"PyTorch vs ONNX - Mean absolute difference: {mean_diff_onnx:.6f}")
    logger.info(f"PyTorch vs ONNX - Max absolute difference: {max_diff_onnx:.6f}")

    # Check if MNN output is available and compare
    if mnn_output is not None:
        mean_diff_mnn = np.abs(torch_output - mnn_output.astype(np.float32)).mean()
        max_diff_mnn = np.abs(torch_output - mnn_output.astype(np.float32)).max()

        logger.info(f"MNN vs PyTorch - Mean absolute difference: {mean_diff_mnn:.6f}")
        logger.info(f"MNN vs PyTorch - Max absolute difference: {max_diff_mnn:.6f}")

    # Define success criteria
    onnx_success = mean_diff_onnx < 1e-2 and max_diff_onnx < 5e-2
    mnn_success = mnn_output is None or (mean_diff_mnn < 1e-2 and max_diff_mnn < 5e-2)

    return onnx_success and mnn_success


def test_model(original_model: VitsV1V2Model, onnx_path: str,
              mnn_path: Optional[str] = None) -> bool:
    """
    Test if the original PyTorch model, ONNX model, and MNN model produce similar outputs

    Args:
        original_model: Original PyTorch model
        onnx_path: Path to ONNX model
        mnn_path: Path to MNN model (optional)

    Returns:
        True if all models produce similar outputs, False otherwise
    """
    logger.info("Testing SoVITS model equivalence across different formats...")

    # Create test inputs
    test_inputs = create_test_inputs()

    # Run inference on all models
    torch_output = run_pytorch_inference(original_model, test_inputs)
    onnx_output = run_onnx_inference(onnx_path, test_inputs)
    mnn_output = run_mnn_inference(mnn_path, test_inputs)

    # Compare outputs
    success = compare_outputs(torch_output, onnx_output, mnn_output)

    if success:
        logger.info("✅ All models produce equivalent outputs!")
    else:
        logger.warning("⚠️  Some models have significant differences in outputs")

    return success


def main() -> int:
    """
    Main execution function

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Export SoVITS v1/v2/v2Pro/v2ProPlus model to ONNX format with automatic testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vits_path",
        type=str,
        default="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        help="Path to the SoVITS model file"
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
    args = parser.parse_args()

    try:
        # Export model
        logger.info("Starting SoVITS model export pipeline")
        original_model, onnx_path, mnn_path = export_sovits_v1v2_to_onnx(
            vits_path=args.vits_path,
            output_dir=args.output_dir,
            version=args.version
        )

        logger.info("Running model equivalence tests")
        success = test_model(
            original_model=original_model,
            onnx_path=onnx_path,
            mnn_path=mnn_path
        )

        if success:
            logger.info("✅ All tests passed! Your ONNX and MNN models are ready to use.")
        else:
            logger.warning("⚠️  Some tests failed. Please check the outputs above.")
            return 1
        return 0

    except KeyboardInterrupt:
        logger.info("❌ Export pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error during export: {e}")
        return 1


if __name__ == "__main__":
    exit(main())