import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Optional, Tuple
import json
import bsdiff4
import onnx
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxsim import simplify
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from export_sovits_v1v2 import export_sovits_v1v2_to_onnx
from genie_t2s_converter.Converter import convert_t2s_only
from t2s_quantization import quantize_t2s, t2s_sdec_fp16_dynamic_quant
from preprocess_utils import preprocess_text, audio_preprocess
from quantization_utils import find_node_by_op_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_COMPRESSION_LEVEL = 9
DEFAULT_MNN_OPTIMIZE_LEVEL = "1"
DEFAULT_MNN_OPTIMIZE_PREFER = "2"
DEFAULT_MNN_WEIGHT_BITS = "8"
DEFAULT_MNN_WEIGHT_BLOCK = "32"

class ExportConfig:
    """Configuration class for export parameters"""

    def __init__(self, sovits_path: str, t2s_ckpt_path: str, ref_voice: str,
                 ref_text: str, project_name: str, output_dir: str,
                 version: str, quantize: bool):
        self.sovits_path = sovits_path
        self.t2s_ckpt_path = t2s_ckpt_path
        self.ref_voice = ref_voice
        self.ref_text = ref_text
        self.project_name = project_name
        self.output_dir = output_dir
        self.version = version
        self.quantize = quantize
        self.is_v2p = version.lower() in ['v2pro', 'v2proplus']

    def validate(self) -> bool:
        """Validate configuration parameters"""
        if not os.path.exists(self.sovits_path):
            logger.error(f"❌ SoVITS model file not found: {self.sovits_path}")
            return False

        if not os.path.exists(self.t2s_ckpt_path):
            logger.error(f"❌ T2S .ckpt file not found: {self.t2s_ckpt_path}")
            return False

        if not self.ref_voice or not os.path.isfile(self.ref_voice):
            logger.error("❌ Reference voice file is required")
            return False

        if not self.ref_text or len(self.ref_text.strip()) == 0:
            logger.error("❌ Reference text is required")
            return False

        return True


def make_archive_with_compression(zip_filepath: str, root_dir: str,
                                compression_level: int = DEFAULT_COMPRESSION_LEVEL) -> None:
    """
    Custom archive function that supports compression level

    Args:
        zip_filepath: Path to the output zip file
        root_dir: Root directory to compress
        compression_level: Compression level (0-9)
    """
    with zipfile.ZipFile(zip_filepath, 'w',
                        compression=zipfile.ZIP_DEFLATED,
                        compresslevel=compression_level) as zipf:

        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, arcname)


def create_output_directories(base_dir: str, project_name: str) -> Tuple[str, str, str]:
    """
    Create output directories for the export pipeline

    Args:
        base_dir: Base output directory
        project_name: Name of the project

    Returns:
        Tuple of (tmp_dir, sovits_output_dir, t2s_output_dir)
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"export_{project_name}_")
    logger.info(f"Temporary working directory: {tmp_dir}")

    sovits_output_dir = os.path.join(tmp_dir, "sovits")
    t2s_output_dir = os.path.join(tmp_dir, "t2s")

    return tmp_dir, sovits_output_dir, t2s_output_dir


def export_sovits_model(sovits_path: str, sovits_output_dir: str, version: str) -> Optional[str]:
    """
    Export SoVITS v1v2 model

    Args:
        sovits_path: Path to SoVITS model file
        sovits_output_dir: Output directory for SoVITS model
        version: SoVITS version

    Returns:
        Path to MNN model file if successful, None otherwise
    """
    logger.info("=> Step 1: Exporting SoVITS v1v2 model...")

    try:
        _, onnx_path, mnn_path = export_sovits_v1v2_to_onnx(
            vits_path=sovits_path,
            output_dir=sovits_output_dir,
            version=version
        )

        # Remove the ONNX file if it exists
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            logger.info(f"   - Removed ONNX file: {onnx_path}")

        logger.info(f"✅ SoVITS v1v2 export completed successfully")
        if mnn_path:
            logger.info(f"   - MNN model: {mnn_path}")

        return mnn_path

    except Exception as e:
        logger.error(f"❌ SoVITS v1v2 export failed: {e}")
        raise


def export_t2s_model(t2s_ckpt_path: str, sovits_path: str, t2s_output_dir: str) -> None:
    """
    Export T2S model

    Args:
        t2s_ckpt_path: Path to T2S checkpoint file
        sovits_path: Path to SoVITS model file (used as t2s_pth_path)
        t2s_output_dir: Output directory for T2S model
    """
    logger.info("=> Step 2: Exporting T2S model...")

    try:
        convert_t2s_only(
            torch_ckpt_path=t2s_ckpt_path,
            torch_pth_path=sovits_path,
            output_dir=t2s_output_dir
        )
        logger.info("✅ T2S export completed successfully")
        logger.info(f"   - Output directory: {t2s_output_dir}")

    except Exception as e:
        logger.error(f"❌ T2S export failed: {e}")
        raise


def combine_t2s_models(t2s_output_dir: str) -> None:
    """
    Combine T2S encoder and first stage decoder models

    Args:
        t2s_output_dir: Directory containing T2S models
    """
    logger.info("=> Step 3: Combining T2S encoder and first stage decoder...")

    encoder = onnx.load(f'{t2s_output_dir}/t2s_encoder_fp32.onnx')
    fsdec = onnx.load(f'{t2s_output_dir}/t2s_first_stage_decoder_fp32.onnx')

    encoder = onnx.compose.add_prefix(encoder, 'encoder_')
    new_fsdec = onnx.compose.merge_models(
        encoder,
        fsdec,
        io_map=[("encoder_x", "x"), ("encoder_prompts", "prompts")],
    )

    # Fix opset_import
    new_imports = {}
    for imp in new_fsdec.opset_import:
        dom = imp.domain
        ver = imp.version if imp.version > 0 else 20
        if dom in new_imports:
            new_imports[dom] = max(new_imports[dom], ver)
        else:
            new_imports[dom] = ver
    del new_fsdec.opset_import[:]
    for dom, ver in new_imports.items():
        imp = new_fsdec.opset_import.add()
        imp.domain = dom
        imp.version = ver

    new_fsdec, check = simplify(new_fsdec)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")

    onnx.save(new_fsdec, f"{t2s_output_dir}/t2s_fsdec.onnx")

    # Clean up temporary files
    os.remove(f'{t2s_output_dir}/t2s_encoder_fp32.onnx')
    os.remove(f'{t2s_output_dir}/t2s_first_stage_decoder_fp32.onnx')
    os.remove(f'{t2s_output_dir}/t2s_encoder_fp32.bin')

    quant_pre_process(f"{t2s_output_dir}/t2s_fsdec.onnx",
                     f"{t2s_output_dir}/t2s_fsdec.onnx",
                     skip_symbolic_shape=True)

    # Process stage decoder
    sdec = onnx.load(f'{t2s_output_dir}/t2s_stage_decoder_fp32.onnx')
    sdec, check = simplify(sdec)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")

    onnx.save(sdec, f"{t2s_output_dir}/t2s_sdec.onnx")

    # Clean up temporary files
    os.remove(f'{t2s_output_dir}/t2s_stage_decoder_fp32.onnx')
    os.remove(f'{t2s_output_dir}/t2s_shared_fp32.bin')

    quant_pre_process(f"{t2s_output_dir}/t2s_sdec.onnx",
                     f"{t2s_output_dir}/t2s_sdec.onnx",
                     skip_symbolic_shape=True)


def export_mnn_models(t2s_output_dir: str) -> None:
    """
    Export T2S models to MNN format with quantization

    Args:
        t2s_output_dir: Directory containing T2S models
    """
    mnn_command = [
        "mnnconvert",
        "--f", "ONNX",
        "--modelFile", f"{t2s_output_dir}/t2s_fsdec.onnx",
        "--optimizeLevel", DEFAULT_MNN_OPTIMIZE_LEVEL,
        "--optimizePrefer", DEFAULT_MNN_OPTIMIZE_PREFER,
        "--MNNModel", f"{t2s_output_dir}/t2s_fsdec.mnn",
        "--weightQuantBits", DEFAULT_MNN_WEIGHT_BITS,
        "--weightQuantBlock", DEFAULT_MNN_WEIGHT_BLOCK
    ]

    try:
        subprocess.run(mnn_command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully exported to MNN: {t2s_output_dir}/t2s_fsdec.mnn")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error exporting to MNN: {e}")
        logger.info(f"stdout: {e.stdout}")
        logger.info(f"stderr: {e.stderr}")

    # Apply dynamic quantization to stage decoder
    t2s_sdec_fp16_dynamic_quant(f"{t2s_output_dir}/t2s_sdec.onnx",
                               f"{t2s_output_dir}/t2s_sdec.onnx")
    os.remove(f"{t2s_output_dir}/t2s_fsdec.onnx")


def quantize_t2s_models(t2s_output_dir: str, ref_text: str, ref_voice: str) -> None:
    """
    Quantize T2S models for mobile inference

    Args:
        t2s_output_dir: Directory containing T2S models
        ref_text: Reference text for calibration
        ref_voice: Reference voice path for calibration
    """
    logger.info("=> Step 4: Quantizing models for mobile inference...")

    quantize_t2s(
        fsdec_path=f"{t2s_output_dir}/t2s_fsdec.onnx",
        fsdec_quant_path=f"{t2s_output_dir}/t2s_fsdec_quant.onnx",
        sdec_path=f"{t2s_output_dir}/t2s_sdec.onnx",
        sdec_quant_path=f"{t2s_output_dir}/t2s_sdec_quant.onnx",
        ref_text=ref_text,
        ref_audio_path=ref_voice
    )

    # Clean up and create diff file
    os.remove(f"{t2s_output_dir}/t2s_fsdec.onnx")
    os.remove(f"{t2s_output_dir}/t2s_sdec.onnx")
    bsdiff4.file_diff(f"{t2s_output_dir}/t2s_sdec_quant.onnx",
                     f"{t2s_output_dir}/t2s_fsdec_quant.onnx",
                     f"{t2s_output_dir}/t2s_fsdec_quant.diff4")
    os.remove(f"{t2s_output_dir}/t2s_fsdec_quant.onnx")


def create_configuration_file(config: ExportConfig, tmp_dir: str) -> None:
    """
    Create configuration JSON file

    Args:
        config: Export configuration
        tmp_dir: Temporary directory path
    """
    config_json = {
        "project_name": config.project_name,
        "type": "GPTSoVITS",
        "version": config.version,
        "bert_base_path": 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
        "cnhuhbert_base_path": 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
        "t2s_weights_path": config.t2s_ckpt_path,
        "vits_weights_path": config.sovits_path,
        "quantized": config.quantize,
    }

    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, ensure_ascii=False, indent=4)

    logger.info(f"Configuration file created: {config_path}")


def create_reference_files(config: ExportConfig, tmp_dir: str) -> None:
    """
    Create reference files for the project

    Args:
        config: Export configuration
        tmp_dir: Temporary directory path
    """
    ref_dir = os.path.join(tmp_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    # Process audio reference
    audio_ssl_feature, spectrum, sv_emb = audio_preprocess(config.ref_voice)
    np.save(os.path.join(ref_dir, "ref_ssl_content.npy"), audio_ssl_feature)
    np.save(os.path.join(ref_dir, "ref_spectrum.npy"), spectrum)
    if config.is_v2p:
        np.save(os.path.join(ref_dir, "ref_sv_emb.npy"), sv_emb)

    # Process text reference
    ref_text_seq, ref_text_bert = preprocess_text(config.ref_text)
    np.save(os.path.join(ref_dir, "ref_text_seq.npy"), ref_text_seq)
    np.save(os.path.join(ref_dir, "ref_text_bert.npy"), ref_text_bert)

    logger.info("Reference files created successfully")


def export_complete_v1v2_pipeline(config: ExportConfig) -> None:
    """
    Complete v1v2 export pipeline that exports both SoVITS and T2S models

    Args:
        config: Export configuration containing all necessary parameters
    """
    logger.info("🚀 Starting complete v1v2 export pipeline...")

    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid configuration provided")

    # Create output directories
    tmp_dir, sovits_output_dir, t2s_output_dir = create_output_directories(
        config.output_dir, config.project_name
    )

    try:
        # Step 1: Export SoVITS v1v2 model
        export_sovits_model(config.sovits_path, sovits_output_dir, config.version)

        # Step 2: Export T2S model
        export_t2s_model(config.t2s_ckpt_path, config.sovits_path, t2s_output_dir)

        # Step 3: Combine T2S models
        combine_t2s_models(t2s_output_dir)

        # Step 4: Handle quantization or MNN export
        if not config.quantize:
            logger.info("=> Skipping quantization as per user request")
            export_mnn_models(t2s_output_dir)
            logger.info("=> Export pipeline completed without quantization")
        else:
            quantize_t2s_models(t2s_output_dir, config.ref_text, config.ref_voice)

        # Step 5: Create configuration and reference files
        logger.info("=> Step 5: Exporting Configuration and Reference File...")
        create_configuration_file(config, tmp_dir)
        create_reference_files(config, tmp_dir)

        # Step 6: Compress the output directory
        logger.info("=> Step 6: Compress the output directory...")
        output_zip_path = os.path.join(config.output_dir, config.project_name + ".gsv")
        make_archive_with_compression(
            zip_filepath=output_zip_path,
            root_dir=tmp_dir,
            compression_level=DEFAULT_COMPRESSION_LEVEL
        )

        logger.info("✅ Complete v1v2 export pipeline finished successfully!")
        logger.info(f"=> All models exported to: {os.path.abspath(config.output_dir)}")
        logger.info(f"   - Output archive: {output_zip_path}")
        logger.info(f"   - SoVITS models: {sovits_output_dir}")
        logger.info(f"   - T2S models: {t2s_output_dir}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Temporary directory cleaned up: {tmp_dir}")



def parse_ref_text(ref_text_arg: str) -> str:
    """
    Parse reference text from argument (direct text or file path)

    Args:
        ref_text_arg: Reference text argument (text or file path)

    Returns:
        Parsed reference text string

    Raises:
        ValueError: If reference text is invalid
    """
    if ref_text_arg is None:
        raise ValueError("Reference text is required")

    # Check if it's a file path
    if os.path.exists(ref_text_arg):
        try:
            with open(ref_text_arg, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            if not ref_text:
                raise ValueError("Reference text file is empty")
            return ref_text
        except Exception as e:
            raise ValueError(f"Error reading reference text file: {e}")

    # Treat as direct text
    ref_text = ref_text_arg.strip()
    if not ref_text:
        raise ValueError("Reference text cannot be empty")

    return ref_text


def parse_arguments() -> ExportConfig:
    """
    Parse command line arguments and create export configuration

    Returns:
        ExportConfig object with parsed parameters

    Raises:
        ValueError: If required arguments are missing
    """
    parser = argparse.ArgumentParser(
        description="Complete v1v2 export pipeline - Export both SoVITS v1v2 and T2S models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--sovits_path",
        type=str,
        required=True,
        help="Path to the SoVITS v1v2 model file (.pth) - also used as t2s_pth_path"
    )

    parser.add_argument(
        "--t2s_ckpt_path",
        type=str,
        required=True,
        help="Path to the T2S model file (.ckpt)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for all exported models"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["v1", "v2", "v2Pro", "v2ProPlus"],
        help="SoVITS model version"
    )

    parser.add_argument(
        "--ref_voice",
        type=str,
        required=True,
        help="Path to the voice file used for quantization calibration (e.g., wav file)"
    )

    parser.add_argument(
        "--ref_text",
        type=str,
        required=True,
        help="Text of ref voice, or Path to the text used for quantization calibration (.txt file, utf-8 encoded)"
    )

    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the project"
    )

    # Optional arguments
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to quantize the models to accelerate mobile inference, may slightly reduce quality"
    )

    args = parser.parse_args()

    # Parse reference text
    try:
        ref_text = parse_ref_text(args.ref_text)
    except ValueError as e:
        logger.error(f"❌ {e}")
        raise ValueError(f"Invalid reference text: {e}")

    # Validate project name
    project_name = args.project_name.strip() if args.project_name.strip() else os.path.basename(args.output_dir.strip('/\\'))

    return ExportConfig(
        sovits_path=args.sovits_path,
        t2s_ckpt_path=args.t2s_ckpt_path,
        ref_voice=args.ref_voice,
        ref_text=ref_text,
        project_name=project_name,
        output_dir=args.output_dir,
        version=args.version,
        quantize=args.quantize
    )


def main() -> int:
    """
    Main execution function

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments and create configuration
        config = parse_arguments()

        # Run export pipeline
        export_complete_v1v2_pipeline(config)

        logger.info("✅ Export pipeline completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("❌ Export pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Export pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())