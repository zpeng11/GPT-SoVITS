import argparse
import subprocess
import logging
import os
import sys
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxsim import simplify
import onnx
import bsdiff4

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from export_sovits_v1v2 import export_sovits_v1v2_to_onnx
from genie_t2s_converter.Converter import convert_t2s_only

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_complete_v1v2_pipeline(
    sovits_path: str,
    t2s_ckpt_path: str,
    output_dir: str = "onnx/export_v1v2",
    version: str = "v2ProPlus",
    quantize: bool = False,
    quantize_calibration_voice: str = ""
):
    """
    Complete v1v2 export pipeline that exports both SoVITS and T2S models

    Args:
        sovits_path: Path to the SoVITS v1v2 model (.pth file) - also used as t2s_pth_path
        t2s_ckpt_path: Path to the T2S model (.ckpt file)
        output_dir: Output directory for all exported models
        version: SoVITS model version
    """

    logger.info("=� Starting complete v1v2 export pipeline...")

    # Create output directories
    sovits_output_dir = os.path.join(output_dir, "sovits")
    t2s_output_dir = os.path.join(output_dir, "t2s")

    os.makedirs(sovits_output_dir, exist_ok=True)
    os.makedirs(t2s_output_dir, exist_ok=True)

    # Step 1: Export SoVITS v1v2 model
    logger.info("=� Step 1: Exporting SoVITS v1v2 model...")
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

        logger.info(f" SoVITS v1v2 export completed successfully")
        logger.info(f"   - ONNX model: {onnx_path}")
        if mnn_path:
            logger.info(f"   - MNN model: {mnn_path}")

    except Exception as e:
        logger.error(f"L SoVITS v1v2 export failed: {e}")
        raise

    # Step 2: Export T2S model (using sovits_path as t2s_pth_path)
    logger.info("=� Step 2: Exporting T2S model...")
    try:
        convert_t2s_only(
            torch_ckpt_path=t2s_ckpt_path,
            torch_pth_path=sovits_path,  # Use sovits_path as t2s_pth_path
            output_dir=t2s_output_dir
        )
        logger.info(f" T2S export completed successfully")
        logger.info(f"   - Output directory: {t2s_output_dir}")

    except Exception as e:
        logger.error(f"L T2S export failed: {e}")
        raise

    logger.info("<� Complete v1v2 export pipeline finished successfully!")
    logger.info(f"=� All models exported to: {os.path.abspath(output_dir)}")
    logger.info(f"   - SoVITS models: {sovits_output_dir}")
    logger.info(f"   - T2S models: {t2s_output_dir}")

    # step 3 combine t2s encoder and t2s fsdecoder , get ready for quantization
    logger.info("=� Step 3: Combining T2S encoder and first stage decoder...")

    encoder = onnx.load(f'{t2s_output_dir}/t2s_encoder_fp32.onnx')
    fsdec = onnx.load(f'{t2s_output_dir}/t2s_first_stage_decoder_fp32.onnx')

    encoder = onnx.compose.add_prefix(encoder, 'encoder_')
    new_fsdec = onnx.compose.merge_models(
        encoder,
        fsdec,
        io_map=[("encoder_x", "x"), ("encoder_prompts", "prompts")],
    )

    # 修复 opset_import
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
    os.remove(f'{t2s_output_dir}/t2s_encoder_fp32.onnx')
    os.remove(f'{t2s_output_dir}/t2s_first_stage_decoder_fp32.onnx')
    os.remove(f'{t2s_output_dir}/t2s_encoder_fp32.bin')
    quant_pre_process(f"{t2s_output_dir}/t2s_fsdec.onnx", f"{t2s_output_dir}/t2s_fsdec.onnx", skip_symbolic_shape=True)

    sdec = onnx.load(f'{t2s_output_dir}/t2s_stage_decoder_fp32.onnx')
    sdec, check = simplify(sdec)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    onnx.save(sdec, f"{t2s_output_dir}/t2s_sdec.onnx")
    os.remove(f'{t2s_output_dir}/t2s_stage_decoder_fp32.onnx')
    os.remove(f'{t2s_output_dir}/t2s_shared_fp32.bin')
    quant_pre_process(f"{t2s_output_dir}/t2s_sdec.onnx", f"{t2s_output_dir}/t2s_sdec.onnx", skip_symbolic_shape=True)

    # Step 4: Optional quantization
    if not quantize:
        logger.info("=� Skipping quantization as per user request")
        def get_command(component_name: str):
            return [
                "mnnconvert",
                "--f", "ONNX",
                "--modelFile", f"{t2s_output_dir}/{component_name}.onnx",
                "--optimizeLevel", "2",
                "--optimizePrefer", "2",
                "--MNNModel", f"{t2s_output_dir}/{component_name}.mnn",
                "--weightQuantBits", "8",
                "--weightQuantBlock", "128"
            ]
        try:
            subprocess.run(get_command("t2s_fsdec"), check=True, capture_output=True, text=True)
            print(f"Successfully exported to MNN: {t2s_output_dir}/t2s_fsdec.mnn")
            subprocess.run(get_command("t2s_sdec"), check=True, capture_output=True, text=True)
            print(f"Successfully exported to MNN: {t2s_output_dir}/t2s_sdec.mnn")
        except subprocess.CalledProcessError as e:
            print(f"Error exporting to MNN: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        os.remove(f"{t2s_output_dir}/t2s_fsdec.onnx")
        os.remove(f"{t2s_output_dir}/t2s_sdec.onnx")

        bsdiff4.file_diff(f"{t2s_output_dir}/t2s_sdec.mnn", f"{t2s_output_dir}/t2s_fsdec.mnn", f"{t2s_output_dir}/t2s_fsdec.diff4")

        os.remove(f"{t2s_output_dir}/t2s_fsdec.mnn")
        logger.info("=� Export pipeline completed without quantization")
        return
    



def main():
    """Main execution function"""
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

    # Optional arguments
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
        "--quantize",
        type=bool,
        default=False,
        help="Whether to quantize the models to accelerate mobile inference, may slightly reduce quality"
    )

    parser.add_argument(
        "--quantize_calibration_voice",
        type=str,
        default="",
        help="Path to the voice file used for quantization calibration (e.g., wav file)"
    )

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.sovits_path):
        logger.error(f"L SoVITS model file not found: {args.sovits_path}")
        return 1

    if not os.path.exists(args.t2s_ckpt_path):
        logger.error(f"L T2S .ckpt file not found: {args.t2s_ckpt_path}")
        return 1

    try:
        export_complete_v1v2_pipeline(
            sovits_path=args.sovits_path,
            t2s_ckpt_path=args.t2s_ckpt_path,
            output_dir=args.output_dir,
            version=args.version,
            quantize=args.quantize,
            quantize_calibration_voice=args.quantize_calibration_voice
        )
        return 0

    except Exception as e:
        logger.error(f"L Export pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())