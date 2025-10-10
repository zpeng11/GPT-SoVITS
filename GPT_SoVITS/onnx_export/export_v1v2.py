import argparse
import logging
import os
import subprocess
import sys
from typing import Optional
import json
import bsdiff4
import onnx
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxsim import simplify
import numpy as np
import shutil,zipfile,tempfile

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from export_sovits_v1v2 import export_sovits_v1v2_to_onnx
from genie_t2s_converter.Converter import convert_t2s_only
from t2s_quantization import quantize_t2s
from preprocess_utils import preprocess_text, audio_preprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def make_archive_with_compression_at_tmp(base_name, format, root_dir, compression_level=9):
    """
    Custom archive function that supports compression level
    """
    if format != 'zip':
        # Fall back to shutil for non-zip formats
        return shutil.make_archive(base_name, format, root_dir)
    temp_dir = tempfile.gettempdir()
    zip_filename = os.path.join(temp_dir, base_name + '.zip')

    with zipfile.ZipFile(zip_filename, 'w', 
                        compression=zipfile.ZIP_DEFLATED,
                        compresslevel=compression_level) as zipf:
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, arcname)
    return zip_filename

def _create_output_directories(output_dir: str) -> tuple:
    """Create output directories for SoVITS and T2S models"""
    sovits_output_dir = os.path.join(output_dir, "sovits")
    t2s_output_dir = os.path.join(output_dir, "t2s")

    os.makedirs(sovits_output_dir, exist_ok=True)
    os.makedirs(t2s_output_dir, exist_ok=True)

    logger.info(f"Created output directories:")
    logger.info(f"  - SoVITS: {sovits_output_dir}")
    logger.info(f"  - T2S: {t2s_output_dir}")

    return sovits_output_dir, t2s_output_dir

def export_complete_v1v2_pipeline(
    sovits_path: str,
    t2s_ckpt_path: str,
    ref_voice: str,
    ref_text: str,
    project_name: str,
    output_dir: str,
    version: str,
    quantize: bool,
):
    """
    Complete v1v2 export pipeline that exports both SoVITS and T2S models

    Args:
        sovits_path: Path to the SoVITS v1v2 model (.pth file) - also used as t2s_pth_path
        t2s_ckpt_path: Path to the T2S model (.ckpt file)
        output_dir: Output directory for all exported models
        version: SoVITS model version
        quantize: Whether to quantize models for mobile inference
        quantize_calibration_voice: Path to voice file for quantization calibration
    """

    logger.info("🚀 Starting complete v1v2 export pipeline...")

    # Create output directories
    sovits_output_dir, t2s_output_dir = _create_output_directories(output_dir)

    # Step 1: Export SoVITS v1v2 model
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
        logger.info(f"   - ONNX model: {onnx_path}")
        if mnn_path:
            logger.info(f"   - MNN model: {mnn_path}")

    except Exception as e:
        logger.error(f"L SoVITS v1v2 export failed: {e}")
        raise

    # Step 2: Export T2S model (using sovits_path as t2s_pth_path)
    logger.info("=> Step 2: Exporting T2S model...")
    try:
        convert_t2s_only(
            torch_ckpt_path=t2s_ckpt_path,
            torch_pth_path=sovits_path,  # Use sovits_path as t2s_pth_path
            output_dir=t2s_output_dir
        )
        logger.info("✅ T2S export completed successfully")
        logger.info(f"   - Output directory: {t2s_output_dir}")

    except Exception as e:
        logger.error(f"❌ T2S export failed: {e}")
        raise

    logger.info("✅ Complete v1v2 export pipeline finished successfully!")
    logger.info(f"=> All models exported to: {os.path.abspath(output_dir)}")
    logger.info(f"   - SoVITS models: {sovits_output_dir}")
    logger.info(f"   - T2S models: {t2s_output_dir}")

    # step 3 combine t2s encoder and t2s fsdecoder , get ready for quantization
    logger.info("=> Step 3: Combining T2S encoder and first stage decoder...")

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
        logger.info("=> Skipping quantization as per user request")
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
        logger.info("=> Export pipeline completed without quantization")
    else:
        logger.info("=> Step 4: Quantizing models for mobile inference...")
        quantize_t2s(
            fsdec_path=f"{t2s_output_dir}/t2s_fsdec.onnx",
            fsdec_quant_path=f"{t2s_output_dir}/t2s_fsdec_quant.onnx",
            sdec_path=f"{t2s_output_dir}/t2s_sdec.onnx",
            sdec_quant_path=f"{t2s_output_dir}/t2s_sdec_quant.onnx",
            ref_text=ref_text,
            ref_audio_path=ref_voice
        )
        os.remove(f"{t2s_output_dir}/t2s_fsdec.onnx")
        os.remove(f"{t2s_output_dir}/t2s_sdec.onnx")
        bsdiff4.file_diff(f"{t2s_output_dir}/t2s_sdec_quant.onnx", f"{t2s_output_dir}/t2s_fsdec_quant.onnx", f"{t2s_output_dir}/t2s_fsdec_quant.diff4")
        os.remove(f"{t2s_output_dir}/t2s_fsdec_quant.onnx")
    
    logger.info("=> Step 5: Exporting Configuration and Reference File...")

    configJson = {
        "project_name": project_name,
        "type": "GPTSoVITS",
        "version" : version,
        "bert_base_path": 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
        "cnhuhbert_base_path": 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
        "t2s_weights_path": t2s_ckpt_path,
        "vits_weights_path": sovits_path,
        "quantized": quantize,
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(configJson, f, ensure_ascii=False, indent=4)

    ref_dir = os.path.join(output_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    audio_ssl_feature, spectrum, sv_emb = audio_preprocess(ref_voice)
    np.save(os.path.join(ref_dir, "ref_ssl_content.npy"), audio_ssl_feature)
    np.save(os.path.join(ref_dir, "ref_spectrum.npy"), spectrum)
    np.save(os.path.join(ref_dir, "ref_sv_emb.npy"), sv_emb)
    
    ref_text_seq, ref_text_bert = preprocess_text(ref_text)
    np.save(os.path.join(ref_dir, "ref_text_seq.npy"), ref_text_seq)
    np.save(os.path.join(ref_dir, "ref_text_bert.npy"), ref_text_bert)

    logger.info("=> Step 6: Compress the output directory...")
    tmp_zip = make_archive_with_compression_at_tmp(
        base_name=project_name,
        format='zip',
        root_dir=output_dir,
        compression_level=9
    )
    # Remove with error handling for read-only files
    def remove_readonly(func, path, _):
        """Clear the readonly bit and reattempt the removal"""
        os.chmod(path, os.stat.S_IWRITE)
        func(path)

    shutil.rmtree(os.path.join(output_dir, 'reference'), onerror=remove_readonly)
    shutil.rmtree(os.path.join(output_dir, 't2s'), onerror=remove_readonly)
    shutil.rmtree(os.path.join(output_dir, 'sovits'), onerror=remove_readonly)
    os.remove(os.path.join(output_dir, 'config.json'))

    shutil.move(tmp_zip, os.path.join(output_dir, os.path.splitext(os.path.basename(tmp_zip))[0]+'.gsv'))



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
        "--quantize",
        action="store_true",
        help="Whether to quantize the models to accelerate mobile inference, may slightly reduce quality"
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="Name of the project"
    )

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.sovits_path):
        logger.error(f"❌ SoVITS model file not found: {args.sovits_path}")
        return 1

    if not os.path.exists(args.t2s_ckpt_path):
        logger.error(f"❌ T2S .ckpt file not found: {args.t2s_ckpt_path}")
        return 1

    ref_text = None
    if args.ref_text is not None and not os.path.exists(args.ref_text) and len(args.ref_text.strip()) > 0:
        ref_text = args.ref_text
    elif args.ref_text is not None and os.path.exists(args.ref_text):
        with open(args.ref_text, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
    else:
        logger.error(f"❌ Reference text is invalid: {args.ref_text}")
        return 1

    if not args.ref_voice or not os.path.isfile(args.ref_voice):
        logger.error("❌ Quantization requires a reference voice file (e.g., wav file)")
        return 1

    if args.project_name is not None and len(args.project_name.strip()) > 0:
        project_name = args.project_name.strip()
    else:
        project_name = os.path.basename(args.output_dir.strip('/\\'))

    try:
        export_complete_v1v2_pipeline(
            sovits_path=args.sovits_path,
            t2s_ckpt_path=args.t2s_ckpt_path,
            ref_voice=args.ref_voice,
            ref_text=ref_text,
            project_name=project_name,
            output_dir=args.output_dir,
            version=args.version,
            quantize=args.quantize,

        )
        logger.info("✅ Export pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Export pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())