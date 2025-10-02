import sys, os
sys.path.append(os.path.dirname(__file__))  # 或者上层目录
from T2SConverter import T2SModelConverter
from EncoderConverter import EncoderConverter

import logging
from typing import Optional, Tuple
import os
import shutil
import traceback
import importlib.resources
import contextlib

logger = logging.getLogger()

CACHE_DIR = os.path.join(os.getcwd(), "Cache")
_ENCODER_RESOURCE_PATH = "onnx_export/genie_t2s_converter/data/Models/t2s_encoder_fp32.onnx"
_STAGE_DECODER_RESOURCE_PATH = "onnx_export/genie_t2s_converter/data/Models/t2s_stage_decoder_fp32.onnx"
_FIRST_STAGE_DECODER_RESOURCE_PATH = "onnx_export/genie_t2s_converter/data/Models/t2s_first_stage_decoder_fp32.onnx"
_T2S_KEYS_RESOURCE_PATH = "onnx_export/genie_t2s_converter/data/Keys/t2s_onnx_keys.txt"


def find_ckpt_and_pth(directory: str) -> Tuple[Optional[str], Optional[str]]:
    ckpt_path: Optional[str] = None
    pth_path: Optional[str] = None
    for filename in os.listdir(directory):
        full_path: str = os.path.join(directory, filename)
        if filename.endswith(".ckpt") and ckpt_path is None:
            ckpt_path = full_path
        elif filename.endswith(".pth") and pth_path is None:
            pth_path = full_path
        if ckpt_path and pth_path:
            break
    return ckpt_path, pth_path


def remove_folder(folder: str) -> None:
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logger.info(f"🧹 Folder cleaned: {folder}")
    except Exception as e:
        logger.error(f"❌ Failed to clean folder {folder}: {e}")

def convert_t2s_only(torch_ckpt_path: str,
            torch_pth_path: str,
            output_dir: str):
    # 确保缓存和输出目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f"The output directory {output_dir} is not empty!")

    try:
        with contextlib.ExitStack() as stack:
            files = importlib.resources.files('GPT_SoVITS')

            encoder_onnx_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_ENCODER_RESOURCE_PATH)))
            stage_decoder_path = stack.enter_context(
                importlib.resources.as_file(files.joinpath(_STAGE_DECODER_RESOURCE_PATH)))
            first_stage_decoder_path = stack.enter_context(
                importlib.resources.as_file(files.joinpath(_FIRST_STAGE_DECODER_RESOURCE_PATH)))
            t2s_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_T2S_KEYS_RESOURCE_PATH)))

            converter_1 = T2SModelConverter(
                torch_ckpt_path=torch_ckpt_path,
                stage_decoder_onnx_path=str(stage_decoder_path),
                first_stage_decoder_onnx_path=str(first_stage_decoder_path),
                key_list_file=str(t2s_keys_path),
                output_dir=output_dir,
                cache_dir=CACHE_DIR,
            )
            converter_3 = EncoderConverter(
                ckpt_path=torch_ckpt_path,
                pth_path=torch_pth_path,
                onnx_input_path=str(encoder_onnx_path),
                output_dir=output_dir,
            )

            try:
                converter_1.run_full_process()
                converter_3.convert()
                logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n")
            except Exception:
                logger.error(f"❌ A critical error occurred during the conversion process")
                logger.error(traceback.format_exc())
                remove_folder(output_dir)  # 只在失败时清理输出目录

    finally:
        # 无论成功还是失败，都尝试清理缓存目录
        remove_folder(CACHE_DIR)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert T2S model to ONNX format")
    parser.add_argument("--torch_ckpt_path", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--torch_pth_path", type=str, required=True, help="Path to the .pth file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted ONNX models")
    args = parser.parse_args()

    convert_t2s_only(
        torch_ckpt_path=args.torch_ckpt_path,
        torch_pth_path=args.torch_pth_path,
        output_dir=args.output_dir
    )