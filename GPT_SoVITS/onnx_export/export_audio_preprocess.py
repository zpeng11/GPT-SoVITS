import os
import tempfile
import urllib.request
import torch
from torch import nn
from GPT_SoVITS.sv import SV
from transformers import HubertModel, HubertConfig
import numpy as np
import onnxruntime as ort
import subprocess
import argparse
import onnx
import onnxsim


# Constants
TEST_AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/4.flac"
SAMPLE_RATE = 32000
MAX_AUDIO_LENGTH = 160000  # 5 seconds at 32kHz


def download_audio_from_url(url):
    """Download audio file from URL to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.flac') as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        return tmp_file.name


def load_and_preprocess_audio(audio_path):
    """Load and preprocess audio file"""
    import torchaudio

    # Handle URL by downloading first
    if audio_path.startswith('http'):
        audio_path = download_audio_from_url(audio_path)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 32kHz if needed
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]

    # Limit length for testing
    if waveform.shape[1] > MAX_AUDIO_LENGTH:
        waveform = waveform[:, :MAX_AUDIO_LENGTH]

    print(f"Processed waveform shape: {waveform.shape}")

    return waveform  # Return full precision


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio from orig_sr to target_sr using linear interpolation.
    audio: (batch, channels, samples) or (channels, samples) or (samples,)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    # audio shape: (batch, channels, samples)
    batch, channels, samples = audio.shape
    # Reshape to combine batch and channels for interpolation
    audio = audio.reshape(batch * channels, 1, samples)
    # Use scale_factor instead of a computed size for ONNX export compatibility
    resampled = torch.nn.functional.interpolate(audio, scale_factor=target_sr / orig_sr, mode='linear', align_corners=False)
    new_samples = resampled.shape[-1]
    resampled = resampled.reshape(batch, channels, new_samples)
    resampled = resampled.squeeze(0).squeeze(0)
    return resampled

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    original_dtype = y.dtype
    if y.dtype == torch.float16:
        y = y.float()
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    if original_dtype == torch.float16:
        spec = spec.half()
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

class AudioPreprocess(nn.Module):
    def __init__(self, cnhubert_base_path):
        super().__init__()

        # Load the model
        self.model = HubertModel.from_pretrained(cnhubert_base_path, local_files_only=True)
        self.model.eval()

        self.sv_model = SV("cpu", False)

    def forward(self, ref_audio_32k):
        spectrum = spectrogram_torch(
            ref_audio_32k,
            2048,
            640,
            2048,
            center=False,
        )
        ref_audio_16k = resample_audio(ref_audio_32k, 32000, 16000)

        sv_emb = self.sv_model.compute_embedding3_onnx(ref_audio_16k)

        zero_tensor = torch.zeros((1, 9600), dtype=ref_audio_32k.dtype, device=ref_audio_32k.device)
        ref_audio_16k = ref_audio_16k.unsqueeze(0)
        # concate zero_tensor with waveform
        ref_audio_16k = torch.cat([ref_audio_16k, zero_tensor], dim=1)
        ssl_content = self.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

        return ssl_content, spectrum, sv_emb


def export_audio_preprocess_to_onnx(
    cnhubert_base_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    output_dir: str = "onnx/audio-preprocess"
):
    """Export AudioPreprocess model to ONNX format (full precision)"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set output paths
    onnx_path = os.path.join(output_dir, "audio-preprocess.onnx")
    mnn_path = os.path.join(output_dir, "audio-preprocess.mnn")

    print(f"Loading AudioPreprocess model with HuBERT from: {cnhubert_base_path}")

    # Create and configure the model
    preprocessor = AudioPreprocess(cnhubert_base_path)
    preprocessor.eval()

    # # Create dummy input (5 seconds at 32kHz) - use float32 for full precision export
    dummy_input = torch.randn((1, 32000 * 5), dtype=torch.float32) - 0.5

    # Export to ONNX
    print(f"Exporting full precision model to ONNX: {onnx_path}")
    torch.onnx.export(
        preprocessor,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['audio32k'],
        output_names=['hubert_ssl_output', 'spectrum', 'sv_emb'],
        dynamic_axes={
            'audio32k': {1: 'sequence_length'},
            'hubert_ssl_output': {2: 'hubert_length'},
            'spectrum': {2: 'spectrum_length'}
        }
    )
    model, _ = onnxsim.simplify(onnx_path)
    onnx.save(model, onnx_path)

    print(f"Full precision model exported successfully to: {onnx_path}")

    # Export to MNN format
    print(f"Exporting to MNN: {mnn_path}")
    mnn_command = [
        "mnnconvert",
        "--f", "ONNX",
        "--modelFile", onnx_path,
        "--optimizeLevel", "2",
        "--optimizePrefer", "2",
        "--MNNModel", mnn_path,
        "--weightQuantBits", "8",
        "--weightQuantBlock", "32"
    ]

    try:
        subprocess.run(mnn_command, check=True, capture_output=True, text=True)
        print(f"Successfully exported to MNN: {mnn_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting to MNN: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        mnn_path = None

    return preprocessor, onnx_path, mnn_path


def test_model_equivalence(original_model, onnx_path: str, mnn_path: str = None, test_audio_url: str = TEST_AUDIO_URL):
    """Test if the original PyTorch model, ONNX model, and MNN model produce similar outputs"""

    print("Testing model equivalence (full precision)...")
    print(f"Using test audio: {test_audio_url}")

    # Load and preprocess audio
    waveform = load_and_preprocess_audio(test_audio_url)

    # Get PyTorch output (use full precision)
    print("Running PyTorch inference...")
    original_model.eval()
    with torch.no_grad():
        torch_outputs = original_model(waveform)
        torch_ssl_content, torch_spectrum, torch_sv_emb = torch_outputs

    # Get ONNX output
    print("Running ONNX inference...")
    ort_session = ort.InferenceSession(onnx_path)
    input_values = waveform.numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: input_values}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_ssl_content, onnx_spectrum, onnx_sv_emb = ort_outputs

    # Compare outputs (convert to float32 for comparison)
    print("\nComparing PyTorch and ONNX outputs:")

    # SSL Content comparison
    torch_ssl_float = torch_ssl_content.numpy()
    onnx_ssl_float = onnx_ssl_content
    ssl_mean_diff = np.abs(torch_ssl_float - onnx_ssl_float).mean()
    ssl_max_diff = np.abs(torch_ssl_float - onnx_ssl_float).max()
    print(f"SSL Content - Mean diff: {ssl_mean_diff:.6f}, Max diff: {ssl_max_diff:.6f}")
    print(f"PyTorch SSL shape: {torch_ssl_float.shape}, ONNX SSL shape: {onnx_ssl_float.shape}")

    # Spectrum comparison
    torch_spec_float = torch_spectrum.numpy()
    onnx_spec_float = onnx_spectrum
    spec_mean_diff = np.abs(torch_spec_float - onnx_spec_float).mean()
    spec_max_diff = np.abs(torch_spec_float - onnx_spec_float).max()
    print(f"Spectrum - Mean diff: {spec_mean_diff:.6f}, Max diff: {spec_max_diff:.6f}")
    print(f"PyTorch Spectrum shape: {torch_spec_float.shape}, ONNX Spectrum shape: {onnx_spec_float.shape}")

    # SV Embedding comparison
    torch_sv_float = torch_sv_emb.numpy()
    onnx_sv_float = onnx_sv_emb
    sv_mean_diff = np.abs(torch_sv_float - onnx_sv_float).mean()
    sv_max_diff = np.abs(torch_sv_float - onnx_sv_float).max()
    print(f"SV Embedding - Mean diff: {sv_mean_diff:.6f}, Max diff: {sv_max_diff:.6f}")
    print(f"PyTorch SV shape: {torch_sv_float.shape}, ONNX SV shape: {onnx_sv_float.shape}")

    # Overall success check (using tighter tolerance for full precision)
    onnx_success = (ssl_mean_diff < 1e-4 and spec_mean_diff < 1e-4 and
                    sv_mean_diff < 1e-4)

    if onnx_success:
        print("✅ PyTorch and ONNX models are numerically equivalent!")
    else:
        print("❌ PyTorch and ONNX models have significant differences!")

    # Test MNN model if path is provided
    mnn_success = True
    if mnn_path and os.path.exists(mnn_path):
        print("\nTesting MNN model...")
        try:
            import MNN
            import MNN.numpy as mnp

            mnn_config = {}
            mnn_config['backend'] = 'CPU'
            mnn_config['thread'] = 12
            mnn_rt = MNN.nn.create_runtime_manager((mnn_config,))

            audio_preprocess_mnn = MNN.nn.load_module_from_file(
                mnn_path,
                ['audio32k'],
                ['hubert_ssl_output', 'spectrum', 'sv_emb'],
                runtime_manager=mnn_rt
            )

            # Prepare inputs for MNN (convert to float32 for MNN)
            mnn_inputs = [mnp.array(waveform.numpy().astype(np.float32))]

            mnn_outputs = audio_preprocess_mnn(mnn_inputs)
            mnn_ssl_content = np.array(mnn_outputs[0].read())
            mnn_spectrum = np.array(mnn_outputs[1].read())
            mnn_sv_emb = np.array(mnn_outputs[2].read())

            print(f"MNN SSL shape: {mnn_ssl_content.shape}")
            print(f"MNN Spectrum shape: {mnn_spectrum.shape}")
            print(f"MNN SV shape: {mnn_sv_emb.shape}")

            # Compare MNN with PyTorch
            print("\nComparing PyTorch and MNN outputs:")

            # SSL Content comparison
            ssl_mean_diff_mnn = np.abs(torch_ssl_float - mnn_ssl_content).mean()
            ssl_max_diff_mnn = np.abs(torch_ssl_float - mnn_ssl_content).max()
            print(f"SSL Content - Mean diff: {ssl_mean_diff_mnn:.6f}, Max diff: {ssl_max_diff_mnn:.6f}")

            # Spectrum comparison
            spec_mean_diff_mnn = np.abs(torch_spec_float - mnn_spectrum).mean()
            spec_max_diff_mnn = np.abs(torch_spec_float - mnn_spectrum).max()
            print(f"Spectrum - Mean diff: {spec_mean_diff_mnn:.6f}, Max diff: {spec_max_diff_mnn:.6f}")

            # SV Embedding comparison
            sv_mean_diff_mnn = np.abs(torch_sv_float - mnn_sv_emb).mean()
            sv_max_diff_mnn = np.abs(torch_sv_float - mnn_sv_emb).max()
            print(f"SV Embedding - Mean diff: {sv_mean_diff_mnn:.6f}, Max diff: {sv_max_diff_mnn:.6f}")

            # Check MNN equivalence (using higher tolerance for quantized MNN)
            mnn_success = (ssl_mean_diff_mnn < 1e-2 and spec_mean_diff_mnn < 1e-2 and
                          sv_mean_diff_mnn < 5e-2)

            if mnn_success:
                print("✅ MNN and PyTorch models are numerically equivalent!")
            else:
                print("❌ MNN and PyTorch models have significant differences!")

        except Exception as e:
            print(f"❌ Error testing MNN model: {e}")
            mnn_success = False
    else:
        print("MNN model path not provided or file does not exist, skipping MNN test")

    overall_success = onnx_success and (mnn_success if mnn_path else onnx_success)

    if overall_success:
        print("\n✨ All models are numerically equivalent!")
    else:
        print("\n⚠️  Some models have significant differences.")

    return overall_success


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Export AudioPreprocess model to ONNX format with automatic testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cnhubert_base_path",
        type=str,
        default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
        help="Path to the CNHuBERT model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx/audio-preprocess",
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
        original_model, onnx_path, mnn_path = export_audio_preprocess_to_onnx(
            cnhubert_base_path=args.cnhubert_base_path,
            output_dir=args.output_dir
        )

        # Test equivalence
        test_model_equivalence(
            original_model=original_model,
            onnx_path=onnx_path,
            mnn_path=mnn_path,
            test_audio_url=args.test_audio_url
        )

        print("\n✨ All done! Your full precision ONNX and MNN AudioPreprocess models are ready to use.")
        return 0

    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        return 1


if __name__ == "__main__":
    main()