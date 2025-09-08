import MNN.numpy as mnp
import MNN
import numpy as np
import torchaudio
import torch
import hashlib
from sv import SV
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
from transformers import HubertModel
import torch.nn.functional as F
import os
import onnx
import torch.nn as nn
import onnxsim
import onnxruntime as ort

def simplify_onnx_model(onnx_model_path: str):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Simplify the model
    model_simplified, _ = onnxsim.simplify(model)
    # Save the simplified model
    onnx.save(model_simplified, onnx_model_path)

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
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
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

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
    resampled = F.interpolate(audio, scale_factor=target_sr / orig_sr, mode='linear', align_corners=False)
    new_samples = resampled.shape[-1]
    resampled = resampled.reshape(batch, channels, new_samples)
    resampled = resampled.squeeze(0).squeeze(0)
    return resampled

class AudioPreprocess(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the model
        self.model = HubertModel.from_pretrained(cnhubert_base_path, local_files_only=True)
        self.model.eval()

        self.sv_model = SV("cpu", False)

    def forward(self, ref_audio_32k):
        spectrum = spectrogram_torch(
            ref_audio_32k,
            2048,
            32000,
            640,
            2048,
            center=False,
        )
        ref_audio_16k = resample_audio(ref_audio_32k, 32000, 16000)

        sv_emb = self.sv_model.compute_embedding3_onnx(ref_audio_16k)

        zero_tensor = torch.zeros((1, 9600), dtype=torch.float16)
        ref_audio_16k = ref_audio_16k.unsqueeze(0)
        # concate zero_tensor with waveform
        ref_audio_16k = torch.cat([ref_audio_16k, zero_tensor], dim=1)
        ssl_content = self.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

        return ssl_content, spectrum, sv_emb

if __name__ == "__main__":
    if not os.path.exists("onnx/audio/audio_preprocess.mnn"):
        os.makedirs("onnx/audio", exist_ok=True)
        preprocessor = AudioPreprocess()
        ref_audio32k = torch.randn((1, 32000 * 5)) - 0.5
        torch.onnx.export(preprocessor, (ref_audio32k,), "onnx/audio/audio_preprocess.onnx",
                        input_names=["audio32k"],
                        output_names=["hubert_ssl_output", "spectrum", "sv_emb"],
                        dynamic_axes={
                            "audio32k": {1: "sequence_length"},
                            "hubert_ssl_output": {2: "hubert_length"},
                            "spectrum": {2: "spectrum_length"}
                        })
        simplify_onnx_model("onnx/audio/audio_preprocess.onnx")
        os.system('mnnconvert -f ONNX --MNNModel onnx/audio/audio_preprocess.mnn --modelFile onnx/audio/audio_preprocess.onnx --info --optimizeLevel 1')

    # 配置执行后端，线程数，精度等信息；key-value请查看API介绍
    config = {}
    config['precision'] = 'low' # 当硬件支持（armv8.2）时使用fp16推理
    config['backend'] = 0       # CPU
    config['numThread'] = 4     # 线程数

    rt = MNN.nn.create_runtime_manager((config,))

    # 加载模型创建_Module
    net = MNN.nn.load_module_from_file('onnx/audio/audio_preprocess.mnn', 
                                ['audio32k'], ['hubert_ssl_output', 'spectrum', 'sv_emb'], runtime_manager=rt)

    np.random.seed(42)
    input_audio = np.random.randn(1, 32000).astype(np.float32)
    print("input shape:", input_audio.shape)


    ort_session = ort.InferenceSession('onnx/audio/audio_preprocess.onnx')
    ort_inputs = {'audio32k': input_audio}
    [hubert_feature, spectrum, sv_emb] = ort_session.run(None, ort_inputs)

    mnn_outputs = net.forward([mnp.array(input_audio)])

    print('hubert_feature diff:', np.abs(np.array(mnn_outputs[0].read()) - hubert_feature).max(), np.abs(np.array(mnn_outputs[0].read()) - hubert_feature).mean())
    print('spectrum diff:', np.abs(np.array(mnn_outputs[1].read()) - spectrum).max(), np.abs(np.array(mnn_outputs[1].read()) - spectrum).mean())
    print('sv_emb diff:', np.abs(np.array(mnn_outputs[2].read()) - sv_emb).max(), np.abs(np.array(mnn_outputs[2].read()) - sv_emb).mean())
    
    hubert_feature_md5 = hashlib.md5(np.array(mnn_outputs[0].read()).tobytes()).hexdigest()
    print("hubert_feature MD5:", hubert_feature_md5)
    spectrum_md5 = hashlib.md5(np.array(mnn_outputs[1].read()).tobytes()).hexdigest()
    print("spectrum MD5:", spectrum_md5)
    sv_emb_md5 = hashlib.md5(np.array(mnn_outputs[2].read()).tobytes()).hexdigest()
    print("sv_emb MD5:", sv_emb_md5)

    print("onnx hubert_feature MD5:", hashlib.md5(hubert_feature.tobytes()).hexdigest())
    print("onnx spectrum MD5:", hashlib.md5(spectrum.tobytes()).hexdigest())
    print("onnx sv_emb MD5:", hashlib.md5(sv_emb.tobytes()).hexdigest())