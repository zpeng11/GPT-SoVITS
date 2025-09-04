import torch
import torch.nn.functional as F
import torchaudio
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
from feature_extractor import cnhubert
from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from torch import nn
from sv import SV
import onnx
from onnx import helper, TensorProto
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
from transformers import HubertModel, HubertConfig
import os
import json
from text import cleaned_text_to_sequence
import onnxsim
from onnxconverter_common import float16
from module.mel_processing import mel_spectrogram_torch
from module.models_onnx import CFMOnnx, Generator, SynthesizerTrnV3

from io import BytesIO

spec_min = -12
spec_max = 2

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1

def denorm_spec(x):
    spec_min = -12
    spec_max = 2
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min

mel_fn_v3 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

def simplify_onnx_model(onnx_model_path: str):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Simplify the model
    model_simplified, _ = onnxsim.simplify(model)
    # Save the simplified model
    onnx.save(model_simplified, onnx_model_path)

def convert_onnx_to_half(onnx_model_path:str):
    try:
        model = onnx.load(onnx_model_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, onnx_model_path)
    except Exception as e:
        print(f"Error converting {onnx_model_path} to half precision: {e}")


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

class HifiGANVocoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        state_dict_g = torch.load(
            "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" , map_location="cpu"
        )
        print("loading vocoder", self.hifigan.load_state_dict(state_dict_g))

    def forward(self, x):
        x = denorm_spec(x)
        return self.hifigan(x)

class T2SInitStage(nn.Module):
    def __init__(self, t2s, vits:SynthesizerTrn):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.vits = vits
        self.num_layers = t2s.num_layers

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)
        prompt = prompt_semantic.unsqueeze(0)
        x = self.encoder(all_phoneme_ids, bert)

        x_seq_len = torch.onnx.operators.shape_as_tensor(x)[1]
        y_seq_len = torch.onnx.operators.shape_as_tensor(prompt)[1]

        init_k = torch.zeros(((x_seq_len + y_seq_len), self.num_layers, 512), dtype=torch.float)
        init_v = torch.zeros(((x_seq_len + y_seq_len), self.num_layers, 512), dtype=torch.float)

        return x, prompt, init_k, init_v, x_seq_len, y_seq_len

class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model.model.top_k = torch.LongTensor([self.config["inference"]["top_k"]])
        self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.init_stage = T2SInitStage(self.t2s_model, self.vits_model)
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        x, prompt, init_k, init_v, x_seq_len, y_seq_len = self.init_stage(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        empty_tensor = torch.empty((1,0,512)).to(torch.float)
        # first step
        y, k, v, y_emb, logits, samples = self.stage_decoder(x, prompt, init_k, init_v, 
                          empty_tensor, 
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, 
                          first_infer=torch.LongTensor([1]), x_seq_len=x_seq_len, y_seq_len=y_seq_len)

        for idx in range(30): # This is a fake one! DO NOT take this as reference
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1))
            y_seq_len = y.shape[1]
            y, k, v, y_emb, logits, samples = self.stage_decoder(empty_tensor, y, k, v, 
                                                                 y_emb, 
                                                                 top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, 
                                                                 first_infer=torch.LongTensor([0]), x_seq_len=x_seq_len, y_seq_len=y_seq_len)
            # if torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS or samples[0, 0] == self.t2s_model.EOS:
            #     break

        return y[:, -30:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        torch.onnx.export(
            self.init_stage,
            (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
            f"onnx/{project_name}/{project_name}_t2s_init_stage.onnx",
            input_names=["ref_text_phones", "input_text_phones", "ref_text_bert", "input_text_bert", "hubert_ssl_content"],
            output_names=["x", "prompt", "init_k", "init_v", 'x_seq_len', 'y_seq_len'],
            dynamic_axes={
                "ref_text_phones": {1: "ref_text_phones_length"},
                "input_text_phones": {1: "input_text_phones_length"},
                "ref_text_bert": {0: "ref_text_bert_length"},
                "input_text_bert": {0: "input_text_bert_length"},
                "hubert_ssl_content": {2: "hubert_ssl_content_length"},
            },
            opset_version=16,
            do_constant_folding=False
        )
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_t2s_init_stage.onnx")
        x, prompt, init_k, init_v, x_seq_len, y_seq_len = self.init_stage(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        empty_tensor = torch.empty((1,0,512)).to(torch.float)
        x_seq_len = torch.Tensor([x_seq_len]).to(torch.int64)
        y_seq_len = torch.Tensor([y_seq_len]).to(torch.int64)

        y, k, v, y_emb, logits, samples = self.stage_decoder(x, prompt, init_k, init_v, 
                                                             empty_tensor, 
                                                             top_k, top_p, repetition_penalty, temperature, 
                                                             torch.LongTensor([1]), x_seq_len, y_seq_len)
        print(y.shape, k.shape, v.shape, y_emb.shape, logits.shape, samples.shape)
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1))
        y_seq_len = torch.Tensor([y.shape[1]]).to(torch.int64)

        torch.onnx.export(
            self.stage_decoder,
            (x, y, k, v, y_emb, top_k, top_p, repetition_penalty, temperature, torch.LongTensor([0]), x_seq_len, y_seq_len),
            f"onnx/{project_name}/{project_name}_t2s_stage_decoder.onnx",
            input_names=["ix", "iy", "ik", "iv", "iy_emb", "top_k", "top_p", "repetition_penalty", "temperature", "if_init_step", "x_seq_len", "y_seq_len"],
            output_names=["y", "k", "v", "y_emb", "logits", "samples"],
            dynamic_axes={
                "ix": {1: "ix_length"},
                "iy": {1: "iy_length"},
                "ik": {0: "ik_length"},
                "iv": {0: "iv_length"},
                "iy_emb": {1: "iy_emb_length"},
            },
            verbose=False,
            opset_version=16,
        )
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_t2s_stage_decoder.onnx")


class VitsModel(nn.Module):
    def __init__(self, vits_path, version:str = 'v2'):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu", weights_only=False)
        self.hps = dict_s2["config"]
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = version

        self.is_v2p = version.lower() in ['v2pro', 'v2proplus']

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model:SynthesizerTrn = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        # print(f"filter_length:{self.hps.data.filter_length} sampling_rate:{self.hps.data.sampling_rate} hop_length:{self.hps.data.hop_length} win_length:{self.hps.data.win_length}")
        #v2 filter_length: 2048 sampling_rate: 32000 hop_length: 640 win_length: 2048
    def forward(self, text_seq, pred_semantic, spectrum, sv_emb, speed):
        if self.is_v2p:
            return self.vq_model(pred_semantic, text_seq, spectrum, sv_emb=sv_emb, speed=speed)[0, 0]
        else:
            return self.vq_model(pred_semantic, text_seq, spectrum, speed=speed)[0, 0]

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

class VitsV4Model(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = load_sovits_new(vits_path)
        self.hps = dict_s2["config"]
        self.hps["model"]["version"] = 'v4'

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrnV3(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        # print(f"filter_length:{self.hps.data.filter_length} sampling_rate:{self.hps.data.sampling_rate} hop_length:{self.hps.data.hop_length} win_length:{self.hps.data.win_length}")
        #v2 filter_length: 2048 sampling_rate: 32000 hop_length: 640 win_length: 2048
    def forward(self, ssl_content:torch.Tensor, spectrum:torch.Tensor, ref_seq:torch.Tensor, text_seq:torch.Tensor, pred_semantic:torch.Tensor, mel2:torch.Tensor):
        codes = self.vq_model.extract_latent(ssl_content)
        prompt = codes[0, 0].unsqueeze(0)
        ge = self.vq_model.create_ge(spectrum)
        fea_ref = self.vq_model(prompt.unsqueeze(0), ref_seq, ge)
        fea_todo = self.vq_model(pred_semantic, text_seq, ge)
        T_min = torch.min(torch.onnx.operators.shape_as_tensor(mel2)[2], 
                          torch.onnx.operators.shape_as_tensor(fea_ref)[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        T_min = torch.min(torch.tensor([500]), T_min)
        mel2 = mel2[:, :, -T_min:]
        fea_ref = fea_ref[:, :, -T_min:]
        chunk_len = 1000 - T_min
        return fea_ref, fea_todo, chunk_len, mel2

class GptSoVits():
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, sv_emb, speed, project_name, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        torch.onnx.export(
            self.vits,
            (text_seq, pred_semantic, spectrum, sv_emb, speed),
            f"onnx/{project_name}/{project_name}_vits.onnx",
            input_names=["input_text_phones", "pred_semantic", "spectrum", "sv_emb", "speed"],
            output_names=["audio"],
            dynamic_axes={
                "input_text_phones": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
                "spectrum": {2: "spectrum_length"},
            },
            opset_version=17,
            verbose=False,
        )
        simplify_onnx_model(f"onnx/{project_name}/{project_name}_vits.onnx")

class GptSoVitsV4(nn.Module):
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s
        in_channels = self.vits.vq_model.cfm.in_channels
        estimator = self.vits.vq_model.cfm.estimator
        self.cfm = CFMOnnx(in_channels, estimator)
        del self.vits.vq_model.cfm
        self.hifigan = HifiGANVocoder()

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, mel2, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        audio = self.vits(ssl_content, spectrum, ref_seq, text_seq, pred_semantic, mel2)
        return audio

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, spectrum, mel2, project_name, top_k=None, top_p=None, repetition_penalty=None, temperature=None):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
        torch.onnx.export(
            self.vits,
            (ssl_content, spectrum, ref_seq, text_seq, pred_semantic, mel2),
            f"onnx/{project_name}/{project_name}_vits.onnx",
            input_names=["hubert_ssl_content", "spectrum", "ref_text_phones", "input_text_phones", "pred_semantic",  "mel2"],
            output_names=["fea_ref", "fea_todo", "chunk_len", "mel2_sliced"],
            dynamic_axes={
                "hubert_ssl_content": {2: "ssl_length"},
                "ref_text_phones": {1: "ref_text_length"},
                "input_text_phones": {1: "input_text_length"},
                "pred_semantic": {2: "pred_length"},
                "spectrum": {2: "spectrum_length"},
                "mel2": {2: "mel2_length"},
            },
            opset_version=17,
            verbose=False,
        )
        # simplify_onnx_model(f"onnx/{project_name}/{project_name}_vits.onnx")
        cmf_res_rand = torch.randn(1, 100, 934).to(torch.float32)
        torch.onnx.export(
            self.hifigan,
            (cmf_res_rand,),
            f"onnx/{project_name}/{project_name}_hifigan.onnx",
            input_names=["cfm_res"],
            output_names=["audio48k"],
            dynamic_axes={
                "cfm_res": {2: "cfm_length"},
                "audio48k": {2: "audio_length"},
            },
            opset_version=17,
            verbose=False,
        )
        # simplify_onnx_model(f"onnx/{project_name}/{project_name}_hifigan.onnx")
        mu_random = torch.randn(1, 930, 512).to(torch.float32)
        prompt_random = torch.randn(1, 100, 486).to(torch.float32)
        n_timesteps = torch.LongTensor([32])
        i_timesteps = torch.LongTensor([16])
        temperature = torch.FloatTensor([1.0])
        x_last = torch.randn(1, 100, 930).to(torch.float32)
        text_cache_empty = torch.empty((0, 930, 512), dtype=mu_random.dtype)
        dt_cache_empty = torch.empty((0, 1024), dtype=mu_random.dtype)
        torch.onnx.export(
            self.cfm,
            (mu_random, prompt_random, n_timesteps, i_timesteps, temperature, x_last, text_cache_empty, dt_cache_empty),
            f"onnx/{project_name}/{project_name}_cfm.onnx",
            input_names=["mu", "prompt", "n_timesteps", "i_timesteps", "temperature", "x_last", "text_cache", "dt_cache"],
            output_names=["x_last_output", "text_cache_output", "dt_cache_output"],
            dynamic_axes={
                "mu": {1: "mu_length"},
                "prompt": {2: "prompt_length"},
                "x_last": {0:"x_last_length", 2: "mu_length"},
                "text_cache": {0: "text_cache_length", 1: "mu_length"},
                "dt_cache": {0: "dt_cache_length"},
            },
            opset_version=17,
            verbose=False,
        )
        # simplify_onnx_model(f"onnx/{project_name}/{project_name}_cfm.onnx")

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

        zero_tensor = torch.zeros((1, 9600), dtype=torch.float32)
        ref_audio_16k = ref_audio_16k.unsqueeze(0)
        # concate zero_tensor with waveform
        ref_audio_16k = torch.cat([ref_audio_16k, zero_tensor], dim=1)
        ssl_content = self.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

        mel2_v4 = mel_fn_v4(ref_audio_32k)
        mel2_v4 = norm_spec(mel2_v4)
        return ssl_content, spectrum, sv_emb, mel2_v4

def export(vits_path, gpt_path, project_name, voice_model_version, export_audio_preprocessor=True, half_precision=False):
    if voice_model_version.lower() == 'v4':
        vits = VitsV4Model(vits_path)
    else:
        vits = VitsModel(vits_path, version=voice_model_version)
    gpt = T2SModel(gpt_path, vits)
    if voice_model_version.lower() == 'v4':
        gpt_sovits = GptSoVitsV4(vits, gpt)
    else:
        gpt_sovits = GptSoVits(vits, gpt)
    preprocessor = AudioPreprocess()
    ref_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "n",
                    "i2",
                    "h",
                    "ao3",
                    ",",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                ],
                version='v2',
            )
        ]
    )
    text_seq = torch.LongTensor(
        [
            cleaned_text_to_sequence(
                [
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                    "w",
                    "o3",
                    "sh",
                    "i4",
                    "b",
                    "ai2",
                    "y",
                    "e4",
                ],
                version='v2',
            )
        ]
    )
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio32k = torch.randn((1, 32000 * 5)).float() - 0.5 # 5 seconds of dummy audio
    top_k = torch.LongTensor([15])
    top_p = torch.FloatTensor([1.0])
    repetition_penalty = torch.FloatTensor([1.0])
    temperature = torch.FloatTensor([1.0])
    speed = torch.FloatTensor([1.0])

    os.makedirs(f"onnx/{project_name}", exist_ok=True)

    [ssl_content, spectrum, sv_emb, mel2_v4] = preprocessor(ref_audio32k)
    if voice_model_version.lower() == 'v4':
        gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content.float(), spectrum.float(), mel2_v4, project_name, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)
    else:
        gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content.float(), spectrum.float(), sv_emb.float(), speed, project_name, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)

    if export_audio_preprocessor:
        torch.onnx.export(preprocessor, (ref_audio32k,), f"onnx/{project_name}/{project_name}_audio_preprocess.onnx",
                    input_names=["audio32k"],
                    output_names=["hubert_ssl_output", "spectrum", "sv_emb", "mel2_v4"],
                    dynamic_axes={
                        "audio32k": {1: "sequence_length"},
                        "hubert_ssl_output": {2: "hubert_length"},
                        "spectrum": {2: "spectrum_length"},
                        "mel2_v4": {2: "mel2_length"}
                    })
        # simplify_onnx_model(f"onnx/{project_name}/{project_name}_audio_preprocess.onnx")
        
    if half_precision:
        if export_audio_preprocessor:
            convert_onnx_to_half(f"onnx/{project_name}/{project_name}_audio_preprocess.onnx")
        convert_onnx_to_half(f"onnx/{project_name}/{project_name}_vits.onnx")
        convert_onnx_to_half(f"onnx/{project_name}/{project_name}_t2s_init_step.onnx")
        convert_onnx_to_half(f"onnx/{project_name}/{project_name}_t2s_stage_step.onnx")

    configJson = {
        "project_name": project_name,
        "type": "GPTSoVITS",
        "version" : voice_model_version,
        "bert_base_path": 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
        "cnhuhbert_base_path": 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
        "t2s_weights_path": gpt_path,
        "vits_weights_path": vits_path,
        "half_precision": half_precision
    }
    with open(f"onnx/{project_name}/config.json", "w", encoding="utf-8") as f:
        json.dump(configJson, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    try:
        os.mkdir("onnx")
    except:
        pass

    # 因为io太频繁，可能导致模型导出出错(wsl非常明显)，请自行重试

    # gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    # vits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
    # exp_path = "v1_export"
    # version = "v1"
    # export(vits_path, gpt_path, exp_path, version)

    # gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    # vits_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    # exp_path = "v2_export"
    # version = "v2"
    # export(vits_path, gpt_path, exp_path, version)
    

    # gpt_path = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    # vits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    # exp_path = "v2pro_export"
    # version = "v2Pro"
    # export(vits_path, gpt_path, exp_path, version)

    # gpt_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    # vits_path = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
    # exp_path = "v2proplus_export"
    # version = "v2ProPlus"
    # export(vits_path, gpt_path, exp_path, version)

    gpt_path = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
    exp_path = "v4_export"
    version = "v4"
    export(vits_path, gpt_path, exp_path, version)

    
