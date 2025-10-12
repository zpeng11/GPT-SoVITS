import numpy as np
import onnxruntime as ort
import MNN
from MNN import numpy as mnp
from typing import List
import os
import shutil
import json
import tempfile
import bsdiff4
import os, sys
from tqdm import tqdm
import wave
sys.path.append(os.path.dirname(__file__))

from preprocess_utils import preprocess_text


class MNNInferenceSession:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MNN model file not found: {model_path}")
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession({
            'backend': 'CPU',
            'thread': 8
        })
        self.inputs = self.interpreter.getSessionInputAll(self.session)
        self.outputs = self.interpreter.getSessionOutputAll(self.session)
    def run(self, output_name_list: List[str], input_dict: dict):
        for name, data in input_dict.items():
            if name not in self.inputs:
                raise ValueError(f"Input name {name} not found in model inputs.")
            input_tensor = self.inputs[name]
            self.interpreter.resizeTensor(input_tensor, data.shape)
        for name in output_name_list:
            if name not in self.outputs:
                raise ValueError(f"Output name {name} not found in model outputs.")
        self.interpreter.resizeSession(self.session)
        for name, data in input_dict.items():
            if data.dtype == np.int64:
                data = data.astype(np.int32)
            if data.dtype == np.float16:
                data = data.astype(np.float32)
            input_tensor: np.ndarray = self.inputs[name].getNumpyData()
            np.copyto(input_tensor, data)
        self.interpreter.runSession(self.session)
        output_list = [None] * len(output_name_list)
        for name, data in self.outputs.items():
            if name in output_name_list:
                idx = output_name_list.index(name)
                output_list[idx] = data.getNumpyData()
        return output_list

def audio_postprocess(
    audios,
    output_path: str,
    fragment_interval: float = 0.3,
):
    zero_wav = np.zeros((int(32000 * fragment_interval),)).astype(np.float32)
    for i, audio in enumerate(audios):
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio
        audio = audio.astype(np.float32)
        audio = np.concatenate([audio, zero_wav], axis=0)
        audios[i] = audio

    audio = np.concatenate(audios, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(32000)
        wav_file.writeframes(audio.tobytes())

class GSVRuntime:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f".gsv model file not found: {model_path}")
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        temp_dir = tempfile.mkdtemp(prefix=f"gsv_{base_name}_")
        shutil.unpack_archive(model_path, temp_dir, "zip")
        self.config: dict = json.load(open(os.path.join(temp_dir, "config.json"), "r", encoding="utf-8"))
        print(f"Loading project: {self.config['project_name']}")
        self.sovits: MNNInferenceSession = MNNInferenceSession(os.path.join(temp_dir, 'sovits', 'sovits_v1v2.mnn'))
        self.t2s_fsdec: MNNInferenceSession | ort.InferenceSession = None
        self.t2s_sdec: ort.InferenceSession = None
        if self.config["quantized"]:
            bsdiff4.file_patch(
                os.path.join(temp_dir, 't2s', 't2s_sdec_quant.onnx'),
                os.path.join(temp_dir, 't2s', 't2s_fsdec_quant.onnx'),
                os.path.join(temp_dir, 't2s', 't2s_fsdec_quant.diff4')
            )
            self.t2s_fsdec = ort.InferenceSession(os.path.join(temp_dir, 't2s', 't2s_fsdec_quant.onnx'))
            self.t2s_sdec = ort.InferenceSession(os.path.join(temp_dir, 't2s', 't2s_sdec_quant.onnx'))
        else:
            self.t2s_fsdec = MNNInferenceSession(os.path.join(temp_dir, 't2s', 't2s_fsdec.mnn'))
            self.t2s_sdec = ort.InferenceSession(os.path.join(temp_dir, 't2s', 't2s_sdec.onnx'))
        self.ref_text_seq = np.load(os.path.join(temp_dir, "reference", "ref_text_seq.npy"))
        self.ref_text_bert = np.load(os.path.join(temp_dir, "reference", "ref_text_bert.npy"))
        self.ref_sv_emb = np.load(os.path.join(temp_dir, "reference", "ref_sv_emb.npy"))
        self.ref_ssl_content = np.load(os.path.join(temp_dir, "reference", "ref_ssl_content.npy"))
        self.ref_spectrum = np.load(os.path.join(temp_dir, "reference", "ref_spectrum.npy"))
        print("Model and reference data loaded successfully.")
    def infer(self, text: str):
        phones, bert_features = preprocess_text(text, version=self.config['version'])
        # T2S fsdec
        fsdec_input = {
            'encoder_ref_seq': self.ref_text_seq,
            'encoder_ref_bert': self.ref_text_bert,
            'encoder_text_seq': phones,
            'encoder_text_bert': bert_features,
            'encoder_ssl_content': self.ref_ssl_content,
        }
        fsdec_output_names = []
        if self.config["quantized"]:
            fsdec_output_names = ['y', 'y_emb'] + [f'present_k_layer_{i}_quantized' for i in range(24)] + [f'present_v_layer_{i}_quantized' for i in range(24)]
        else:
            fsdec_output_names = ['y', 'y_emb'] + [f'present_k_layer_{i}' for i in range(24)] + [f'present_v_layer_{i}' for i in range(24)]
        y, y_emb, *present_kv = self.t2s_fsdec.run(fsdec_output_names, fsdec_input)
        if not self.config["quantized"]:
            y = y.astype(np.int64)
            y_emb = y_emb.astype(np.float16)
            present_kv = [kv.astype(np.float16) for kv in present_kv]

        # T2S sdec
        sdec_input_names = ['iy', 'iy_emb'] + [f'past_k_layer_{i}' for i in range(24)] + [f'past_v_layer_{i}' for i in range(24)]
        sdec_output_names = []
        if self.config["quantized"]:
            sdec_output_names = ['y', 'stop_condition_tensor', 'increased_y_emb'] + \
                                [f'increased_k_layer_{i}_quantized' for i in range(24)] + \
                                [f'increased_v_layer_{i}_quantized' for i in range(24)]
        else:
            sdec_output_names = ['y', 'stop_condition_tensor', 'increased_y_emb'] + \
                                [f'increased_k_layer_{i}' for i in range(24)] + \
                                [f'increased_v_layer_{i}' for i in range(24)]

        idx: int = 0
        for idx in tqdm(range(1000), desc="T2S SDec Inference"):
            sdec_input = {}
            for name, tensor in zip(sdec_input_names, [y, y_emb] + present_kv):
                sdec_input[name] = tensor
            y, stop_condition_tensor, y_emb_new, *new_key_values = self.t2s_sdec.run(sdec_output_names, sdec_input)
            y_emb = np.concatenate([y_emb, y_emb_new], axis=1)
            for i,(kv, new_kv) in enumerate(zip(present_kv, new_key_values)):
                present_kv[i] = np.concatenate([kv, new_kv], axis=0)
            if stop_condition_tensor:
                break
        y = y[:,:-1]
        pred_semantic = np.expand_dims(y[:, -idx:], axis=0)

        # SoVITS
        sovits_input = {
            'input_text_phones': phones.astype(np.int32),
            'pred_semantic': pred_semantic.astype(np.int32),
            'spectrum': self.ref_spectrum.astype(np.float32),
            'sv_emb': self.ref_sv_emb.astype(np.float32),
        }
        sovits_output_names = ['audio32k']
        audio32k, = self.sovits.run(sovits_output_names, sovits_input)
        return audio32k




if __name__ == "__main__":
    rt = GSVRuntime('/home/eleven/GPT-SoVITS-export/onnx/sakiko_v2pp_quant.gsv')
    audio = rt.infer("やがて来る世界を見渡せば、必ず赤い旗の世界となるだろう。")
    audio_postprocess([audio], 'onnx/output.wav')

