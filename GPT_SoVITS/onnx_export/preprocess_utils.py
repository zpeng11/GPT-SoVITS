import os
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
ROBERTA_PATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
AUDIO_PREPROCESSOR_PATH = "onnx/audio-preprocess/audio-preprocess.onnx"
import torchaudio
import subprocess
import onnxruntime as ort
import numpy as np


def preprocess_text(text:str, version:str = "v2"):
    if not hasattr(preprocess_text, "preprocessor"):
        bert_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
        bert_model = AutoModelForMaskedLM.from_pretrained(ROBERTA_PATH)
        preprocess_text.preprocessor = TextPreprocessor(bert_model, bert_tokenizer, 'cpu')
    [phones, bert_features, norm_text] = preprocess_text.preprocessor.segment_and_extract_feature_for_text(text, 'auto', version)
    phones = np.expand_dims(np.array(phones, dtype=np.int64), axis=0)
    return phones.astype(np.int64), bert_features.float().T.detach().cpu().numpy()

def audio_preprocess(audio_path:str):
    """Get HuBERT features for the audio file"""
    waveform, sample_rate = torchaudio.load(audio_path)
    # Resample to 32kHz if needed
    if sample_rate != 32000:
        resampler = torchaudio.transforms.Resample(sample_rate, 32000)
        waveform = resampler(waveform)
    # If stereo, take only the first channel
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
    waveform = waveform.numpy().astype(np.float32)
    
    if not hasattr(audio_preprocess, "ort_session"):
        if os.path.isfile(AUDIO_PREPROCESSOR_PATH) is False:
            subprocess.run(
                [sys.executable, "GPT_SoVITS/onnx_export/export_audio_preprocess.py"],
                check=True
            )
        audio_preprocess.ort_session = ort.InferenceSession(AUDIO_PREPROCESSOR_PATH, providers=['CPUExecutionProvider'])
    ort_inputs = {audio_preprocess.ort_session.get_inputs()[0].name: waveform}
    [hubert_feature, spectrum, sv_emb] = audio_preprocess.ort_session.run(None, ort_inputs)
    return hubert_feature.astype(np.float32), spectrum.astype(np.float32), sv_emb.astype(np.float32)