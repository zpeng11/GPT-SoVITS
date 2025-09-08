import onnxruntime as ort
import numpy as np
import onnx
from tqdm import tqdm
import torchaudio
import torch
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from transformers import AutoModelForMaskedLM, AutoTokenizer
from AR.models.t2s_model_onnx import sample


MODEL_PATH = "onnx/v1_export/v1"
VERSION = "v1"
OUTPUT_PATH = 'playground/output.wav'
REF_AUDIO_PATH = "playground/ref/audio.wav"
REF_TEXT = "近日江苏苏州荷花市集开张热闹与浪漫交织"
INPUT_TEXT = "天上的风筝在天上飞，地上的人儿在地上追。"
ROBERTA_PATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
TEMPERATURE = 1.0
TOP_K = 15
TOP_P = 1.0
REPETITION_PENALTY = 1.35
SPEED = 1.0

def audio_postprocess(
    audios,
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

    # audio = (audio * 32768).astype(np.int16)

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    torchaudio.save(OUTPUT_PATH, audio_tensor, 32000)

    return audio

def load_audio(audio_path):
    """Load and preprocess audio file to 32k"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 32kHz if needed
    if sample_rate != 32000:
        resampler = torchaudio.transforms.Resample(sample_rate, 32000)
        waveform = resampler(waveform)

    # Take first channel
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]

    return waveform

def audio_preprocess(audio_path):
    """Get HuBERT features for the audio file"""
    waveform = load_audio(audio_path)
    ort_session = ort.InferenceSession(MODEL_PATH + "_export_audio_preprocess.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: waveform.numpy().astype(np.float32)}
    [hubert_feature, spectrum, sv_emb] = ort_session.run(None, ort_inputs)
    return hubert_feature, spectrum, sv_emb

def preprocess_text(text:str):
    bert_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    bert_model = AutoModelForMaskedLM.from_pretrained(ROBERTA_PATH)
    preprocessor = TextPreprocessor(bert_model, bert_tokenizer, 'cpu')
    [phones, bert_features, norm_text] = preprocessor.segment_and_extract_feature_for_text(text, 'all_zh', VERSION)
    phones = np.expand_dims(np.array(phones, dtype=np.int64), axis=0)
    return phones, bert_features.float().T.detach().cpu().numpy()

[input_phones, input_bert] = preprocess_text(INPUT_TEXT)

[ref_phones, ref_bert] = preprocess_text(REF_TEXT)

[audio_prompt_hubert, spectrum, sv_emb] = audio_preprocess(REF_AUDIO_PATH)


top_k = torch.Tensor([TOP_K]).to(torch.int64)
top_p = torch.Tensor([TOP_P]).to(torch.float32)
repetition_penalty = torch.Tensor([REPETITION_PENALTY]).to(torch.float32)
temperature = torch.Tensor([TEMPERATURE]).to(torch.float32)

fsdec = ort.InferenceSession(MODEL_PATH+"_export_t2s_fsdec.onnx")
k, v, y_emb, x_example, logits, prompts = fsdec.run(None, {
    "input_text_phones": input_phones,
    "input_text_bert": input_bert,
    "ref_text_phones": ref_phones,
    "ref_text_bert": ref_bert,
    "hubert_ssl_content": audio_prompt_hubert,
})

y = sample(torch.from_numpy(logits), torch.from_numpy(prompts[0]), top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0].unsqueeze(0).detach().cpu().numpy()

sdec = ort.InferenceSession(MODEL_PATH+"_export_t2s_sdec.onnx")

for idx in tqdm(range(1, 1500)):
    k = np.pad(k, ((0,1), (0,0), (0,0)))
    v = np.pad(v, ((0,1), (0,0), (0,0)))
    y_emb = np.pad(y_emb, ((0,0), (0,1), (0,0)))
    [k_increasement, v_increasement, y_emb_increasement, logits] = sdec.run(None, {
        "iy": y,
        "ik": k,
        "iv": v,
        "iy_emb": y_emb,
        "ix_example": x_example,
    })
    y_sample = sample(torch.from_numpy(logits), torch.from_numpy(y[0]), top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0].unsqueeze(0).detach().cpu().numpy()
    k[-1:,:,:] = k_increasement
    v[-1:,:,:] = v_increasement
    y_emb[:, -1, :] = y_emb_increasement
    y = np.concatenate([y, y_sample[:, -1:]], axis=1)
    if np.argmax(logits) == 1024 or y_sample[0, 0] == 1024: # 1024 is the EOS token
        break
y = y[:,:-1]


pred_semantic = np.expand_dims(y[:, -idx:], axis=0)

vtis = ort.InferenceSession(MODEL_PATH+"_export_vits.onnx")

[audio] = vtis.run(None, {
    "input_text_phones": input_phones,
    "pred_semantic": pred_semantic,
    "spectrum": spectrum.astype(np.float32),
    "sv_emb": sv_emb.astype(np.float32) if VERSION in ['v2pro', 'v2proplus'] else None,
})

audio_postprocess([audio])