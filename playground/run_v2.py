import onnxruntime as ort
import numpy as np
import onnx
from tqdm import tqdm
import torchaudio
import torch
from TTS_infer_pack.TextPreprocessor_onnx import TextPreprocessorOnnx


MODEL_PATH = "onnx/v2_export/v2"
OUTPUT_PATH = 'playground/output.wav'
REF_AUDIO_PATH = "playground/ref/audio.wav"
REF_TEXT = "近日江苏苏州荷花市集开张热闹与浪漫交织"
INPUT_TEXT = "天上的风筝在天上飞，地上的人儿在地上追。"
ROBERTA_PATH = "playground/chinese-roberta-wwm-ext-large"
TEMPERATURE = 1.0
TOP_K = 15
TOP_P = 1.0
REPETITION_PENALTY = 1.35
SPEED = 1.0

def audio_postprocess(
    audios,
    fragment_interval: float = 0.3,
):
    zero_wav = np.zeros((int(32000 * fragment_interval),)).astype(np.float16)
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
    ort_inputs = {ort_session.get_inputs()[0].name: waveform.numpy().astype(np.float16)}
    [hubert_feature, spectrum, sv_emb, mel2_v4] = ort_session.run(None, ort_inputs)
    return hubert_feature, spectrum

def preprocess_text(text:str):
    preprocessor = TextPreprocessorOnnx(ROBERTA_PATH)
    [phones, bert_features, norm_text] = preprocessor.segment_and_extract_feature_for_text(text, 'all_zh', 'v2')
    phones = np.expand_dims(np.array(phones, dtype=np.int64), axis=0)
    return phones, bert_features.T.astype(np.float16)


# input_phones_saved = np.load("playground/ref/input_phones.npy")
# input_bert_saved = np.load("playground/ref/input_bert.npy").T.astype(np.float16)
[input_phones, input_bert] = preprocess_text(INPUT_TEXT)


# ref_phones = np.load("playground/ref/ref_phones.npy")
# ref_bert = np.load("playground/ref/ref_bert.npy").T.astype(np.float16)
[ref_phones, ref_bert] = preprocess_text(REF_TEXT)


[audio_prompt_hubert, spectrum] = audio_preprocess(REF_AUDIO_PATH)

# audio_prompt_hubert_saved = np.load("playground/ref/audio_prompt_hubert.npy").astype(np.float16)

top_k = np.array([TOP_K], dtype=np.int64)
top_p = np.array([TOP_P], dtype=np.float16)
repetition_penalty = np.array([REPETITION_PENALTY], dtype=np.float16)
temperature = np.array([TEMPERATURE], dtype=np.float16)

t2s_init_stage = ort.InferenceSession(MODEL_PATH+"_export_t2s_init_stage.onnx")
# t2s_init_step = ort.InferenceSession(MODEL_PATH+"_export_t2s_init_step.onnx")

[x, prompts, init_k, init_v, init_y_emb, x_seq_len, y_seq_len] = t2s_init_stage.run(None, {
    "input_text_phones": input_phones,
    "input_text_bert": input_bert,
    "ref_text_phones": ref_phones,
    "ref_text_bert": ref_bert,
    "hubert_ssl_content": audio_prompt_hubert,
})
empty_tensor = np.empty((1,0,512)).astype(np.float16)

t2s_stage_decoder = ort.InferenceSession(MODEL_PATH+"_export_t2s_stage_decoder.onnx")
y, k, v, y_emb, logits, samples = t2s_stage_decoder.run(None, {
    "ix": x,
    "iy": prompts,
    "ik": init_k,
    "iv": init_v,
    "iy_emb": init_y_emb,
    "top_k": top_k,
    "top_p": top_p,
    "repetition_penalty": repetition_penalty,
    "temperature": temperature,
    "if_init_step": np.array([1]).astype(np.int64),
    "x_seq_len": np.array([x_seq_len]).astype(np.int64),
    "y_seq_len": np.array([y_seq_len]).astype(np.int64)
})

for idx in tqdm(range(1, 1500)):
    k = np.pad(k, ((0,1), (0,0), (0,0)))
    v = np.pad(v, ((0,1), (0,0), (0,0)))
    y_emb = np.pad(y_emb, ((0,0), (0,1), (0,0)))
    y_seq_len = np.array([y.shape[1]]).astype(np.int64)
    # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
    [y, k_increasement, v_increasement, y_emb_increasement, logits, samples] = t2s_stage_decoder.run(None, {
        "ix": empty_tensor,
        "iy": y,
        "ik": k,
        "iv": v,
        "iy_emb": y_emb,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "if_init_step": np.array([0]).astype(np.int64),
        "x_seq_len": np.array([x_seq_len]).astype(np.int64),
        "y_seq_len": y_seq_len
    })
    k[-1:,:,:] = k_increasement
    v[-1:,:,:] = v_increasement
    y_emb[:,-1:,:] = y_emb_increasement
    if np.argmax(logits, axis=-1)[0] == 1024 or samples[0, 0] == 1024: # 1024 is the EOS token
        break
y = y[:,:-1]


pred_semantic = np.expand_dims(y[:, -idx:], axis=0)

vtis = ort.InferenceSession(MODEL_PATH+"_export_vits.onnx")

[audio] = vtis.run(None, {
    "input_text_phones": input_phones,
    "pred_semantic": pred_semantic,
    "spectrum": spectrum.astype(np.float16),
    "speed": np.array([SPEED]).astype(np.float16)
})

audio_postprocess([audio])
