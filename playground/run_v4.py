import onnxruntime as ort
import numpy as np
import onnx
from tqdm import tqdm
import torchaudio
import torch
from TTS_infer_pack.TextPreprocessor_onnx import TextPreprocessorOnnx


MODEL_PATH = "onnx/v4_export/v4"
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
    zero_wav = np.zeros((int(48000 * fragment_interval),)).astype(np.float32)
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

    torchaudio.save(OUTPUT_PATH, audio_tensor, 48000)

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
    return hubert_feature, spectrum, sv_emb, mel2_v4

def preprocess_text(text:str):
    preprocessor = TextPreprocessorOnnx(ROBERTA_PATH)
    [phones, bert_features, norm_text] = preprocessor.segment_and_extract_feature_for_text(text, 'all_zh', 'v2')
    phones = np.expand_dims(np.array(phones, dtype=np.int64), axis=0)
    return phones, bert_features.T.astype(np.float16)


# input_phones_saved = np.load("playground/ref/input_phones.npy")
# input_bert_saved = np.load("playground/ref/input_bert.npy").T.astype(np.float32)
[input_phones, input_bert] = preprocess_text(INPUT_TEXT)


# ref_phones = np.load("playground/ref/ref_phones.npy")
# ref_bert = np.load("playground/ref/ref_bert.npy").T.astype(np.float32)
[ref_phones, ref_bert] = preprocess_text(REF_TEXT)


[audio_prompt_hubert, spectrum, sv_emb, mel2_v4] = audio_preprocess(REF_AUDIO_PATH)

# audio_prompt_hubert_saved = np.load("playground/ref/audio_prompt_hubert.npy").astype(np.float32)

top_k = np.array([TOP_K], dtype=np.int64)
top_p = np.array([TOP_P], dtype=np.float16)
repetition_penalty = np.array([REPETITION_PENALTY], dtype=np.float16)
temperature = np.array([TEMPERATURE], dtype=np.float16)

t2s_init_stage = ort.InferenceSession(MODEL_PATH+"_export_t2s_init_stage.onnx")
# t2s_init_step = ort.InferenceSession(MODEL_PATH+"_export_t2s_init_step.onnx")

[x, prompts, init_k, init_v, x_seq_len, y_seq_len] = t2s_init_stage.run(None, {
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
    "iy_emb": empty_tensor,
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
    y_seq_len = np.array([y.shape[1]]).astype(np.int64)
    # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
    [y, k, v, y_emb, logits, samples] = t2s_stage_decoder.run(None, {
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
    if np.argmax(logits, axis=-1)[0] == 1024 or samples[0, 0] == 1024: # 1024 is the EOS token
        break
y = y[:,:-1]


pred_semantic = np.expand_dims(y[:, -idx:], axis=0)

vits = ort.InferenceSession(MODEL_PATH+"_export_vits.onnx")

speed = np.array([SPEED], dtype=np.float16)
fea_ref, fea_todo, chunk_len, mel2_v4 = vits.run(None, {
    'hubert_ssl_content': audio_prompt_hubert,
    'spectrum': spectrum,
    'ref_text_phones': ref_phones,
    'input_text_phones': input_phones,
    'pred_semantic': pred_semantic,
    'mel2': mel2_v4,
    'speed': speed
})


def cfm_onnx(fea, mel, sample_steps):
    # Implement the ONNX export logic for CFM here
    cfm_onnx = ort.InferenceSession('onnx/v4_export/v4_export_cfm.onnx')
    mu_length = fea.shape[1]
    x_last = np.empty((0, 100, mu_length), dtype=np.float16)
    temperature = np.array([1.0], dtype=np.float16)
    n_timesteps = np.array([sample_steps], dtype=np.int64)
    i_timestep = np.array([0], dtype=np.int64)
    mu = fea
    prompt = mel
    text_cache = np.empty((0, mu_length, 512), dtype=np.float16)
    dt_cache = np.empty((0, 1024), dtype=np.float16)
    for _ in tqdm(range(sample_steps)):
        x_last, text_cache, dt_cache = cfm_onnx.run(None, {
            'x_last': x_last,
            'mu': mu,
            'prompt': prompt,
            'temperature': temperature,
            'n_timesteps': n_timesteps,
            'i_timesteps': i_timestep,
            'text_cache': text_cache,
            'dt_cache': dt_cache,
        })
        i_timestep[0] += 1
    return x_last

chunk_len = chunk_len[0]
sample_steps = 32
T_min = mel2_v4.shape[2]
cfm_resss = []
idx = 0
while 1:
    fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
    if fea_todo_chunk.shape[-1] == 0:
        break
    idx += chunk_len
    fea = np.swapaxes(np.concatenate([fea_ref, fea_todo_chunk], axis=2), 1, 2)

    cfm_res = cfm_onnx(fea, mel2_v4, sample_steps)

    cfm_res = cfm_res[:, :, mel2_v4.shape[2] :]

    mel2_v4 = cfm_res[:, :, -T_min:]
    fea_ref = fea_todo_chunk[:, :, -T_min:]

    cfm_resss.append(cfm_res)
cfm_res = np.concatenate(cfm_resss, axis=2)

hifigan = ort.InferenceSession(MODEL_PATH+"_export_hifigan.onnx")
audio = hifigan.run(None, {
    "cfm_res": cfm_res
})[0]

audio_postprocess([audio[0,0,:]])