import MNN
import MNN.cv as cv
import MNN.expr as expr
import onnxruntime as ort
import numpy as np
import MNN.numpy as mnp
import onnx
from tqdm import tqdm
import torchaudio
import torch
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from transformers import AutoModelForMaskedLM, AutoTokenizer
from playground.sampler import sample


MODEL_PATH = "onnx/v2proplus_export/v2proplus"
VERSION = "v2proplus"
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

config = {}
config['backend'] = 'CPU'
config['thread'] = 1
rt = MNN.nn.create_runtime_manager((config,))

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

    return waveform.detach().cpu().numpy()

def audio_preprocess(audio_path):
    """Get HuBERT features for the audio file"""
    net = MNN.nn.load_module_from_file(MODEL_PATH + "_export_audio_preprocess.mnn", 
                               ['audio32k'], ['hubert_ssl_output', 'spectrum', 'sv_emb'], runtime_manager=rt)
    waveform = mnp.array(load_audio(audio_path)).astype(mnp.float32)
    outputs = net.forward([waveform])
    return np.array(outputs[0].read()), np.array(outputs[1].read()), np.array(outputs[2].read())

def audio_preprocess_onnx(audio_path):
    """Get HuBERT features for the audio file"""
    waveform = load_audio(audio_path).astype(np.float32)
    ort_session = ort.InferenceSession(MODEL_PATH + "_export_audio_preprocess.onnx")
    ort_inputs = {'audio32k': waveform}
    [hubert_feature, spectrum, sv_emb] = ort_session.run(None, ort_inputs)
    return hubert_feature, spectrum, sv_emb

def preprocess_text(text:str):
    bert_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    bert_model = AutoModelForMaskedLM.from_pretrained(ROBERTA_PATH)
    preprocessor = TextPreprocessor(bert_model, bert_tokenizer, 'cpu')
    [phones, bert_features, norm_text] = preprocessor.segment_and_extract_feature_for_text(text, 'all_zh', VERSION)
    phones = np.expand_dims(np.array(phones, dtype=np.int64), axis=0)
    return mnp.array(phones).astype(mnp.int64), mnp.array(bert_features.float().T.detach().cpu().numpy())

[input_phones, input_bert] = preprocess_text(INPUT_TEXT)

[ref_phones, ref_bert] = preprocess_text(REF_TEXT)


[audio_prompt_hubert, spectrum, sv_emb] = audio_preprocess(REF_AUDIO_PATH)
[audio_prompt_hubert_onnx, spectrum_onnx, sv_emb_onnx] = audio_preprocess_onnx(REF_AUDIO_PATH)
print('mnn:', audio_prompt_hubert.dtype, spectrum.dtype, sv_emb.dtype)
print('onnx:', audio_prompt_hubert_onnx.dtype, spectrum_onnx.dtype, sv_emb_onnx.dtype)
print('hubert diff:', np.abs(audio_prompt_hubert - audio_prompt_hubert_onnx).max(), np.abs(audio_prompt_hubert - audio_prompt_hubert_onnx).mean(), audio_prompt_hubert_onnx.max(), audio_prompt_hubert_onnx.min())
print('spectrum diff:', np.abs(spectrum - spectrum_onnx).max(), np.abs(spectrum - spectrum_onnx).mean(), spectrum_onnx.max(), spectrum_onnx.min())
print('sv_emb diff:', np.abs(sv_emb - sv_emb_onnx).max(), np.abs(sv_emb - sv_emb_onnx).mean(), sv_emb_onnx.max(), sv_emb_onnx.min())
# audio_prompt_hubert = audio_prompt_hubert_onnx
# spectrum = spectrum_onnx
# sv_emb = sv_emb_onnx

fsdec = MNN.nn.load_module_from_file(MODEL_PATH + "_export_t2s_fsdec.mnn",
                                              ['input_text_phones', 'input_text_bert', 'ref_text_phones', 'ref_text_bert', 'hubert_ssl_content'],
                                                ['k', 'v', 'y_emb', 'x_example', 'logits', 'prompts'], runtime_manager=rt)


top_k = np.array([TOP_K], dtype=np.int32)
top_p = np.array([TOP_P], dtype=np.float32)
repetition_penalty = np.array([REPETITION_PENALTY], dtype=np.float32)
temperature = np.array([TEMPERATURE], dtype=np.float32)

[k, v, y_emb, x_example, logits, prompts] = fsdec([input_phones, input_bert, ref_phones, ref_bert, mnp.array(audio_prompt_hubert)])
y = sample(np.array(logits.read()), np.array(prompts.read())[0], top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
k = np.array(k.read())
v = np.array(v.read())
y_emb = np.array(y_emb.read())
prompts = np.array(prompts.read())
logits = np.array(logits.read())


fsdec_onnx = ort.InferenceSession(MODEL_PATH+"_export_t2s_fsdec.onnx")
k_onnx, v_onnx, y_emb_onnx, x_example_onnx, logits_onnx, prompts_onnx = fsdec_onnx.run(None, {
    "input_text_phones": np.array(input_phones.read()).astype(np.int64),
    "input_text_bert": np.array(input_bert.read()).astype(np.float32),
    "ref_text_phones": np.array(ref_phones.read()).astype(np.int64),
    "ref_text_bert": np.array(ref_bert.read()).astype(np.float32),
    "hubert_ssl_content": np.array(audio_prompt_hubert).astype(np.float32),
})
# k = k_onnx
# v = v_onnx
# y_emb = y_emb_onnx
# x_example = x_example_onnx
# logits = logits_onnx
# prompts = prompts_onnx
# y = sample(logits, prompts[0], top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]


print('k diff:', np.abs(k - k_onnx).max(), np.abs(k - k_onnx).mean(), k_onnx.max(), k_onnx.min())
print('v diff:', np.abs(v - v_onnx).max(), np.abs(v - v_onnx).mean(), v_onnx.max(), v_onnx.min())
print('y_emb diff:', np.abs(y_emb - y_emb_onnx).max(), np.abs(y_emb - y_emb_onnx).mean(), y_emb_onnx.max(), y_emb_onnx.min())
print('logits diff:', np.abs(logits - logits_onnx).max(), np.abs(logits - logits_onnx).mean(), logits_onnx.max(), logits_onnx.min())
print('prompts diff:', np.abs(prompts - prompts_onnx).max(), np.abs(prompts - prompts_onnx).mean(), prompts_onnx.max(), prompts_onnx.min())


sdec = MNN.nn.load_module_from_file(MODEL_PATH + "_export_t2s_sdec.mnn",
                                              ['iy', 'ik', 'iv', 'iy_emb', 'ix_example'],
                                              ['k_increasement', 'v_increasement', 'y_emb_increasement', 'logits'], runtime_manager=rt)

# k = mnp.pad(k, ((0,1), (0,0), (0,0)))
# v = mnp.pad(v, ((0,1), (0,0), (0,0)))
# y_emb = mnp.pad(y_emb, ((0,0), (0,1), (0,0)))
# [k_increasement, v_increasement, y_emb_increasement, logits] = sdec([y, k, v, y_emb, x_example])


# sdec = ort.InferenceSession(MODEL_PATH+"_export_t2s_sdec.onnx")
# [k_increasement_onnx, v_increasement_onnx, y_emb_increasement_onnx, logits_onnx] = sdec.run(None, {
#         "iy": y,
#         "ik": k.read(),
#         "iv": v.read(),
#         "iy_emb": y_emb.read(),
#         "ix_example": x_example.read(),
#     })

# print('k diff:', np.abs(k_increasement.read() - k_increasement_onnx).max(), np.abs(k_increasement.read() - k_increasement_onnx).mean(), k_increasement_onnx.max(), k_increasement_onnx.min())
# print('v diff:', np.abs(v_increasement.read() - v_increasement_onnx).max(), np.abs(v_increasement.read() - v_increasement_onnx).mean(), v_increasement_onnx.max(), v_increasement_onnx.min())
# print('y_emb diff:', np.abs(y_emb_increasement.read() - y_emb_increasement_onnx).max(), np.abs(y_emb_increasement.read() - y_emb_increasement_onnx).mean(), y_emb_increasement_onnx.max(), y_emb_increasement_onnx.min())
# print('logits diff:', np.abs(logits.read() - logits_onnx).max(), np.abs(logits.read() - logits_onnx).mean(), logits_onnx.max(), logits_onnx.min())


for idx in tqdm(range(1, 1500)):
    k = np.pad(k, ((0,1), (0,0), (0,0)))
    v = np.pad(v, ((0,1), (0,0), (0,0)))
    y_emb = np.pad(y_emb, ((0,0), (0,1), (0,0)))
    [k_increasement, v_increasement, y_emb_increasement, logits] = sdec([mnp.array(y), mnp.array(k), mnp.array(v), mnp.array(y_emb), mnp.array(x_example)])
    k_increasement = np.array(k_increasement.read())
    v_increasement = np.array(v_increasement.read())
    y_emb_increasement = np.array(y_emb_increasement.read())
    logits = np.array(logits.read())
    y_sample = sample(logits, y[0], top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
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
    "input_text_phones": input_phones.read().astype(np.int64),
    "pred_semantic": pred_semantic,
    "spectrum": spectrum.astype(np.float32),
    "sv_emb": sv_emb.astype(np.float32) if VERSION in ['v2pro', 'v2proplus'] else None,
})

audio_postprocess([audio])