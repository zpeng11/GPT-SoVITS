import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader, QuantFormat
from typing import List
from tqdm import tqdm
import copy
import numpy as np
from preprocess_utils import preprocess_text, audio_preprocess
from quantization_utils import optimize_quantize,remove_quantize_and_change_input,get_initializer,set_initializer_value,find_node_by_op_name,find_nodes_children
import onnxruntime as ort
import onnx

CALIB_TEXTs =[
    "今天的天气格外晴朗，阳光透过窗户洒在桌案上。",
    "她正在图书馆里专心致志地阅读一本关于人工智能的书籍。",
    "这家餐厅的招牌菜是麻婆豆腐，味道非常地道。",
    "小猫咪懒洋洋地躺在沙发上，享受着午后的阳光。",
    "他每天早上都会去公园里跑步锻炼身体。",
    "春天来了，樱花盛开，整个城市都被粉色的花瓣装点着。",
    "学生们正在教室里认真听老师讲解数学公式。",
    "这部电影的剧情跌宕起伏，让观众看得津津有味。",
    "奶奶在厨房里忙碌着，准备一桌丰盛的晚餐。",
    "夜晚的星空格外美丽，繁星点点闪烁着微弱的光芒。",
    "量子纠缠现象在超低温环境下表现出非局域性关联特征。",
    "胞间连丝的结构复杂性影响了植物细胞间的物质传输效率。",
    "拓扑绝缘体的能带结构在狄拉克锥附近呈现线性色散关系。",
    "古生物学家在寒武纪地层中发现了三叶虫的钙化外骨骼化石。",
    "蛋白质的二级结构主要由氢键维持其α螺旋和β折叠构象。",
    "傅里叶变换在信号处理中用于频域分析和滤波器设计。",
    "星系团中的暗物质分布通过引力透镜效应得以间接观测。",
    "酶的活性位点通过诱导契合机制与底物分子结合。",
    "黎曼几何中的测地线是连接两点间距离最短的曲线。",
    "电化学腐蚀过程涉及阳极溶解和阴极还原的耦合反应。",
    "The cytochrome c oxidase complex facilitates electron transfer in the mitochondrial respiratory chain.",
    "Perovskite crystal structures exhibit exceptional photovoltaic efficiency due to their direct bandgap properties.",
    "Ribozymes demonstrate catalytic activity through RNA-mediated phosphodiester bond cleavage mechanisms.",
    "The Hardy-Weinberg equilibrium assumes random mating and absence of evolutionary forces in population genetics.",
    "Synchrotron radiation provides high-intensity X-ray beams for protein crystallography studies.",
    "Topological insulators possess gapless surface states protected by time-reversal symmetry.",
    "The Krebs cycle involves substrate-level phosphorylation and NADH production in cellular respiration.",
    "Metamorphic facies classification relies on pressure-temperature conditions during rock formation.",
    "CRISPR-Cas9 systems utilize guide RNA sequences for targeted genomic editing applications.",
    "Galois theory establishes the correspondence between field extensions and group automorphisms.",
    "Holographic interferometry measures nanometer-scale surface deformations using coherent light sources.",
    "The endoplasmic reticulum facilitates protein folding through chaperone-mediated quality control mechanisms.",
    "Superconducting quantum interference devices detect minute magnetic flux variations with unprecedented sensitivity.",
    "Paleomagnetic reversal chronology provides temporal constraints for stratigraphic correlation studies.",
    "Histone deacetylases regulate chromatin structure and gene expression through epigenetic modifications.",
    "The Navier-Stokes equations describe viscous fluid flow in incompressible turbulent systems.",
    "Molecular dynamics simulations predict protein conformational changes using force field parameters.",
    "Radiometric dating employs isotopic decay constants to determine geological time scales.",
    "The Born-Oppenheimer approximation separates nuclear and electronic motion in quantum mechanical calculations.",
    "Transmission electron microscopy reveals ultrastructural details of cellular organelles at nanometer resolution.",
    "量子もつれ現象は非局所的な相関関係を示し、ベルの不等式を破る。",
    "細胞膜のリン脂質二分子膜は選択的透過性を持つ生体バリアを形成する。",
    "超伝導体のクーパー対は格子振動との相互作用により電気抵抗ゼロを実現する。",
    "酵素の基質特異性は活性部位のアミノ酸残基配列によって決定される。",
    "トポロジカル絶縁体の表面状態は時間反転対称性により保護されている。",
    "リボソームの翻訳過程では転移RNAがコドンに対応するアミノ酸を運搬する。",
    "フォトニック結晶の周期構造は特定波長の光に対してバンドギャップを形成する。",
    "プロテアソームはユビキチン化されたタンパク質を選択的に分解する。",
    "ガロア理論は体の拡大と群の自己同型写像の対応関係を明らかにする。",
    "シンクロトロン放射光はタンパク質結晶構造解析に高輝度X線を提供する。",
    "ミトコンドリアの電子伝達系は酸化的リン酸化によりATPを合成する。",
    "変成岩の鉱物組み合わせは圧力温度条件を反映した変成相を示す。",
    "CRISPR-Cas9システムはガイドRNAによる標的配列認識機構を利用する。",
    "ホログラフィ干渉法は可干渉光源を用いてナノメートル級の変形測定を行う。",
    "小胞体ストレス応答は未折り畳みタンパク質の蓄積により活性化される。",
    "超伝導量子干渉素子は磁束量子の変化を極高感度で検出する。",
    "古地磁気極性年代学は地磁気逆転記録から地質年代を決定する。",
    "ヒストン脱アセチル化酵素はクロマチン構造の制御を通じて遺伝子発現を調節する。",
    "ナビエ・ストークス方程式は粘性流体の運動を記述する偏微分方程式である。",
    "分子動力学シミュレーションは力場パラメータを用いてタンパク質の構造変化を予測する。"
]

class FSDEC_Calib(CalibrationDataReader):
    def __init__(self, ref_text:str, ref_audio_path:str, input_texts:List[str]):
        super().__init__()
        self.data_list = None
        encoder_ref_seq, encoder_ref_bert = preprocess_text(ref_text)
        ref_audio_hubert, _, _ = audio_preprocess(ref_audio_path)
        self.saved_data = []
        for input_text in input_texts:
            text_phones, text_bert = preprocess_text(input_text)
            self.saved_data.append({
                'encoder_text_seq': text_phones,
                'encoder_text_bert': text_bert,
                'encoder_ref_seq': encoder_ref_seq,
                'encoder_ref_bert': encoder_ref_bert,
                'encoder_ssl_content': ref_audio_hubert,
            })
        self.saved_data = iter(self.saved_data)
    def rewind(self):
        self.data_list = copy.deepcopy(self.saved_data)
    def get_next(self):
        if self.data_list is None:
            self.rewind()
        return next(self.data_list, None)
    
class SDEC_Calib(CalibrationDataReader):
    def __init__(self, fdec_calib:FSDEC_Calib, fsdec:ort.InferenceSession, sdec:ort.InferenceSession):
        super().__init__()
        self.fdec_calib = fdec_calib
        fdec_calib.rewind()
        self.fsdec = fsdec
        self.sdec = sdec

    def get_next(self):
        next_fsdec = self.fdec_calib.get_next()
        if next_fsdec is None:
            return None
        y, y_emb, *self.present_key_values = self.fsdec.run(None, next_fsdec)
        names = [inp.name for inp in self.sdec.get_inputs()]
        self.end_condition = False

        for _ in tqdm(range(1000), desc="Calibrating SDEC"):
            if self.end_condition:
                break
            self.input_feed = { name:data for name, data in zip(names, [y, y_emb, *self.present_key_values])}
            y, self.end_condition, new_y_emb, *new_key_values = self.sdec.run(None, self.input_feed)
            y_emb = np.concatenate([y_emb, new_y_emb], axis=1)
            for i,(kv, new_kv) in enumerate(zip(self.present_key_values, new_key_values)):
                self.present_key_values[i] = np.concatenate([kv, new_kv], axis=0)
        
        return self.input_feed

def quantize_t2s(fsdec_path, fsdec_quant_path, sdec_path, sdec_quant_path, ref_text, ref_audio_path):

    fsdec_calib = FSDEC_Calib(ref_text, ref_audio_path, CALIB_TEXTs)
    fsdec = ort.InferenceSession(fsdec_path)
    sdec = ort.InferenceSession(sdec_path)
    sdec_calib = SDEC_Calib(fsdec_calib, fsdec, sdec)

    fsdec_to_exclude_names = []
    for node in onnx.load(fsdec_path).graph.node:
        should_exclude = False
        if '/transformer_encoder' not in node.name:
            should_exclude = True
        if node.op_type in ['Gather', 'Softmax','Add','LayerNormalization']:
            should_exclude = True
        if should_exclude:
            fsdec_to_exclude_names.append(node.name)

    print("starting fsdec quantization")
    quantize_static(
        model_input=fsdec_path,
        model_output=fsdec_quant_path,
        calibration_data_reader=fsdec_calib,
        quant_format=QuantFormat.QOperator,  # 使用QOperator格式
        weight_type=QuantType.QInt8,  # 量化权重为8位整数
        activation_type=QuantType.QUInt8,  # 量化激活为8位整数
        per_channel=True,  # 按通道量化
        reduce_range=True,  # 使用减少的量化范围
        # nodes_to_quantize=fsdec_to_quantize_names,
        nodes_to_exclude=fsdec_to_exclude_names,
        extra_options={'ActivationSymmetric': False,
                    'WeightSymmetric': True,}  # 为权重添加QDQ对
    )
    optimize_quantize(fsdec_quant_path, fsdec_quant_path)

    sdec_to_exclude_names = []
    for node in onnx.load(sdec_path).graph.node:
        should_exclude = False
        if '/transformer_encoder' not in node.name:
            should_exclude = True
        if node.op_type in ['Gather','Softmax','Add','LayerNormalization']:
            should_exclude = True
        if should_exclude:
            sdec_to_exclude_names.append(node.name)
    fsdec_calib.rewind()
    print("starting sdec quantization")
    quantize_static(
        model_input=sdec_path,
        model_output=sdec_quant_path,
        calibration_data_reader=sdec_calib,
        quant_format=QuantFormat.QOperator,  # 使用QOperator格式
        weight_type=QuantType.QInt8,  # 量化权重为8位整数
        activation_type=QuantType.QUInt8,  # 量化激活为8位整数
        per_channel=True,  # 按通道量化
        reduce_range=True,  # 使用减少的量化范围
        # nodes_to_quantize=sdec_to_quantize_names,
        nodes_to_exclude=sdec_to_exclude_names,
        extra_options={'ActivationSymmetric': False,
                    'WeightSymmetric': True,}  # 为权重添加QDQ对
    )
    optimize_quantize(sdec_quant_path, sdec_quant_path)

    k_quantizers = []
    v_quantizers = []
    sdec = onnx.load(sdec_quant_path)
    keep_outputs = [o for o in sdec.graph.output if 'increased_k_' not in o.name and 'increased_v_' not in o.name]
    sdec.graph.ClearField("output")   # 清空所有 output
    sdec.graph.output.extend(keep_outputs)  # 重新填充
    for i in range(24):
        k_quantize_param = (get_initializer(sdec, f'present_k_layer_{i}_scale'), get_initializer(sdec, f'present_k_layer_{i}_zero_point'))
        remove_quantize_and_change_input(sdec.graph, f'past_k_layer_{i}', f'past_k_layer_{i}_QuantizeLinear')
        k_quantizers.append(k_quantize_param)
        set_initializer_value(sdec, f'increased_k_layer_{i}_scale', k_quantize_param[0])
        set_initializer_value(sdec, f'increased_k_layer_{i}_zero_point', k_quantize_param[1])
        k_vi = onnx.helper.make_tensor_value_info(
            f'increased_k_layer_{i}_quantized',
            onnx.TensorProto.UINT8,  
            [1, 1, 512]                 
        )
        sdec.graph.output.append(k_vi)
        v_quantize_param = (get_initializer(sdec, f'present_v_layer_{i}_scale'), get_initializer(sdec, f'present_v_layer_{i}_zero_point'))
        remove_quantize_and_change_input(sdec.graph, f'past_v_layer_{i}', f'past_v_layer_{i}_QuantizeLinear')
        v_quantizers.append(v_quantize_param)
        set_initializer_value(sdec, f'increased_v_layer_{i}_scale', v_quantize_param[0])
        set_initializer_value(sdec, f'increased_v_layer_{i}_zero_point', v_quantize_param[1])
        v_vi = onnx.helper.make_tensor_value_info(
            f'increased_v_layer_{i}_quantized',
            onnx.TensorProto.UINT8,   
            [1, 1, 512]                 
        )
        sdec.graph.output.append(v_vi)
    onnx.checker.check_model(sdec)
    onnx.save(sdec, sdec_quant_path)

    fsdec = onnx.load(fsdec_quant_path)
    keep_outputs = [o for o in fsdec.graph.output if 'present_' not in o.name]
    fsdec.graph.ClearField("output")   # 清空所有 output
    fsdec.graph.output.extend(keep_outputs)  # 重新填充
    for i in range(24):
        k_scale = k_quantizers[i][0]
        k_zp = k_quantizers[i][1]
        set_initializer_value(fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_1_output_0_scale", k_scale)
        set_initializer_value(fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_1_output_0_zero_point", k_zp)
        quantizer_node = find_node_by_op_name(fsdec,'QuantizeLinear', f'/transformer_encoder/layers.{i}/self_attn/Unsqueeze_1_output_0_QuantizeLinear')
        following_node = find_nodes_children(fsdec, quantizer_node)[0]
        assert(quantizer_node.output[0] == following_node.input[0])
        quantizer_node.output[0] = f'present_k_layer_{i}_quantized'
        following_node.input[0] = f'present_k_layer_{i}_quantized'
        k_vi = onnx.helper.make_tensor_value_info(
            f'present_k_layer_{i}_quantized',
            onnx.TensorProto.UINT8,  
            [None, 1, 512]        
        )
        fsdec.graph.output.append(k_vi)

        v_scale = v_quantizers[i][0]
        v_zp = v_quantizers[i][1]
        set_initializer_value(fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_2_output_0_scale", v_scale)
        set_initializer_value(fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_2_output_0_zero_point", v_zp)
        quantizer_node = find_node_by_op_name(fsdec,'QuantizeLinear', f'/transformer_encoder/layers.{i}/self_attn/Unsqueeze_2_output_0_QuantizeLinear')
        following_node = find_nodes_children(fsdec, quantizer_node)[0]
        assert(quantizer_node.output[0] == following_node.input[0])
        quantizer_node.output[0] = f'present_v_layer_{i}_quantized'
        following_node.input[0] = f'present_v_layer_{i}_quantized'
        v_vi = onnx.helper.make_tensor_value_info(
            f'present_v_layer_{i}_quantized',
            onnx.TensorProto.UINT8,   
            [None, 1, 512]                 
        )
        fsdec.graph.output.append(v_vi)
    onnx.checker.check_model(fsdec)
    onnx.save(fsdec, fsdec_quant_path)


if __name__ == "__main__":
    fsdec_path = 'onnx/v2pp/t2s/t2s_fsdec.onnx'
    fsdec_quant_path = 'onnx/v2pp/t2s/t2s_fsdec_quant.onnx'
    sdec_path = 'onnx/v2pp/t2s/t2s_sdec.onnx'
    sdec_quant_path = 'onnx/v2pp/t2s/t2s_sdec_quant.onnx'
    ref_text = 'あなたと空を見上げるのは、いつも夏でしたわね'
    ref_audio_path = '/home/eleven/GPT-SoVITS/playground/(A)あなたと空を見上げるのは、いつも夏でしたわね.wav'
    quantize_t2s(fsdec_path, fsdec_quant_path, sdec_path, sdec_quant_path, ref_text, ref_audio_path)