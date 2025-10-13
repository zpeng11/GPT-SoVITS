import copy
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common.float16 import convert_float_to_float16
from onnxsim import simplify
from onnxruntime.quantization.quantize import (
    quantize_dynamic, quantize_static, CalibrationDataReader,
    QuantFormat, QuantType
)
from tqdm import tqdm

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocess_utils import preprocess_text, audio_preprocess
from quantization_utils import (
    optimize_quantize, remove_quantize_and_change_input, get_initializer,
    set_initializer_value, find_node_by_op_name, find_nodes_children
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CALIBRATION_SEQUENCE_LENGTH = 1000
QUANTIZATION_TRANSFORMER_LAYERS = 24
QUANTIZATION_DIMENSION = 512
EXCLUDED_NODE_TYPES = ['Gather', 'Softmax', 'Add', 'LayerNormalization']

# Calibration texts organized by language for comprehensive quantization testing
CALIBRATION_TEXTS = {
    "chinese_daily": [
        "今天的天气格外晴朗，阳光透过窗户洒在桌案上。",
        "她正在图书馆里专心致志地阅读一本关于人工智能的书籍。",
        "这家餐厅的招牌菜是麻婆豆腐，味道非常地道。",
        "小猫咪懒洋洋地躺在沙发上，享受着午后的阳光。",
        "他每天早上都会去公园里跑步锻炼身体。",
        "春天来了，樱花盛开，整个城市都被粉色的花瓣装点着。",
        "学生们正在教室里认真听老师讲解数学公式。",
        "这部电影的剧情跌宕起伏，让观众看得津津有味。",
        "奶奶在厨房里忙碌着，准备一桌丰盛的晚餐。",
        "夜晚的星空格外美丽，繁星点点闪烁着微弱的光芒。"
    ],
    "chinese_technical": [
        "量子纠缠现象在超低温环境下表现出非局域性关联特征。",
        "胞间连丝的结构复杂性影响了植物细胞间的物质传输效率。",
        "拓扑绝缘体的能带结构在狄拉克锥附近呈现线性色散关系。",
        "古生物学家在寒武纪地层中发现了三叶虫的钙化外骨骼化石。",
        "蛋白质的二级结构主要由氢键维持其α螺旋和β折叠构象。",
        "傅里叶变换在信号处理中用于频域分析和滤波器设计。",
        "星系团中的暗物质分布通过引力透镜效应得以间接观测。",
        "酶的活性位点通过诱导契合机制与底物分子结合。",
        "黎曼几何中的测地线是连接两点间距离最短的曲线。",
        "电化学腐蚀过程涉及阳极溶解和阴极还原的耦合反应。"
    ],
    "english_technical": [
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
        "Transmission electron microscopy reveals ultrastructural details of cellular organelles at nanometer resolution."
    ],
    "japanese_technical": [
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
}

# Flatten all calibration texts for backward compatibility
CALIB_TEXTs = [text for texts in CALIBRATION_TEXTS.values() for text in texts]


def get_nodes_to_exclude(model_path: str) -> List[str]:
    """
    Get list of nodes to exclude from quantization

    Args:
        model_path: Path to ONNX model file

    Returns:
        List of node names to exclude from quantization
    """
    exclude_names = []
    model = onnx.load(model_path)

    for node in model.graph.node:
        should_exclude = (
            '/transformer_encoder' not in node.name or
            node.op_type in EXCLUDED_NODE_TYPES
        )
        if should_exclude:
            exclude_names.append(node.name)

    logger.info(f"Found {len(exclude_names)} nodes to exclude from quantization")
    return exclude_names


def quantize_model_static(model_path: str, quantized_path: str,
                         calibration_reader: CalibrationDataReader,
                         exclude_nodes: List[str]) -> None:
    """
    Perform static quantization on ONNX model

    Args:
        model_path: Path to input ONNX model
        quantized_path: Path to save quantized model
        calibration_reader: Calibration data reader
        exclude_nodes: List of nodes to exclude from quantization
    """
    logger.info(f"Starting static quantization: {model_path} -> {quantized_path}")

    quantize_static(
        model_input=model_path,
        model_output=quantized_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=True,
        nodes_to_exclude=exclude_nodes,
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    )

    # Optimize quantized model
    optimize_quantize(quantized_path, quantized_path)
    logger.info(f"Static quantization completed: {quantized_path}")


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

def plugin_sampling_parameters(t2s_sdec_path: str, is_quantized: bool):
    """
    Plugin sampling parameters into the T2S stage decoder model

    Args:
        t2s_sdec_path: Path to the T2S stage decoder ONNX model
    """
    model = onnx.load(t2s_sdec_path)

    # Define new inputs for temperature, top_k, and top_p
    temperature_input = onnx.helper.make_tensor_value_info('temperature', onnx.TensorProto.FLOAT if is_quantized else onnx.TensorProto.FLOAT16, [1])
    top_k_input = onnx.helper.make_tensor_value_info('top_k', onnx.TensorProto.INT64, [1])
    repeat_penalty_input = onnx.helper.make_tensor_value_info('repeat_penalty', onnx.TensorProto.FLOAT if is_quantized else onnx.TensorProto.FLOAT16, [1])

    inputs = list(model.graph.input)
    inputs = inputs[:2] + [temperature_input, top_k_input, repeat_penalty_input] + inputs[2:]
    del model.graph.input[:]
    model.graph.input.extend(inputs)

    div_node = find_node_by_op_name(model, 'Div', '/Div')
    div_node.input[1] = 'repeat_penalty'

    mul_node = find_node_by_op_name(model, 'Mul', '/Mul')
    mul_node.input[1] = 'repeat_penalty'

    topk_node = find_node_by_op_name(model, 'TopK', '/TopK')
    topk_node.input[1] = 'top_k'

    where_node = find_node_by_op_name(model, 'Where', '/Where_1')
    connection_output = where_node.output[0]
    where_node.output[0] = 'temperature_input'

    temperature_input_vi = onnx.helper.make_tensor_value_info('temperature_input', onnx.TensorProto.FLOAT if is_quantized else onnx.TensorProto.FLOAT16, [1025])

    temperature_div = onnx.helper.make_node(
        'Div',
        inputs=['temperature_input', 'temperature'],
        outputs=[connection_output],
        name='/Temperature/Div'
    )

    model.graph.value_info.extend([temperature_input_vi])

    # Find insertion point and add div node
    insertion_index = 0
    for i, node in enumerate(model.graph.node):
        if node.name == '/Softmax':
            insertion_index = i
            break
    model.graph.node.insert(insertion_index, temperature_div)

    initializers_to_keep = []
    for init in model.graph.initializer:
        if init.name not in ['/Reshape_output_0', '/Constant_13_output_0']:
            initializers_to_keep.append(init)

    # Clear and rebuild initializer list
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers_to_keep)

    onnx.checker.check_model(model)
    onnx.save(model, t2s_sdec_path)

def quantize_t2s(fsdec_path: str, fsdec_quant_path: str,
                sdec_path: str, sdec_quant_path: str,
                ref_text: str, ref_audio_path: str) -> None:
    """
    Quantize T2S models (FSDEC and SDEC) for mobile inference

    Args:
        fsdec_path: Path to FSDEC ONNX model
        fsdec_quant_path: Path to save quantized FSDEC model
        sdec_path: Path to SDEC ONNX model
        sdec_quant_path: Path to save quantized SDEC model
        ref_text: Reference text for calibration
        ref_audio_path: Path to reference audio file
    """
    logger.info("Starting T2S model quantization pipeline")

    # Initialize calibration readers
    fsdec_calib = FSDEC_Calib(ref_text, ref_audio_path, CALIB_TEXTs)
    fsdec = ort.InferenceSession(fsdec_path)
    sdec = ort.InferenceSession(sdec_path)
    sdec_calib = SDEC_Calib(fsdec_calib, fsdec, sdec)

    # Quantize FSDEC model
    print("starting fsdec quantization")
    fsdec_exclude_nodes = get_nodes_to_exclude(fsdec_path)
    quantize_model_static(fsdec_path, fsdec_quant_path, fsdec_calib, fsdec_exclude_nodes)

    # Quantize SDEC model
    print("starting sdec quantization")
    sdec_exclude_nodes = get_nodes_to_exclude(sdec_path)
    fsdec_calib.rewind()
    quantize_model_static(sdec_path, sdec_quant_path, sdec_calib, sdec_exclude_nodes)

    plugin_sampling_parameters(sdec_quant_path, is_quantized=True)

    # Configure model outputs for cross-model quantization
    k_quantizers, v_quantizers = _configure_quantizer_outputs(
        sdec_quant_path, fsdec_quant_path
    )

    logger.info("T2S model quantization pipeline completed successfully")


def _configure_quantizer_outputs(sdec_quant_path: str, fsdec_quant_path: str) -> tuple:
    """
    Configure quantizer outputs for cross-model quantization compatibility

    Args:
        sdec_quant_path: Path to quantized SDEC model
        fsdec_quant_path: Path to quantized FSDEC model

    Returns:
        Tuple of (k_quantizers, v_quantizers) for cross-model compatibility
    """
    k_quantizers = []
    v_quantizers = []

    # Configure SDEC model outputs
    sdec = onnx.load(sdec_quant_path)
    keep_outputs = [
        o for o in sdec.graph.output
        if 'increased_k_' not in o.name and 'increased_v_' not in o.name
    ]
    sdec.graph.ClearField("output")
    sdec.graph.output.extend(keep_outputs)

    for i in range(QUANTIZATION_TRANSFORMER_LAYERS):
        # Configure key quantizer
        k_quantize_param = (
            get_initializer(sdec, f'present_k_layer_{i}_scale'),
            get_initializer(sdec, f'present_k_layer_{i}_zero_point')
        )
        remove_quantize_and_change_input(
            sdec.graph, f'past_k_layer_{i}',
            f'past_k_layer_{i}_QuantizeLinear'
        )
        k_quantizers.append(k_quantize_param)
        set_initializer_value(sdec, f'increased_k_layer_{i}_scale', k_quantize_param[0])
        set_initializer_value(sdec, f'increased_k_layer_{i}_zero_point', k_quantize_param[1])

        k_vi = onnx.helper.make_tensor_value_info(
            f'increased_k_layer_{i}_quantized',
            onnx.TensorProto.UINT8,
            [1, 1, QUANTIZATION_DIMENSION]
        )
        sdec.graph.output.append(k_vi)

        # Configure value quantizer
        v_quantize_param = (
            get_initializer(sdec, f'present_v_layer_{i}_scale'),
            get_initializer(sdec, f'present_v_layer_{i}_zero_point')
        )
        remove_quantize_and_change_input(
            sdec.graph, f'past_v_layer_{i}',
            f'past_v_layer_{i}_QuantizeLinear'
        )
        v_quantizers.append(v_quantize_param)
        set_initializer_value(sdec, f'increased_v_layer_{i}_scale', v_quantize_param[0])
        set_initializer_value(sdec, f'increased_v_layer_{i}_zero_point', v_quantize_param[1])

        v_vi = onnx.helper.make_tensor_value_info(
            f'increased_v_layer_{i}_quantized',
            onnx.TensorProto.UINT8,
            [1, 1, QUANTIZATION_DIMENSION]
        )
        sdec.graph.output.append(v_vi)

    onnx.checker.check_model(sdec)
    onnx.save(sdec, sdec_quant_path)

    # Configure FSDEC model outputs to match SDEC
    fsdec = onnx.load(fsdec_quant_path)
    keep_outputs = [o for o in fsdec.graph.output if 'present_' not in o.name]
    fsdec.graph.ClearField("output")
    fsdec.graph.output.extend(keep_outputs)

    for i in range(QUANTIZATION_TRANSFORMER_LAYERS):
        k_scale, k_zp = k_quantizers[i]
        v_scale, v_zp = v_quantizers[i]

        # Configure key quantizer
        set_initializer_value(
            fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_1_output_0_scale", k_scale
        )
        set_initializer_value(
            fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_1_output_0_zero_point", k_zp
        )
        quantizer_node = find_node_by_op_name(
            fsdec, 'QuantizeLinear',
            f'/transformer_encoder/layers.{i}/self_attn/Unsqueeze_1_output_0_QuantizeLinear'
        )
        following_node = find_nodes_children(fsdec, quantizer_node)[0]
        assert quantizer_node.output[0] == following_node.input[0]
        quantizer_node.output[0] = f'present_k_layer_{i}_quantized'
        following_node.input[0] = f'present_k_layer_{i}_quantized'

        k_vi = onnx.helper.make_tensor_value_info(
            f'present_k_layer_{i}_quantized',
            onnx.TensorProto.UINT8,
            [None, 1, QUANTIZATION_DIMENSION]
        )
        fsdec.graph.output.append(k_vi)

        # Configure value quantizer
        set_initializer_value(
            fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_2_output_0_scale", v_scale
        )
        set_initializer_value(
            fsdec, f"/transformer_encoder/layers.{i}/self_attn/Unsqueeze_2_output_0_zero_point", v_zp
        )
        quantizer_node = find_node_by_op_name(
            fsdec, 'QuantizeLinear',
            f'/transformer_encoder/layers.{i}/self_attn/Unsqueeze_2_output_0_QuantizeLinear'
        )
        following_node = find_nodes_children(fsdec, quantizer_node)[0]
        assert quantizer_node.output[0] == following_node.input[0]
        quantizer_node.output[0] = f'present_v_layer_{i}_quantized'
        following_node.input[0] = f'present_v_layer_{i}_quantized'

        v_vi = onnx.helper.make_tensor_value_info(
            f'present_v_layer_{i}_quantized',
            onnx.TensorProto.UINT8,
            [None, 1, QUANTIZATION_DIMENSION]
        )
        fsdec.graph.output.append(v_vi)

    onnx.checker.check_model(fsdec)
    onnx.save(fsdec, fsdec_quant_path)

    return k_quantizers, v_quantizers

def get_fp16_block_list(onnx_model: onnx.ModelProto) -> List[str]:
    """
    Get list of nodes that should be converted to FP16 precision

    Args:
        onnx_model: ONNX model to analyze

    Returns:
        List of node names to include in FP16 conversion
    """
    node_block_list = [node.name for node in onnx_model.graph.node]
    node_block_list_new = []

    for node_name in node_block_list:
        # Skip nodes not in transformer encoder or specific operations
        if ('/transformer_encoder' not in node_name and
            node_name not in ['/Gather', '/ar_predict_layer/MatMul']):
            continue

        # Check if node should be excluded based on layer-specific patterns
        should_exclude = False
        for i in range(QUANTIZATION_TRANSFORMER_LAYERS):
            layer_prefix = f'/transformer_encoder/layers.{i}/self_attn/'
            exclude_patterns = [
                'Add', 'Slice', 'Slice_1', 'Slice_2',
                'Concat', 'Concat_1', 'Reshape', 'Reshape_1',
                'Reshape_2', 'Reshape_3', 'Transpose_1', 'Transpose_2',
                'Transpose_3', 'Transpose_5', 'Unsqueeze', 'Unsqueeze_1',
                'Unsqueeze_2', 'Mul_3', 'Mul_4', 'MatMul_1',
                'MatMul_2', 'Softmax'
            ]

            if node_name in [layer_prefix + pattern for pattern in exclude_patterns]:
                should_exclude = True
                break

        if not should_exclude:
            node_block_list_new.append(node_name)

    logger.info(f"Found {len(node_block_list_new)} nodes for FP16 conversion")
    return node_block_list_new


def remove_redundant_cast(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Remove redundant cast operations to optimize the model

    Args:
        onnx_model: ONNX model to optimize

    Returns:
        Optimized ONNX model
    """
    logger.info("Removing redundant cast operations")

    # Fix input connections for layer 0
    find_node_by_op_name(onnx_model, 'Add', '/transformer_encoder/layers.0/Add').input[0] = \
        '/transformer_encoder/layers.0/self_attn/MatMul_input_cast_0'

    # Fix connections for subsequent layers
    for i in range(QUANTIZATION_TRANSFORMER_LAYERS):
        if i != 0:
            prev_layer = i - 1
            find_node_by_op_name(onnx_model, 'Add', f'/transformer_encoder/layers.{i}/Add').input[0] = \
                f'/transformer_encoder/layers.{prev_layer}/norm2/LayerNormalization_output_cast_0'
            find_node_by_op_name(onnx_model, 'MatMul', f'/transformer_encoder/layers.{i}/self_attn/MatMul').input[0] = \
                f'/transformer_encoder/layers.{prev_layer}/norm2/LayerNormalization_output_cast_0'

        # Fix current layer connections
        find_node_by_op_name(onnx_model, 'MatMul', f'/transformer_encoder/layers.{i}/linear1/MatMul').input[0] = \
            f'/transformer_encoder/layers.{i}/norm1/LayerNormalization_output_cast_0'
        find_node_by_op_name(onnx_model, 'MatMul', f'/transformer_encoder/layers.{i}/Add_1').input[0] = \
            f'/transformer_encoder/layers.{i}/norm1/LayerNormalization_output_cast_0'

    # Simplify the model after removing redundant casts
    onnx_model, _ = simplify(onnx_model)
    logger.info("Redundant cast operations removed")

    return onnx_model


def convert_random_normal_like_to_fp16(model: onnx.ModelProto, node_name: str) -> onnx.ModelProto:
    """
    Convert a RandomNormalLike node to output FP16

    Args:
        model: ONNX model to modify
        node_name: Name of the RandomNormalLike node to convert

    Returns:
        Modified ONNX model
    """
    for node in model.graph.node:
        if node.name == node_name and node.op_type == 'RandomNormalLike':
            # Check if dtype attribute exists
            dtype_attr_found = False
            for attr in node.attribute:
                if attr.name == 'dtype':
                    attr.i = onnx.TensorProto.FLOAT16
                    dtype_attr_found = True
                    break

            # If no dtype attribute, add one
            if not dtype_attr_found:
                dtype_attr = onnx.helper.make_attribute('dtype', onnx.TensorProto.FLOAT16)
                node.attribute.append(dtype_attr)

            logger.info(f"Converted {node_name} to FP16 output")
            return model

    logger.warning(f"Node {node_name} not found or not RandomNormalLike")
    return model


def t2s_sdec_fp16_dynamic_quant(input_model_path: str, output_model_path: str) -> None:
    """
    Apply FP16 conversion and dynamic quantization to T2S SDEC model

    Args:
        input_model_path: Path to input ONNX model
        output_model_path: Path to save quantized model
    """
    logger.info(f"Starting FP16 dynamic quantization: {input_model_path} -> {output_model_path}")

    # Load model
    onnx_model = onnx.load(input_model_path)

    # Convert to FP16 with selective node conversion
    node_block_list = get_fp16_block_list(onnx_model)
    onnx_model_fp16 = convert_float_to_float16(
        onnx_model, keep_io_types=False, node_block_list=node_block_list
    )

    # Convert RandomNormalLike nodes to FP16
    onnx_model_fp16 = convert_random_normal_like_to_fp16(onnx_model_fp16, '/RandomNormalLike')

    # Remove redundant cast operations
    onnx_model_fp16 = remove_redundant_cast(onnx_model_fp16)

    # Apply dynamic quantization
    quantize_dynamic(
        onnx_model_fp16, output_model_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Attention', 'Conv', 'Gemm'],
        nodes_to_exclude=['/ar_predict_layer/MatMul'],
        per_channel=True,
        reduce_range=True
    )

    plugin_sampling_parameters(output_model_path, is_quantized=False)

    logger.info(f"FP16 dynamic quantization completed: {output_model_path}")

def main() -> None:
    """Main execution function for testing quantization"""
    fsdec_path = 'onnx/v2pp/t2s/t2s_fsdec.onnx'
    fsdec_quant_path = 'onnx/v2pp/t2s/t2s_fsdec_quant.onnx'
    sdec_path = 'onnx/v2pp/t2s/t2s_sdec.onnx'
    sdec_quant_path = 'onnx/v2pp/t2s/t2s_sdec_quant.onnx'
    ref_text = 'あなたと空を見上げるのは、いつも夏でしたわね'
    ref_audio_path = '/home/eleven/GPT-SoVITS/playground/(A)あなたと空を見上げるのは、いつも夏でしたわね.wav'

    try:
        quantize_t2s(fsdec_path, fsdec_quant_path, sdec_path, sdec_quant_path, ref_text, ref_audio_path)
        logger.info("✅ Quantization test completed successfully")
    except Exception as e:
        logger.error(f"❌ Quantization test failed: {e}")


if __name__ == "__main__":
    main()