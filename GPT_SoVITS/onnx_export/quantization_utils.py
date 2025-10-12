import onnx
import numpy as np
import onnxsim

def find_node_by_op_name(onnx_model:onnx.ModelProto, op_type:str, name:str) -> onnx.NodeProto:
    """
    Find a node in the ONNX model by its operation type and name.
    """
    for node in onnx_model.graph.node:
        if node.op_type is not None and node.op_type == op_type and node.name == name:
            return node
        elif node.name == name:
            return node
    raise ValueError(f"Node with op_type {op_type} and name {name} not found.")

def find_nodes_parent(onnx_model:onnx.ModelProto, target_node:onnx.NodeProto) -> list[onnx.NodeProto]:
    """
    Find parent nodes of the given target node in the ONNX model.
    """
    parents = []
    target_inputs = set(target_node.input)
    for node in onnx_model.graph.node:
        if any(output in target_inputs for output in node.output):
            parents.append(node)
    if len(parents) == 0:
        raise ValueError(f"No parent nodes found for node {target_node.name}.")
    return parents

def find_nodes_children(onnx_model:onnx.ModelProto, target_node:onnx.NodeProto) -> list[onnx.NodeProto]:
    """
    Find child nodes of the given target node in the ONNX model.
    """
    children = []
    target_outputs = set(target_node.output)
    for node in onnx_model.graph.node:
        if any(input in target_outputs for input in node.input):
            children.append(node)
    if len(children) == 0:
        raise ValueError(f"No child nodes found for node {target_node.name}.")
    return children

def swtich_nodes(onnx_model:onnx.ModelProto, node1:onnx.NodeProto, node2:onnx.NodeProto) -> onnx.ModelProto:
    """
    Switch the positions of two nodes in the ONNX model.
    """

    # Switch connections
    assert(node1.output[0] == node2.input[0])
    inter_node_connection: str = node1.output[0]+"_switched"
    node2.input[0] = node1.input[0]
    node1.output[0] = node2.output[0]
    node1.input[0] = inter_node_connection
    node2.output[0] = inter_node_connection
    
    # 2. 交换 graph.node 里的顺序 (保持拓扑一致)
    nodes = list(onnx_model.graph.node)
    index1 = nodes.index(node1)
    index2 = nodes.index(node2)

    if index2 > index1:
        nodes.pop(index2)
        nodes.insert(index1, node2)
    elif index1 > index2:
        nodes.pop(index1)
        nodes.insert(index2, node1)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(nodes)

    # 3. 保存修改后的模型
    onnx.checker.check_model(onnx_model)

    return onnx_model

def optimize_quantize(input_model_path:str, output_model_path:str):
    """
    Optimize the ONNX model by switching QuantizeLinear/DequantizeLinear nodes with adjacent nodes.
    """
    onnx_model = onnx.load(input_model_path)
    exchangable_nodes = ['Unsqueeze', 'Reshape', 'Transpose', 'Squeeze']
    quantizel_linear_should_skip = []
    while True:
        modified = False
        for node in onnx_model.graph.node:
            if node.op_type == 'QuantizeLinear' and node.name not in quantizel_linear_should_skip:
                try:
                    parents = find_nodes_parent(onnx_model, node)
                except ValueError:
                    quantizel_linear_should_skip.append(node.name)
                    continue
                if len(parents) != 1:
                    quantizel_linear_should_skip.append(node.name)
                    continue
                parent = parents[0]
                if parent.op_type in exchangable_nodes:
                    children1 = find_nodes_children(onnx_model, parent)
                    if len(children1) != 1:
                        quantizel_linear_should_skip.append(node.name)
                        continue
                    parents2 = find_nodes_parent(onnx_model, node)
                    if len(parents2) != 1:
                        quantizel_linear_should_skip.append(node.name)
                        continue
                    print(f"Switching {parent.name} and {node.name}")
                    onnx_model = swtich_nodes(onnx_model, parent, node)
                    modified = True
                    break
        if not modified:
            break

    dequant_linear_should_skip = []
    while True:
        modified = False
        for node in onnx_model.graph.node:
            if node.op_type == 'DequantizeLinear' and node.name not in dequant_linear_should_skip:
                try:
                    children = find_nodes_children(onnx_model, node)
                except ValueError:
                    dequant_linear_should_skip.append(node.name)
                    continue
                if len(children) != 1:
                    dequant_linear_should_skip.append(node.name)
                    continue
                child = children[0]
                if child.op_type in exchangable_nodes:
                    children1 = find_nodes_children(onnx_model, node)
                    if len(children1) != 1:
                        dequant_linear_should_skip.append(node.name)
                        continue
                    parents2 = find_nodes_parent(onnx_model, child)
                    if len(parents2) != 1:
                        dequant_linear_should_skip.append(node.name)
                        continue
                    print(f"Switching {node.name} and {child.name}")
                    onnx_model = swtich_nodes(onnx_model, node, child)
                    modified = True
                    break
        if not modified:
            break
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_model_path)

def remove_quantize_and_change_input(graph:onnx.GraphProto, input_name:str, qlinear_name:str):
    def safe_remove_initializer(graph, name):
        # 检查是否还被别的节点使用
        for node in graph.node:
            if name in node.input:
                return  # 还在用，不能删
        # 检查 graph.input / graph.output
        for inp in graph.input:
            if inp.name == name:
                return
        for out in graph.output:
            if out.name == name:
                return
        # 确认没用再删除
        for init in graph.initializer:
            if init.name == name:
                graph.initializer.remove(init)
                break
    # 1. 找到 QuantizeLinear 节点
    qnode = None
    for node in graph.node:
        if node.name == qlinear_name:
            qnode = node
            break
    if qnode is None:
        raise ValueError(f"QuantizeLinear node {qlinear_name} not found")

    # 2. 获取输入、输出
    q_in = qnode.input[0]    # 原始输入 (float32)
    q_out = qnode.output[0]  # 量化后的输出 (uint8)

    # 3. 修改后续节点，把 q_out 替换成 q_in
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == q_out:
                node.input[i] = q_in

    # 4. 删除 qnode
    graph.node.remove(qnode)

    safe_remove_initializer(graph, qnode.input[1])  # scale
    safe_remove_initializer(graph, qnode.input[2])  # zero_point

    # 5. 修改 graph.input 数据类型为 uint8
    for i, inp in enumerate(graph.input):
        if inp.name == input_name:
            old_shape = inp.type.tensor_type.shape
            new_input = onnx.helper.make_tensor_value_info(
                name=input_name,
                elem_type=onnx.TensorProto.UINT8,  # 改成 uint8
                shape=['batch']+[d.dim_value for d in old_shape.dim[-2:]]
            )
            graph.input.remove(inp)
            graph.input.insert(i, new_input)
            break

def get_initializer(model:onnx.ModelProto, name:str):
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None

def set_initializer_value(model:onnx.ModelProto, name:str, array:np.ndarray):
    for init in model.graph.initializer:
        if init.name == name:
            init.CopyFrom(onnx.numpy_helper.from_array(array, init.name))
            return
    # 如果没有找到对应的 initializer，则添加一个新的
    new_init = onnx.numpy_helper.from_array(array, name)
    model.graph.initializer.append(new_init)

def is_constant(model:onnx.ModelProto, tensor_name:str) -> bool:
    """
    Check if a tensor is a constant (from initializer).
    """
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return True
    return False

def get_tensor_shape(model:onnx.ModelProto, tensor_name:str) -> list[int]:
    """
    Get the shape of a tensor from initializer or value info.
    """
    # Check initializer first
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)

    # Check value info
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Unknown dimension
            return shape

    # Check input
    for inp in model.graph.input:
        if inp.name == tensor_name:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Unknown dimension
            return shape

    return None