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

def _build_node_lookup_tables(onnx_model: onnx.ModelProto) -> tuple[dict, dict]:
    """
    Build lookup tables for efficient parent/child node searches.
    Returns (parents_dict, children_dict) where:
    - parents_dict[node_name] = list of parent nodes
    - children_dict[node_name] = list of child nodes
    """
    parents = {}
    children = {}
    output_to_node = {}

    # Initialize dictionaries
    for node in onnx_model.graph.node:
        parents[node.name] = []
        children[node.name] = []
        for output in node.output:
            output_to_node[output] = node.name

    # Build relationships
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in output_to_node:
                parent_name = output_to_node[input_name]
                parents[node.name].append(parent_name)
                children[parent_name].append(node.name)

    return parents, children

def find_nodes_parent(onnx_model:onnx.ModelProto, target_node:onnx.NodeProto) -> list[onnx.NodeProto]:
    """
    Find parent nodes of the given target node in the ONNX model.
    """
    parents, children = _build_node_lookup_tables(onnx_model)
    parent_names = parents.get(target_node.name, [])
    if not parent_names:
        raise ValueError(f"No parent nodes found for node {target_node.name}.")

    node_dict = {node.name: node for node in onnx_model.graph.node}
    return [node_dict[name] for name in parent_names]

def find_nodes_children(onnx_model:onnx.ModelProto, target_node:onnx.NodeProto) -> list[onnx.NodeProto]:
    """
    Find child nodes of the given target node in the ONNX model.
    """
    parents, children = _build_node_lookup_tables(onnx_model)
    child_names = children.get(target_node.name, [])
    if not child_names:
        raise ValueError(f"No child nodes found for node {target_node.name}.")

    node_dict = {node.name: node for node in onnx_model.graph.node}
    return [node_dict[name] for name in child_names]

def switch_nodes(onnx_model:onnx.ModelProto, node1:onnx.NodeProto, node2:onnx.NodeProto) -> onnx.ModelProto:
    """
    Switch the positions of two nodes in the ONNX model while maintaining topological order.
    """
    # Validate connections
    if not node1.output or not node2.input:
        raise ValueError("Node1 must have output and Node2 must have input")
    if node1.output[0] != node2.input[0]:
        raise ValueError(f"Node1 output '{node1.output[0]}' must match Node2 input '{node2.input[0]}'")

    # Switch connections (exactly as original function did)
    inter_node_connection = node1.output[0] + "_switched"
    node2.input[0] = node1.input[0]
    node1.output[0] = node2.output[0]
    if node1.input:
        node1.input[0] = inter_node_connection
    node2.output[0] = inter_node_connection

    # Switch node positions in graph (exactly as original function did)
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

    return onnx_model

def optimize_quantize(input_model_path:str, output_model_path:str):
    """
    Optimize the ONNX model by switching QuantizeLinear/DequantizeLinear nodes with adjacent nodes.
    """
    onnx_model = onnx.load(input_model_path)
    exchangable_nodes = {'Unsqueeze', 'Reshape', 'Transpose', 'Squeeze'}

    # Build lookup tables once for efficiency
    parents_dict, children_dict = _build_node_lookup_tables(onnx_model)
    node_dict = {node.name: node for node in onnx_model.graph.node}

    # Track which nodes to skip to avoid repeated failed attempts
    quantize_skip = set()
    dequantize_skip = set()

    max_iterations = len(onnx_model.graph.node) * 2  # Prevent infinite loops
    iteration = 0

    # Optimize QuantizeLinear nodes
    while iteration < max_iterations:
        modified = False
        iteration += 1

        for node in onnx_model.graph.node:
            if node.op_type != 'QuantizeLinear' or node.name in quantize_skip:
                continue

            parent_names = parents_dict.get(node.name, [])
            if len(parent_names) != 1:
                quantize_skip.add(node.name)
                continue

            parent = node_dict.get(parent_names[0])
            if not parent or parent.op_type not in exchangable_nodes:
                quantize_skip.add(node.name)
                continue

            child_names = children_dict.get(parent.name, [])
            if len(child_names) != 1:
                quantize_skip.add(node.name)
                continue

            parent_names_of_node = parents_dict.get(node.name, [])
            if len(parent_names_of_node) != 1:
                quantize_skip.add(node.name)
                continue

            # print(f"Switching {parent.name} and {node.name}")
            onnx_model = switch_nodes(onnx_model, parent, node)

            # Rebuild lookup tables after modification
            parents_dict, children_dict = _build_node_lookup_tables(onnx_model)
            node_dict = {node.name: node for node in onnx_model.graph.node}

            modified = True
            break

        if not modified:
            break

    # Optimize DequantizeLinear nodes
    iteration = 0
    while iteration < max_iterations:
        modified = False
        iteration += 1

        for node in onnx_model.graph.node:
            if node.op_type != 'DequantizeLinear' or node.name in dequantize_skip:
                continue

            child_names = children_dict.get(node.name, [])
            if len(child_names) != 1:
                dequantize_skip.add(node.name)
                continue

            child = node_dict.get(child_names[0])
            if not child or child.op_type not in exchangable_nodes:
                dequantize_skip.add(node.name)
                continue

            children_of_node = children_dict.get(node.name, [])
            if len(children_of_node) != 1:
                dequantize_skip.add(node.name)
                continue

            parents_of_child = parents_dict.get(child.name, [])
            if len(parents_of_child) != 1:
                dequantize_skip.add(node.name)
                continue

            # print(f"Switching {node.name} and {child.name}")
            onnx_model = switch_nodes(onnx_model, node, child)

            # Rebuild lookup tables after modification
            parents_dict, children_dict = _build_node_lookup_tables(onnx_model)
            node_dict = {node.name: node for node in onnx_model.graph.node}

            modified = True
            break

        if not modified:
            break

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_model_path)

def remove_quantize_and_change_input(graph:onnx.GraphProto, input_name:str, qlinear_name:str):
    """
    Remove QuantizeLinear node and change input data type to uint8.
    """
    def safe_remove_initializer(graph, name):
        """Safely remove initializer if not used by any nodes."""
        # Check if still used by nodes
        for node in graph.node:
            if name in node.input:
                return  # Still in use, cannot remove

        # Check graph.input / graph.output
        for inp in graph.input:
            if inp.name == name:
                return
        for out in graph.output:
            if out.name == name:
                return

        # Safe to remove
        for init in graph.initializer:
            if init.name == name:
                graph.initializer.remove(init)
                break

    # 1. Find QuantizeLinear node
    qnode = None
    for node in graph.node:
        if node.name == qlinear_name:
            qnode = node
            break

    if qnode is None:
        raise ValueError(f"QuantizeLinear node {qlinear_name} not found")

    # 2. Get input/output names
    if not qnode.input or not qnode.output:
        raise ValueError(f"QuantizeLinear node {qlinear_name} must have input and output")

    q_in = qnode.input[0]    # Original input (float32)
    q_out = qnode.output[0]  # Quantized output (uint8)

    # 3. Replace q_out with q_in in all subsequent nodes
    replacement_count = 0
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == q_out:
                node.input[i] = q_in
                replacement_count += 1

    if replacement_count == 0:
        raise ValueError(f"No nodes found using quantized output {q_out}")

    # 4. Remove the quantization node
    graph.node.remove(qnode)

    # 5. Remove scale and zero_point initializers if safe
    if len(qnode.input) > 1:
        safe_remove_initializer(graph, qnode.input[1])  # scale
    if len(qnode.input) > 2:
        safe_remove_initializer(graph, qnode.input[2])  # zero_point

    # 6. Change graph.input data type to uint8
    input_found = False
    for i, inp in enumerate(graph.input):
        if inp.name == input_name:
            old_shape = inp.type.tensor_type.shape

            # Extract dimension values, using symbolic 'batch' for first dimension
            new_shape_dims = []
            for j, dim in enumerate(old_shape.dim):
                if j == 0 and dim.dim_value == 0:  # Assume first dim is batch if 0
                    new_shape_dims.append('batch')
                elif dim.dim_value > 0:
                    new_shape_dims.append(dim.dim_value)
                else:
                    new_shape_dims.append(dim.dim_param or f'dim_{j}')

            new_input = onnx.helper.make_tensor_value_info(
                name=input_name,
                elem_type=onnx.TensorProto.UINT8,
                shape=new_shape_dims
            )
            graph.input.remove(inp)
            graph.input.insert(i, new_input)
            input_found = True
            break

    if not input_found:
        raise ValueError(f"Input {input_name} not found in graph")

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

def _extract_shape_from_tensor_type(tensor_type) -> list[int]:
    """Extract shape dimensions from tensor type, using -1 for unknown dimensions."""
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append(-1)  # Unknown dimension
    return shape

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
            return _extract_shape_from_tensor_type(vi.type.tensor_type)

    # Check input
    for inp in model.graph.input:
        if inp.name == tensor_name:
            return _extract_shape_from_tensor_type(inp.type.tensor_type)

    # Check output
    for out in model.graph.output:
        if out.name == tensor_name:
            return _extract_shape_from_tensor_type(out.type.tensor_type)

    return None