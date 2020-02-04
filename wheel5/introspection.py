from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, List, Dict, Set, Optional, Callable, Tuple, Iterable
from collections import deque

import torch
from graphviz import Digraph
from torch import nn


#
# Inspired by torchviz and pytorch-summary
#

def size_to_str(size: Iterable[int]) -> str:
    return 'x'.join(['%d' % v for v in size])


class ControlFlow(Enum):
    CONTINUE = 1
    STOP = 2
    BREAK = 3


class GradId(NamedTuple):
    id: str

    def __repr__(self):
        return f'grad-{self.id}'


class TensorId(NamedTuple):
    id: str

    def __repr__(self):
        return f'tensor-{self.id}'


class NodeData(ABC):

    @abstractmethod
    def __repr__(self):
        pass


class ModuleData(NodeData):
    def __init__(self, class_name: str, params: Dict[TensorId, Tuple[str, List[int]]], fake: bool = False):
        super(ModuleData, self).__init__()

        self.class_name = class_name
        self.params = params
        self.fake = fake

    def __repr__(self):
        fake_str = 'fake_' if self.fake else ''
        params_str = ', '.join([f'{tid}: {name} {size_to_str(size)}' for tid, (name, size) in self.params.items()])

        return f'{fake_str}module.{self.class_name}({params_str})'


class VarData(NodeData):
    def __init__(self, class_name: str, variable_id: Optional[TensorId], variable_size: Optional[List[int]], comment: Optional[str] = None):
        super(VarData, self).__init__()

        self.class_name = class_name
        self.variable_id = variable_id
        self.variable_size = variable_size
        self.comment = comment

    def update_if_needed(self, variable_id: TensorId, variable_size: List[int]):
        if self.variable_id is None:
            self.variable_id = variable_id

        if self.variable_size is None:
            self.variable_size = variable_size

    def __repr__(self):
        dump = f'var.{self.class_name}'

        if self.variable_id is not None:
            dump += f' - {self.variable_id}'

        if self.variable_size is not None:
            dump += f' {size_to_str(self.variable_size)}'

        if self.comment is not None:
            dump += f' # {self.comment}'

        return dump

    def is_variable(self) -> bool:
        return (self.variable_id is not None) and (self.variable_size is not None)

    def is_param(self) -> bool:
        return (self.is_variable()) and (self.comment is not None)


class Beam(NamedTuple):
    seed: GradId
    inners: Set[GradId]
    terminators: Set[GradId]
    edges: List[Tuple[GradId, GradId]]

    def __repr__(self):
        dump = ''

        dump += f'seed: {self.seed}\n'
        dump += f'inners: {", ".join([str(gid) for gid in self.inners])}\n'
        dump += f'terminators: {", ".join([str(gid) for gid in self.terminators])}\n'
        dump += f'edges: {", ".join([str(a) + " <- " + str(b) for a, b in self.edges])}\n'

        return dump


def var2list(var):
    if isinstance(var, tuple):
        return list(var)
    else:
        return [var]


def grad_id(o) -> GradId:
    return GradId(hex(id(o)))


def tensor_id(o) -> TensorId:
    return TensorId(hex(id(o)))


class NetworkGraph(object):
    def __init__(self):
        self.nodes: Dict[GradId, NodeData] = {}  # id -> data
        self.edges: Dict[GradId, Set[GradId]] = {}  # out -> set(in)

    def add_node(self, gid: GradId, data: NodeData):
        assert gid not in self.nodes
        self.nodes[gid] = data

    def replace_node(self, gid: GradId, data: NodeData):
        assert gid in self.nodes
        self.nodes[gid] = data

    def delete_node(self, gid: GradId):
        assert gid in self.nodes
        self.nodes.pop(gid, None)

    def get_node(self, gid: GradId) -> NodeData:
        return self.nodes[gid]

    def add_edge(self, gid_in: GradId, gid_out: GradId):
        assert gid_in in self.nodes
        assert gid_out in self.nodes

        gids_in = self.edges.setdefault(gid_out, set())
        gids_in.add(gid_in)

    def delete_edge(self, gid_in: GradId, gid_out: GradId):
        assert gid_in in self.nodes
        assert gid_out in self.nodes

        gids_in = self.edges.get(gid_out) or set()
        gids_in.remove(gid_in)

    def get_edges(self, gid_out: GradId) -> Set[GradId]:
        return self.edges.get(gid_out) or set()

    def contains(self, gid: GradId) -> bool:
        return gid in self.nodes

    def beam_search(self, gid: GradId, control: Callable[[GradId, NodeData], ControlFlow]) -> Optional[Beam]:
        if gid not in self.nodes:
            return None

        stack = deque([gid])

        inners = set()
        terminators = set()
        beam_edges = []

        while len(stack) > 0:
            gid_out = stack.pop()
            gids_in = self.edges.get(gid_out) or set()

            if len(gids_in) == 0:
                return None

            for gid_in in gids_in:
                beam_edges.append((gid_out, gid_in))

                flow = control(gid_in, self.nodes[gid_in])
                if flow == ControlFlow.STOP:
                    terminators.add(gid_in)
                elif flow == ControlFlow.CONTINUE:
                    if gid_in not in inners:
                        inners.add(gid_in)
                        stack.append(gid_in)
                elif flow == ControlFlow.BREAK:
                    return None
                else:
                    raise AssertionError(f'Unexpected control flow: {flow}')

        return Beam(gid, inners, terminators, beam_edges)

    def drop_beam(self, beam: Beam, exclude_gids: Set[GradId]):
        for gid in beam.inners.union(beam.terminators):
            if gid not in exclude_gids:
                self.nodes.pop(gid, None)

        for gid_out, gid_in in beam.edges:
            if gid_out in self.edges:
                self.edges[gid_out].remove(gid_in)
                if len(self.edges[gid_out]) == 0:
                    self.edges.pop(gid_out)

    def __repr__(self) -> str:
        dump = ''

        dump += 'Nodes:\n'
        for gid in sorted(self.nodes.keys(), key=lambda x: x.id):
            data = self.nodes[gid]
            data_str = "" if data is None else f': {data}'
            dump += f'    {gid}{data_str}\n'

        dump += '\nEdges:\n'
        for gid_out in sorted(self.edges.keys(), key=lambda x: x.id):
            gids_in = sorted(self.edges[gid_out], key=lambda x: x.id)
            gids_in_str = ', '.join([str(gid_in) for gid_in in gids_in])
            dump += f'    {gid_out} <- [{gids_in_str}]\n'

        return dump


class LogRecord(object):
    def __init__(self, module: nn.Module, input: List[torch.Tensor], output: List[torch.Tensor]):
        self.module = module
        self.input = input
        self.output = output

        d = dict(module.named_parameters(recurse=False))
        self.module_params = {tensor_id(v): (k, tuple(v.size())) for k, v in d.items()}

        self.input_gids = set([grad_id(t.grad_fn) for t in input])
        self.output_gids = set([grad_id(t.grad_fn) for t in output])

        self.module_class_name = str(self.module.__class__).split(".")[-1].split("'")[0]

    def __repr__(self):
        params_str = ', '.join([f'{tid}: {name} {size_to_str(size)}' for tid, (name, size) in self.module_params.items()])
        input_str = ', '.join([f'{tensor_id(t)} # {grad_id(t.grad_fn)}' for t in self.input])
        output_str = ', '.join([f'{tensor_id(t)} # {grad_id(t.grad_fn)}' for t in self.output])

        return f'{self.module_class_name}({params_str}) - {input_str} => {output_str}'


def introspect(model: nn.Module, input_size) -> NetworkGraph:
    anti_gc = set()  # Needed to prevent the variables from being reused by the torch backend

    hook_handles = []

    params = dict(model.named_parameters())
    param_map = {tensor_id(v): k for k, v in params.items()}

    network_graph = NetworkGraph()
    log = []

    def traverse_grad(var):
        anti_gc.add(var)
        var_id = grad_id(var)

        if not network_graph.contains(var_id):
            class_name = str(type(var).__name__)

            if hasattr(var, 'variable'):
                u = var.variable
                variable_id = tensor_id(u)
                variable_size = u.size()
                comment = param_map.get(variable_id)
            else:
                variable_id = None
                variable_size = None
                comment = None

            network_graph.add_node(var_id, VarData(class_name=class_name,
                                                   variable_id=variable_id,
                                                   variable_size=variable_size,
                                                   comment=comment))

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        u_id = traverse_grad(u[0])
                        network_graph.add_edge(u_id, var_id)

            if hasattr(var, 'saved_tensors'):
                for u in var.saved_tensors:
                    u_id = traverse_grad(u)
                    network_graph.add_edge(u_id, var_id)

        return var_id

    def forward_hook(module: nn.Module, input, output):
        input = var2list(input)
        output = var2list(output)

        for entry in (input + output):
            anti_gc.add(entry.grad_fn)

        log.append(LogRecord(module=module, input=input, output=output))

    def register_hook(module: nn.Module):
        handle = module.register_forward_hook(forward_hook)
        hook_handles.append(handle)

    def run_model():
        dtype = torch.cuda.FloatTensor
        input_size_tuple = [input_size] if not isinstance(input_size, tuple) else input_size
        x = [torch.rand(*in_size).type(dtype) for in_size in input_size_tuple]

        model.apply(register_hook)
        y = model(*x)

        for t in var2list(y):
            traverse_grad(t.grad_fn)

        for h in hook_handles:
            h.remove()

    def control_flow(record: LogRecord):
        def check(gid: GradId, data: NodeData) -> ControlFlow:

            if isinstance(data, VarData):
                if gid in record.input_gids:
                    return ControlFlow.STOP
                if data.variable_id is not None:
                    return ControlFlow.STOP if data.variable_id in record.module_params else ControlFlow.CONTINUE
                return ControlFlow.CONTINUE
            elif isinstance(data, ModuleData):
                return ControlFlow.BREAK
            else:
                raise AssertionError(f'Unexpected node data type: {type(data)}')

        return check

    def attach_or_update_tensor(gid: GradId, tensor: torch.Tensor, class_name: str, force_class: bool):
        if network_graph.contains(gid):
            data = network_graph.get_node(gid)
            if isinstance(data, VarData):
                data.update_if_needed(variable_id=tensor_id(tensor), variable_size=list(tensor.size()))
                if force_class:
                    data.class_name = class_name
            else:
                raise AssertionError(f'Unexpected node data type: {type(data)}')
        else:
            network_graph.add_node(gid, VarData(class_name=class_name,
                                                variable_id=tensor_id(tensor),
                                                variable_size=list(tensor.size())))

    def compact_graph():
        # Replace parts of the graph with known modules
        for index, record in enumerate(log):
            for tensor_out in record.output:
                gid_out = grad_id(tensor_out.grad_fn)

                beam = network_graph.beam_search(gid_out, control_flow(record))
                if beam is not None:
                    network_graph.drop_beam(beam, record.input_gids)

                    # Replacing removed nodes with the module data
                    module_id = GradId(id=f'module#{index}')
                    if not network_graph.contains(module_id):
                        d = dict(record.module.named_parameters(recurse=False))
                        module_params = {tensor_id(v): (k, tuple(v.size())) for k, v in d.items()}
                        module_class_name = str(record.module.__class__).split(".")[-1].split("'")[0]

                        network_graph.add_node(module_id, ModuleData(class_name=module_class_name, params=module_params))

                    attach_or_update_tensor(gid_out, tensor_out, 'Tensor', force_class=True)
                    network_graph.add_edge(module_id, gid_out)

                    for tensor_in in record.input:
                        gid_in = grad_id(tensor_in.grad_fn)
                        attach_or_update_tensor(gid_in, tensor_in, 'Tensor', force_class=False)
                        network_graph.add_edge(gid_in, module_id)

        # Combine parameters and functional transformations into fake modules
        edges_to_delete = []  # [(in, out)]
        affected_param_nodes = set()

        for gid_out, data_out in network_graph.nodes.items():
            if isinstance(data_out, VarData):
                if not data_out.is_variable():

                    module_params = {}
                    for gid_in in network_graph.get_edges(gid_out):
                        data_in = network_graph.get_node(gid_in)

                        if isinstance(data_in, VarData):
                            if data_in.is_param() and len(network_graph.get_edges(gid_in)) == 0:
                                module_params[data_in.variable_id] = (data_in.comment, data_in.variable_size)

                                edges_to_delete.append((gid_in, gid_out))
                                affected_param_nodes.add(gid_in)

                    network_graph.replace_node(gid_out, ModuleData(data_out.class_name, module_params, fake=True))

        for gid_in, gid_out in edges_to_delete:
            network_graph.delete_edge(gid_in, gid_out)

        reverse_edges = {}  # in -> set(out)
        for gid_out, gids_in in network_graph.edges.items():
            for gid_in in gids_in:
                gids_out = reverse_edges.setdefault(gid_in, set())
                gids_out.add(gid_out)

        nodes_to_delete = []
        for gid in affected_param_nodes:
            if len(network_graph.get_edges(gid)) == 0 and len(reverse_edges.get(gid) or set()) == 0:
                nodes_to_delete.append(gid)

        for gid in nodes_to_delete:
            network_graph.delete_node(gid)

    run_model()

    # print(network_graph)
    # for r in log:
    #     print(r)

    compact_graph()

    print(network_graph)

    return network_graph


def make_dot(graph: NetworkGraph) -> Digraph:
    def resize_graph(g: Digraph, size_per_element: float = 0.15, min_size: float = 12):
        num_rows = len(g.body)
        content_size = num_rows * size_per_element
        side_size = max(min_size, content_size)
        size_str = str(side_size) + "," + str(side_size)
        g.graph_attr.update(size=size_str)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for gid, data in graph.nodes.items():
        if isinstance(data, VarData):
            text = f'{data.class_name}'
            if data.comment is not None:
                text += f'\n{data.comment}'
            if data.variable_size is not None:
                text += f'\n{size_to_str(data.variable_size)}'

            color = 'darkseagreen2' if data.class_name != 'Tensor' else 'gray85'
            dot.node(str(gid), text, fillcolor=color)
        elif isinstance(data, ModuleData):
            text = f'{data.class_name}'
            for _, (name, size) in data.params.items():
                text += f'\n{name} - {size_to_str(size)}'

            color = 'darkseagreen2' if data.fake else 'lightblue'
            dot.node(str(gid), text, fillcolor=color)
        else:
            raise AssertionError(f'Unexpected node data type: {type(data)}')

    for gid_out, gid_ins in graph.edges.items():
        for gid_in in gid_ins:
            dot.edge(str(gid_in), str(gid_out))

    resize_graph(dot)

    return dot
