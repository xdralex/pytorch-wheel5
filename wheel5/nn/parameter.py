from typing import Dict, Tuple, List

from torch.nn import Parameter


class ParamGroup(object):
    def __init__(self, config: Dict[str, float]):
        self.config: Dict[str, float] = config
        self.params: List[Tuple[str, Parameter]] = []

    def parameters(self) -> List[Parameter]:
        return [param for _, param in self.params]

    def __repr__(self):
        dump = 'ParamGroup(\n'

        dump += '  config:\n'
        for k, v in self.config.items():
            dump += f'    {k} -> {v:.8f}\n'

        dump += '  params:\n'
        for param_name, param in self.params:
            param_shape = 'x'.join([str(dim) for dim in param.shape])
            dump += f'    {param_name} - {param_shape}\n'

        dump += ')\n'

        return dump
