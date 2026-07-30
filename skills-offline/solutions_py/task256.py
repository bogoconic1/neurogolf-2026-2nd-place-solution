from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task256'
TASK_NUM = 256
KAGGLE = {'score': 20.435652, 'date': '2026-07-11'}
MEMORY_BYTES = 0
PARAMS = 96
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task256_m0_p102_foldH'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task256_m0_p102'
OPSETS = [('', 17)]


# T: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_T_0 = [-0.054893508553504944, -0.12669946253299713, 0.3960760235786438, 0.8082855343818665, -0.24500322341918945,
 5.253284931182861, -0.951344907283783, 4.580333709716797, -0.44723033905029297, -4.925486087799072,
 0.49952560663223267, -4.070756912231445, 0.23044312000274658, -4.173483371734619, 0.5119039416313171,
 -5.381165027618408]

# code: FLOAT[2, 10], 20 value(s)
INIT_CODE_1 = [0.7710484862327576, 1.8093370199203491, -0.1019984632730484, 2.3690755367279053, 1.0249643325805664,
 1.0266799926757812, 1.0261192321777344, 1.020029067993164, 1.0288450717926025, 1.0258632898330688, -0.7710535526275635,
 -0.2338392585515976, 1.071225881576538, 0.5742306113243103, 1.4804733991622925, 1.4828951358795166, 1.481850028038025,
 1.4755278825759888, 1.4861104488372803, 1.4816783666610718]

# coord: FLOAT[2, 30], 60 value(s)
INIT_COORD_2 = [1.0011265277862549, 1.0015381574630737, 0.9997373819351196, 1.0005916357040405, 0.999506950378418, 0.9990650415420532,
 0.999501645565033, 0.9990717172622681, 1.0003511905670166, 0.9991843700408936, 0.999565064907074, 0.9997923374176025,
 1.000007152557373, -0.4981101453304291, -0.13549749553203583, -0.9190816879272461, 0.35635846853256226,
 -0.8066427707672119, -1.0249234437942505, -1.2155752182006836, -1.1351561546325684, -0.9610480666160583,
 -0.3368450105190277, -0.7559136748313904, -0.9564995765686035, 0.04832533746957779, -0.8374195694923401,
 -0.9242187142372131, -1.1698648929595947, -0.7148013710975647, -0.0023926307912915945, 0.10010766237974167,
 0.20150651037693024, 0.29952871799468994, 0.40096378326416016, 0.5015479922294617, 0.599981963634491,
 0.7019933462142944, 0.7980664968490601, 0.902113676071167, 1.001031517982483, 1.1014807224273682, 1.2032548189163208,
 -0.765290379524231, -0.5514254570007324, -0.43879935145378113, -0.14786581695079803, -1.230204463005066,
 -0.5988892912864685, -0.7712613940238953, -0.8735709190368652, -0.7783988118171692, 0.10433019697666168,
 0.8612608909606934, -0.4747318923473358, -0.48775678873062134, -1.1711899042129517, -0.4940734803676605,
 -0.7205645442008972, -0.2533276677131653]


NP_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
}


def _num(value):
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan":
        return float("nan")
    return value


def _tensor(name: str, elem_type: int, shape: tuple[int, ...], values: list) -> onnx.TensorProto:
    if elem_type == TensorProto.STRING:
        vals = [v.encode("utf-8") if isinstance(v, str) else v for v in values]
        return helper.make_tensor(name=name, data_type=TensorProto.STRING, dims=list(shape), vals=vals)
    flat = [_num(value) for value in values]
    array = np.asarray(flat, dtype=NP_DTYPE[elem_type]).reshape(shape)
    return numpy_helper.from_array(array, name=name)


def _vi(name: str, elem_type: int, shape: list[int | str | None]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


class _EmptyAttr:
    # Sentinel for an EMPTY list-valued attribute (INTS/FLOATS/STRINGS, e.g. a scalar
    # RandomUniform's shape=[]). helper.make_node() cannot infer the attribute type from a
    # bare [], so _node() re-adds these with an explicit attr_type after building the node.
    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


def _node(op_type, inputs, outputs, name="", domain="", **attrs):
    empties = [(k, v.kind) for k, v in attrs.items() if isinstance(v, _EmptyAttr)]
    for k, _ in empties:
        attrs.pop(k)
    node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
    for k, kind in empties:
        node.attribute.append(helper.make_attribute(k, [], attr_type=getattr(AttributeProto, kind)))
    return node


def make_onnx() -> onnx.ModelProto:
    nodes = [
        # 0: Einsum inputs=94 outputs=1
        _node(
            'Einsum',
            ['coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord',
             'coord', 'coord', 'coord', 'coord', 'T', 'T', 'T', 'T', 'coord', 'coord', 'coord', 'coord',
             'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord',
             'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord',
             'coord', 'coord', 'code', 'coord', 'input', 'coord', 'T', 'coord', 'coord', 'coord', 'coord',
             'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'code', 'coord', 'input',
             'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord', 'coord',
             'coord', 'coord', 'coord', 'coord', 'coord', 'T', 'coord', 'coord', 'code', 'code', 'code',
             'coord', 'coord', 'coord', 'coord', 'input'],
            ['output'],
            name='output',
            domain='',
            equation='Sm,Sm,Sm,Sm,Sm,Sm,Sm,Sm,Sm,Sm,Sm,Sm,Lm,Dm,Tm,SDzX,SLzX,STzX,CTLL,Sb,Sb,Sb,Sb,Sb,Sb,Kb,Kb,Kb,Kb,Kb,Kb,Kb,Vb,Vb,Vb,Eb,Eb,KO,JO,JO,HO,HO,JN,EN,EN,IN,IN,Yy,EB,nyAB,FA,VICU,IM,FM,FM,GM,GM,HQ,GP,fr,fr,er,er,gr,Za,gd,nacd,ic,is,is,js,js,ls,gt,gt,lt,lt,pt,pu,pu,ku,ku,qu,VWql,kx,jv,Vo,Uo,Wo,Gh,jh,kw,Hw,nRhw->nohw',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('T', TensorProto.FLOAT, (2, 2, 2, 2), INIT_T_0),
            _tensor('code', TensorProto.FLOAT, (2, 10), INIT_CODE_1),
            _tensor('coord', TensorProto.FLOAT, (2, 30), INIT_COORD_2),
        ],
        value_info=[

        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain, version) for domain, version in OPSETS],
        producer_name=PRODUCER_NAME,
        producer_version=PRODUCER_VERSION,
        domain=DOMAIN,
        model_version=MODEL_VERSION,
        doc_string="",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def save_model(path: str | Path = OUT) -> None:
    onnx.save_model(make_onnx(), path)


if __name__ == "__main__":
    save_model(sys.argv[1] if len(sys.argv) > 1 else OUT)
