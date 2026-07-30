from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task323'
TASK_NUM = 323
KAGGLE = {'score': 20.522663, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 18)]


# Q: FLOAT[2, 30], 60 value(s)
INIT_Q_0 = [-0.1365225911140442, 0.2739807963371277, 0.22623319923877716, -0.36637187004089355, 0.2582419514656067,
 -0.40280380845069885, -0.26881685853004456, 0.40333279967308044, -0.2583896219730377, -0.36802226305007935,
 0.2265191674232483, -0.276154100894928, -0.13809619843959808, -0.40043067932128906, -0.4847048223018646,
 0.9304414987564087, -1.017276406288147, -1.0360875129699707, 0.9035420417785645, -0.4637770354747772,
 -0.381635457277298, 0.8572992086410522, 0.6519967317581177, -0.8834975361824036, 0.9428994059562683, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.09094258397817612, 0.24578124284744263, -0.15065471827983856, -0.3286360204219818, -0.1719161868095398,
 -0.3612859845161438, 0.17889992892742157, 0.3617308437824249, 0.17190653085708618, -0.33003538846969604,
 -0.1506558209657669, -0.24762952327728271, 0.09181767702102661, 1.1297521591186523, 1.0066282749176025,
 -1.0076160430908203, 0.39566317200660706, -0.3855251371860504, 0.9980717897415161, -1.0045208930969238,
 -1.1268752813339233, -0.4646410048007965, 0.07285353541374207, -0.1532759964466095, 0.5664483904838562, 0.0, 0.0, 0.0,
 0.0, 0.0]

# N: FLOAT[2, 2, 2], 8 value(s)
INIT_N_1 = [-2.253123420814518e-05, -16.24058723449707, 16.242324829101562, -0.001169057795777917, 0.0014796426985412836,
 -0.1223401352763176, -0.12633906304836273, 0.18241310119628906]

# O: FLOAT[2, 10], 20 value(s)
INIT_O_2 = [-11508.2685546875, 0.0, 0.0, 0.0, 0.0, 11508.2685546875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 11508.2685546875, 0.0]


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
        # 0: Einsum inputs=62 outputs=1
        _node(
            'Einsum',
            ['Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'N', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'Q', 'Q', 'N', 'Q', 'Q', 'Q', 'Q', 'input', 'O', 'O', 'O', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'N', 'Q', 'Q', 'Q', 'Q', 'Q', 'N', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'input', 'O', 'O', 'O'],
            ['output'],
            name='',
            domain='',
            equation='uH,uH,tH,tI,aI,aI,bI,bI,bI,bI,tcd,ah,ch,dr,br,mr,mJ,mJ,oJ,oJ,oJ,oJ,TJ,oh,xr,Txy,yh,TK,UK,UK,nqrs,Mq,Mq,MP,fs,fL,fL,fL,fL,uL,eL,eL,ugi,is,ew,gw,zs,Bs,UBC,Cw,UR,zR,zR,AR,AR,AR,AR,Aw,nphw,Gp,Gp,Gv->nvhw',
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
            _tensor('Q', TensorProto.FLOAT, (2, 30), INIT_Q_0),
            _tensor('N', TensorProto.FLOAT, (2, 2, 2), INIT_N_1),
            _tensor('O', TensorProto.FLOAT, (2, 10), INIT_O_2),
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
