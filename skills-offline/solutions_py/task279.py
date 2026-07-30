from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task279'
TASK_NUM = 279
KAGGLE = {'score': 19.976119, 'date': '2026-07-15'}
MEMORY_BYTES = 36
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task279_flex_C8_L5'
OPSETS = [('', 20)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [1.4108096361160278, 1.2855271100997925, 1.344118595123291, 1.3544286489486694, 1.5088955163955688, 2.1785173416137695,
 1.410071611404419, 3.235286235809326, 1.9589946269989014, 2.2399566173553467, 2.6105127334594727, 2.534848928451538,
 2.0682995319366455, 2.842311143875122, 2.759934425354004, 3.056389093399048, 1.8222747104736934e-36,
 -1.8691965505595102e-36, -1.866401128278998e-36, -1.833436310580958e-36, 1.7761110513297904e-36,
 1.8370600459889266e-36, -1.7871822508705245e-36, -1.774407251763375e-36, -1.7576526547006444e-36,
 -1.8731180338651785e-36, -1.834967380493467e-36, -1.8337771063674818e-36, 1.770173671263732e-36,
 -1.7762705078846428e-36, -1.6117793321609497, -1.451951026916504, -1.357587218284607, -1.361472487449646,
 -1.4747941493988037, -1.0294599533081055, -0.3976086676120758, -0.2637298107147217, 0.5418962240219116,
 0.6126876473426819, 1.202108383178711, 1.2015864849090576, 1.043596625328064, 1.5027278661727905, 1.502797245979309,
 1.5815483331680298, 1.8041086807561442e-36, 1.847252709865243e-36, -1.7733513229237614e-36, 1.8463609011017713e-36,
 -1.835281809448086e-36, -1.8097696575027113e-36, 1.843396336491421e-36, -1.8554178181779463e-36, 1.811436686997423e-36,
 -1.8106372518287195e-36, 1.805451774887455e-36, -1.797428365875464e-36, -1.78980440539852e-36, 1.84502874830887e-36]

# U3: FLOAT[4, 3], 12 value(s)
INIT_U3_1 = [-0.00864376500248909, 0.3016620874404907, 0.8251881003379822, 0.013990785926580429, 2.0363497734069824,
 -0.26748764514923096, -0.021950967609882355, -0.3189685344696045, -0.2569514811038971, -0.01334733422845602,
 -0.741858959197998, -0.2233729362487793]

# V3: FLOAT[4, 3], 12 value(s)
INIT_V3_2 = [0.01472671888768673, 0.0026416205801069736, -0.364766001701355, -0.005336633883416653, -0.617597222328186,
 -0.7582607269287109, 0.003748547751456499, 1.6234592199325562, -0.3043660521507263, -0.02219478413462639,
 -1.637393832206726, 0.6065248847007751]

# P: FLOAT[3, 2], 6 value(s)
INIT_P_3 = [11.693769454956055, 0.0, -0.0, -1.1685456037521362, 1.4347695112228394, 1.4347695112228394]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_4 = [0.0, 0.0, 0.8701027035713196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8701027035713196,
 0.0, 0.0, 1.7552659511566162]

# prob: FLOAT[1, 6], 6 value(s)
INIT_PROB_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Multinomial inputs=1 outputs=1
        _node(
            'Multinomial',
            ['prob'],
            ['token'],
            name='',
            domain='',
            dtype=6,
            sample_size=5,
            seed=145.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['token'],
            ['G'],
            name='',
            domain='',
            max_gram_length=5,
            max_skip_count=4,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 2, 46, 295, 499],
            ngram_indexes=[3, 0, 3, 2, 2, 1, 2, 1, 2, 1, 3, 3, 1, 3, 1, 3, 1, 3, 2, 1, 1, 0, 0, 0, 1, 2, 0, 3, 3, 3, 3, 2, 0,
 1, 0, 1, 2, 0, 1, 2, 1, 2, 2, 2, 3, 0, 2, 2, 2, 3, 2, 0, 0, 2, 1, 0, 1, 0, 0, 1, 2, 2, 0, 0, 0, 1,
 2, 2, 3, 1, 3, 2, 0, 2, 2, 2, 3, 0, 0, 0, 2, 1, 1, 3, 1, 0, 0, 3, 2, 2, 1, 2, 3, 0, 2, 1, 0, 2, 0,
 2, 1, 0, 0, 0, 2, 3, 1, 2, 0, 2, 3, 2, 0, 1, 1, 1, 2, 3, 3, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 3,
 1, 2, 2, 0, 3, 0, 2, 2, 3, 1, 2, 0, 3, 1, 2, 3, 3, 2, 0, 1, 2, 0, 3, 0, 2, 3, 3, 0, 2, 2, 1, 1, 1,
 1, 1, 0, 1, 2, 2, 0, 2, 2, 0, 3, 1, 1, 0, 1, 0, 3, 0, 0],
            pool_int64s=[1, 2, 0, 0, 0, 1, 0, 2, 0, 5, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 2, 4, 2, 5, 3, 1, 3, 2, 3, 3, 3,
 5, 4, 4, 4, 5, 5, 0, 5, 1, 5, 2, 5, 3, 0, 0, 4, 0, 0, 5, 0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 2, 3, 0, 2,
 5, 0, 3, 1, 0, 3, 4, 0, 4, 0, 0, 4, 2, 0, 4, 3, 0, 4, 4, 0, 5, 1, 0, 5, 4, 0, 5, 5, 1, 0, 4, 1, 1,
 5, 1, 2, 2, 1, 2, 3, 1, 2, 4, 1, 3, 0, 1, 3, 1, 1, 3, 5, 1, 4, 1, 1, 4, 4, 1, 5, 0, 1, 5, 2, 1, 5,
 5, 2, 0, 1, 2, 0, 3, 2, 1, 2, 2, 1, 3, 2, 1, 4, 2, 1, 5, 2, 3, 0, 2, 3, 1, 2, 3, 5, 2, 4, 0, 2, 4,
 1, 2, 4, 3, 2, 4, 4, 2, 5, 0, 2, 5, 1, 2, 5, 3, 2, 5, 5, 3, 0, 0, 3, 0, 1, 3, 0, 4, 3, 0, 5, 3, 1,
 1, 3, 1, 3, 3, 1, 4, 3, 1, 5, 3, 2, 1, 3, 2, 4, 3, 3, 5, 3, 4, 1, 3, 4, 2, 3, 4, 5, 3, 5, 2, 3, 5,
 3, 4, 0, 4, 4, 0, 5, 4, 1, 1, 4, 1, 2, 4, 2, 3, 4, 2, 4, 4, 2, 5, 4, 3, 2, 4, 4, 2, 4, 5, 0, 4, 5,
 4, 5, 0, 0, 5, 1, 2, 5, 1, 5, 5, 2, 4, 5, 3, 0, 5, 3, 4, 5, 4, 1, 5, 4, 3, 5, 5, 1, 5, 5, 5, 0, 0,
 5, 5, 0, 1, 3, 0, 0, 1, 3, 5, 0, 1, 4, 4, 0, 2, 3, 4, 0, 2, 4, 3, 0, 2, 5, 5, 0, 3, 4, 2, 0, 4, 0,
 4, 0, 4, 4, 2, 0, 5, 1, 5, 1, 0, 4, 0, 1, 2, 4, 3, 1, 3, 0, 4, 1, 3, 1, 1, 1, 4, 1, 1, 2, 0, 1, 3,
 2, 1, 3, 1, 2, 1, 5, 5, 2, 3, 4, 1, 2, 3, 5, 3, 2, 4, 0, 1, 2, 4, 3, 2, 2, 5, 1, 2, 2, 5, 3, 0, 2,
 5, 5, 5, 3, 0, 0, 5, 3, 0, 1, 4, 3, 0, 4, 3, 3, 2, 1, 4, 3, 2, 1, 5, 3, 3, 5, 4, 3, 4, 1, 2, 3, 4,
 2, 3, 3, 5, 2, 4, 3, 5, 3, 0, 3, 5, 3, 4, 3, 5, 4, 3, 4, 1, 1, 5, 4, 1, 2, 3, 4, 2, 5, 3, 4, 3, 2,
 1, 4, 4, 2, 4, 4, 5, 4, 1, 5, 0, 0, 4, 5, 1, 2, 2, 5, 1, 5, 2, 5, 3, 4, 5, 5, 4, 1, 2, 5, 5, 4, 1,
 5, 5, 5, 4, 0, 1, 2, 4, 3, 0, 1, 3, 0, 4, 0, 2, 3, 4, 1, 0, 2, 4, 3, 2, 0, 2, 5, 5, 5, 0, 3, 4, 2,
 3, 0, 5, 1, 5, 2, 1, 0, 4, 0, 4, 1, 4, 1, 1, 5, 2, 0, 1, 3, 5, 2, 1, 3, 1, 1, 2, 2, 4, 0, 1, 2, 3,
 5, 3, 0, 2, 5, 1, 2, 2, 3, 0, 0, 5, 5, 3, 0, 1, 4, 4, 3, 2, 1, 5, 5, 3, 3, 5, 4, 3, 3, 4, 1, 2, 3,
 3, 5, 2, 4, 4, 3, 5, 3, 4, 5, 4, 0, 5, 4, 0, 4, 2, 5, 3, 0, 4, 3, 2, 1, 4, 4, 5, 4, 1, 2, 5, 5, 5,
 4, 1],
        ),
        # 2: Einsum inputs=16 outputs=1
        _node(
            'Einsum',
            ['input', 'G', 'U3', 'P', 'B', 'P', 'B', 'V3', 'P', 'B', 'P', 'B', 'U3', 'P', 'E', 'E'],
            ['output'],
            name='',
            domain='',
            equation='nihw,ql,lt,ta,ah,td,dh,ls,se,ew,sf,fw,lk,kr,ir,or->nohw',
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
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_0),
            _tensor('U3', TensorProto.FLOAT, (4, 3), INIT_U3_1),
            _tensor('V3', TensorProto.FLOAT, (4, 3), INIT_V3_2),
            _tensor('P', TensorProto.FLOAT, (3, 2), INIT_P_3),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_4),
            _tensor('prob', TensorProto.FLOAT, (1, 6), INIT_PROB_5),
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
