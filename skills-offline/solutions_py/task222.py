from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task222'
TASK_NUM = 222
KAGGLE = {'score': 20.605551, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 81
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task222_homogeneous_scale_absorbed_p81'
OPSETS = [('', 18)]


# T: FLOAT[10, 2, 3], 60 value(s)
INIT_T_0 = [-0.26572033762931824, -0.4760555922985077, 0.18712064623832703, 0.30349063873291016, -0.16347099840641022,
 0.4799102544784546, 0.6102515459060669, -0.7407151460647583, -2.1361749172210693, 0.3921286463737488,
 -0.9292358756065369, -0.6906734108924866, -0.6804676651954651, -0.6550074815750122, -0.5639013648033142,
 -0.6210151314735413, -1.386702537536621, 1.9392565488815308, -0.3046528398990631, -0.7518188953399658,
 -1.2795283794403076, 0.3343290388584137, -0.6781724691390991, -2.798588991165161, 1.1358532905578613,
 -0.7935349345207214, 0.9990646243095398, 0.018022269010543823, 1.5424418449401855, 0.5510801076889038,
 -0.42326033115386963, -0.45860743522644043, 0.5088151097297668, -2.1471171379089355, -0.03283026069402695,
 0.18916869163513184, 1.782152771949768, -0.8490023612976074, -1.4422526359558105, 0.18216465413570404,
 -0.45867329835891724, -0.524336040019989, -0.07323701679706573, -0.6971637606620789, 1.491572380065918,
 0.7328632473945618, 1.5071355104446411, -0.07814600318670273, -0.30101802945137024, -0.8171973824501038,
 0.3315103352069855, -1.444988489151001, -0.38447946310043335, -2.970060348510742, 1.7901149988174438,
 -0.830644965171814, -0.15746277570724487, -1.2594527006149292, -0.6189565658569336, 0.30566179752349854]

# C: FLOAT[3, 2, 3], 18 value(s)
INIT_C_1 = [1.4412819147109985, 0.7705930471420288, 0.43502944707870483, -1.0794665813446045, 0.7488076686859131,
 2.1005349159240723, -2.7970871925354004, 0.8351597785949707, -1.1062555313110352, 0.7097720503807068,
 1.0919125080108643, -2.2041401863098145, 0.04430798441171646, 0.9833654165267944, 0.8915823698043823,
 0.9195486903190613, -0.8686583638191223, -0.002797401510179043]

# P: FLOAT[3], 3 value(s)
INIT_P_2 = [0.028614209964871407, -0.013860995881259441, 2.0683298110961914]


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
        # 0: Einsum inputs=15 outputs=1
        _node(
            'Einsum',
            ['T', 'input', 'T', 'input', 'T', 'input', 'T', 'input', 'C', 'C', 'P', 'C', 'C', 'C', 'T'],
            ['output'],
            name='',
            domain='',
            equation='aqy,nahj,bqy,nbij,dqy,ndiw,cqy,nchw,kqy,kxz,k,lqy,lxz,lxz,oxz->nohw',
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
            _tensor('T', TensorProto.FLOAT, (10, 2, 3), INIT_T_0),
            _tensor('C', TensorProto.FLOAT, (3, 2, 3), INIT_C_1),
            _tensor('P', TensorProto.FLOAT, (3,), INIT_P_2),
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
