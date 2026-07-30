from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task106'
TASK_NUM = 106
KAGGLE = {'score': 19.864202, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 170
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task106_p170_D_absorbed_into_F'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task106_p170_D_absorbed_into_F'
OPSETS = [('', 18)]


# B: FLOAT[30, 4], 120 value(s)
INIT_B_0 = [-0.6190122365951538, -0.03293991833925247, -0.9253268241882324, -0.4297840893268585, -0.611513614654541,
 0.7936578392982483, 0.13124728202819824, -0.4955260157585144, -0.6196925640106201, -0.7089199423789978,
 0.5774383544921875, 0.2993050217628479, -0.5723261833190918, 0.5429267883300781, -0.14322611689567566,
 0.8567661643028259, -0.5663244128227234, -0.6847149729728699, -0.29459041357040405, -0.04779372736811638,
 -0.5602009296417236, 0.08546216040849686, 0.6246782541275024, -0.2164919227361679, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# U: FLOAT[5, 4], 20 value(s)
INIT_U_1 = [-7.937877178192139, -0.001896850997582078, 0.006821763701736927, -0.1363871693611145, -7.701587677001953,
 0.013323256745934486, 0.004166860599070787, -0.07251264899969101, -7.597186088562012, -0.0022016349248588085,
 0.017940381541848183, 0.052879005670547485, -7.4652886390686035, 0.01637333445250988, -0.007510317023843527,
 -0.009989862330257893, -7.7554731369018555, -0.014122835360467434, -0.010289324447512627, -0.042467277497053146]

# V: FLOAT[2, 4], 8 value(s)
INIT_V_2 = [2.169941581087187e-05, 62.09088897705078, -65.51974487304688, 7.395383834838867, 0.1955927461385727,
 0.44228219985961914, -0.8132419586181641, -0.04550720378756523]

# group: FLOAT[2, 5], 10 value(s)
INIT_GROUP_3 = [0.2989850640296936, 0.2845793068408966, 0.03131110593676567, 0.005673978012055159, -0.0635078102350235,
 2.953346847789362e-05, -0.002198904287070036, 0.8195092678070068, 0.8836799263954163, 0.8573722839355469]

# F: FLOAT[2, 2, 3], 12 value(s)
INIT_F_4 = [92.97628784179688, -52.14451599121094, 1.2910438776016235, 50.65394973754883, 48.163543701171875, -0.4299118220806122,
 -50.839054107666016, -47.642642974853516, 0.23293252289295197, 3.4403979778289795, -3.9462709426879883,
 -25.118513107299805]


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
        # 0: Einsum inputs=58 outputs=1
        _node(
            'Einsum',
            ['B', 'input', 'B', 'V', 'group', 'U', 'V', 'U', 'B', 'B', 'V', 'B', 'U', 'V', 'B', 'U', 'V',
             'group', 'U', 'V', 'B', 'input', 'V', 'U', 'B', 'B', 'B', 'U', 'V', 'U', 'V', 'F', 'F', 'F', 'U',
             'V', 'V', 'U', 'B', 'B', 'B', 'V', 'U', 'V', 'B', 'U', 'V', 'U', 'U', 'V', 'V', 'U', 'U', 'V', 'B',
             'B', 'B', 'B'],
            ['output'],
            name='task106_p173_rank3_coeff',
            domain='',
            equation='kp,nekl,lp,gp,ga,aA,qA,aD,iA,iD,uD,iB,aB,rB,iC,aC,sC,gb,bG,rG,jG,ndij,sH,bH,jH,jI,jF,bI,uI,bF,qF,qsz,ruz,xyz,aJ,qJ,sV,bV,hJ,hV,hW,uW,bW,rK,hK,aK,qL,bL,bM,rM,sN,aN,aR,uR,wL,wR,wN,wM->ndhw',
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
            _tensor('B', TensorProto.FLOAT, (30, 4), INIT_B_0),
            _tensor('U', TensorProto.FLOAT, (5, 4), INIT_U_1),
            _tensor('V', TensorProto.FLOAT, (2, 4), INIT_V_2),
            _tensor('group', TensorProto.FLOAT, (2, 5), INIT_GROUP_3),
            _tensor('F', TensorProto.FLOAT, (2, 2, 3), INIT_F_4),
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
