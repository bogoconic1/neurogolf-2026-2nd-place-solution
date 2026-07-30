from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task089'
TASK_NUM = 89
KAGGLE = {'score': 18.817915, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 484
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task089_alias_sz_zb_diag'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task089_alias_sz_zb_diag'
OPSETS = [('', 18)]


# F: FLOAT[7, 2, 30], 420 value(s)
INIT_F_0 = [0.40548014640808105, 0.40548014640808105, 0.40548014640808105, 0.40548014640808105, 0.40548014640808105,
 0.40548014640808105, 0.40548014640808105, 0.40548014640808105, 0.40548014640808105, 0.40548014640808105,
 0.40548014640808105, 0.40548014640808105, 0.40548014640808105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.39499786496162415, -0.39499786496162415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.405480146408081, 0.0, -1.0,
 0.5108729600906372, 0.4523555338382721, 0.2902089059352875, 0.06157892942428589, -0.18115805089473724,
 -0.38239389657974243, -0.4960279166698456, -0.4960279166698456, -0.38239389657974243, -0.18115805089473724,
 0.06157892942428589, 0.2902089059352875, 0.4523555338382721, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.4642268121242523, -0.4642268121242523, 0.0, 0.23741450905799866, 0.4204401969909668,
 0.5071481466293335, 0.4776745140552521, 0.33877143263816833, 0.12225989997386932, -0.12225989997386932,
 -0.33877143263816833, -0.4776745140552521, -0.5071481466293335, -0.4204401969909668, -0.23741450905799866, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5108729600906372, 0.0, 0.0, 0.5108729600906372,
 0.2902089059352875, -0.18115805089473724, -0.4960279166698456, -0.38239389657974243, 0.06157892942428589,
 0.4523555338382721, 0.4523555338382721, 0.06157892942428589, -0.38239389657974243, -0.4960279166698456,
 -0.18115805089473724, 0.2902089059352875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.37156930565834045, -0.37156930565834045, 0.0, 0.4204401969909668, 0.4776745140552521, 0.12225989997386932,
 -0.33877143263816833, -0.5071481466293335, -0.23741450905799866, 0.23741450905799866, 0.5071481466293335,
 0.33877143263816833, -0.12225989997386932, -0.4776745140552521, -0.4204401969909668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5108729600906372, 0.0, 0.0, 0.5108729600906372, 0.06157892942428589,
 -0.4960279166698456, -0.18115805089473724, 0.4523555338382721, 0.2902089059352875, -0.38239389657974243,
 -0.38239389657974243, 0.2902089059352875, 0.4523555338382721, -0.18115805089473724, -0.4960279166698456,
 0.06157892942428589, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24092040956020355,
 -0.24092040956020355, 0.0, 0.5071481466293335, 0.12225989997386932, -0.4776745140552521, -0.23741450905799866,
 0.4204401969909668, 0.33877143263816833, -0.33877143263816833, -0.4204401969909668, 0.23741450905799866,
 0.4776745140552521, -0.12225989997386932, -0.5071481466293335, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.5108729600906372, 0.0, 0.0, 0.5108729600906372, -0.18115805089473724, -0.38239389657974243,
 0.4523555338382721, 0.06157892942428589, -0.4960279166698456, 0.2902089059352875, 0.2902089059352875,
 -0.4960279166698456, 0.06157892942428589, 0.4523555338382721, -0.38239389657974243, -0.18115805089473724, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1022101566195488, -0.1022101566195488, 0.0,
 0.4776745140552521, -0.33877143263816833, -0.23741450905799866, 0.5071481466293335, -0.12225989997386932,
 -0.4204401969909668, 0.4204401969909668, 0.12225989997386932, -0.5071481466293335, 0.23741450905799866,
 0.33877143263816833, -0.4776745140552521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.5108729600906372, 0.0, 0.0, 0.5108729600906372, -0.38239389657974243, 0.06157892942428589, 0.2902089059352875,
 -0.4960279166698456, 0.4523555338382721, -0.18115805089473724, -0.18115805089473724, 0.4523555338382721,
 -0.4960279166698456, 0.2902089059352875, 0.06157892942428589, -0.38239389657974243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.012784580700099468, 0.012784580700099468, 0.0, 0.33877143263816833,
 -0.5071481466293335, 0.4204401969909668, -0.12225989997386932, -0.23741450905799866, 0.4776745140552521,
 -0.4776745140552521, 0.23741450905799866, 0.12225989997386932, -0.4204401969909668, 0.5071481466293335,
 -0.33877143263816833, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5108729600906372, 0.0,
 0.0, 0.5108729600906372, -0.4960279166698456, 0.4523555338382721, -0.38239389657974243, 0.2902089059352875,
 -0.18115805089473724, 0.06157892942428589, 0.06157892942428589, -0.18115805089473724, 0.2902089059352875,
 -0.38239389657974243, 0.4523555338382721, -0.4960279166698456, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, -0.07771989703178406, 0.07771989703178406, 0.0, 0.12225989997386932, -0.23741450905799866,
 0.33877143263816833, -0.4204401969909668, 0.4776745140552521, -0.5071481466293335, 0.5071481466293335,
 -0.4776745140552521, 0.4204401969909668, -0.33877143263816833, 0.23741450905799866, -0.12225989997386932, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5108729600906372, 0.0, 0.0]

# CBAS: FLOAT[2, 2, 10], 40 value(s)
INIT_CBAS_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0999999046325684, -0.49000000953674316, -0.949999988079071,
 -1.149999976158142, -0.3266666829586029, -0.2177777886390686, -0.145185187458992, -0.0967901274561882,
 -0.0645267516374588, -0.043017834424972534, 0.0, 1.0, 1.0, 2.0, 1.5, 2.25, 3.375, 5.0625, 7.59375, 11.390625, 0.0, 0.0,
 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# TRP: FLOAT[2, 2, 2], 8 value(s)
INIT_TRP_2 = [1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0]

# CGU: FLOAT[2, 2], 4 value(s)
INIT_CGU_3 = [64289612.0, 30614014.0, 383502272.0, -383496608.0]

# ZA: FLOAT[2, 2, 2], 8 value(s)
INIT_ZA_4 = [2.2165422706166282e-05, 2.216541724919807e-05, 2.8768530683009885e-05, 2.876864527934231e-05, 3.1114584544411628e-06,
 -5.293983122101054e-06, -1.0100966392201371e-06, 3.1926235806167824e-06]

# ZB: FLOAT[2, 2], 4 value(s)
INIT_ZB_5 = [0.0, 1.0, 1.0, -1.0]


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
        # 0: Einsum inputs=46 outputs=1
        _node(
            'Einsum',
            ['ZB', 'TRP', 'F', 'F', 'F', 'F', 'F', 'input', 'F', 'CGU', 'CBAS', 'CBAS', 'input', 'F', 'F',
             'TRP', 'F', 'F', 'TRP', 'F', 'F', 'input', 'F', 'F', 'CBAS', 'ZA', 'ZB', 'CBAS', 'F', 'F', 'F',
             'F', 'input', 'F', 'F', 'TRP', 'F', 'F', 'input', 'F', 'F', 'TRP', 'F', 'F', 'input', 'CBAS'],
            ['output'],
            name='output',
            domain='',
            equation='ZZ,VVV,JVK,JZG,DVG,HVG,DEo,...nop,HIp,bS,bSn,hYn,...nkl,aqk,yAl,qxw,DEi,awi,ABC,HIj,yBj,...mij,WXj,TUi,bbm,bOC,bv,OOm,Tvz,WvF,TUt,WXu,...mtu,LNt,PRu,MNx,axd,Lxd,...gde,yCe,PCe,QCR,PQs,LMr,...frs,Yhc->...crs',
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
            _tensor('F', TensorProto.FLOAT, (7, 2, 30), INIT_F_0),
            _tensor('CBAS', TensorProto.FLOAT, (2, 2, 10), INIT_CBAS_1),
            _tensor('TRP', TensorProto.FLOAT, (2, 2, 2), INIT_TRP_2),
            _tensor('CGU', TensorProto.FLOAT, (2, 2), INIT_CGU_3),
            _tensor('ZA', TensorProto.FLOAT, (2, 2, 2), INIT_ZA_4),
            _tensor('ZB', TensorProto.FLOAT, (2, 2), INIT_ZB_5),
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
