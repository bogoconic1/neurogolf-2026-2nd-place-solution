from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task169'
TASK_NUM = 169
KAGGLE = {'score': 19.454823, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 256
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task169_updated_v6_frontier'
OPSETS = [('', 12)]


# U: FLOAT[5, 30], 150 value(s)
INIT_U_0 = [0.12013116478919983, 0.2305300086736679, 0.32225269079208374, 0.38786837458610535, 0.4220612943172455,
 0.4220612943172455, 0.38786840438842773, 0.32225272059440613, 0.23053009808063507, 0.12013117969036102,
 -0.0023350692354142666, -0.19423365592956543, -0.0023350692354142666, -0.19423365592956543, -0.0023350692354142666,
 -0.19423365592956543, -0.0023350692354142666, -0.19423365592956543, -0.0023350692354142666, -0.19423365592956543,
 -0.0023350692354142666, -0.19423365592956543, -0.0023350692354142666, -0.19423365592956543, -0.0023350692354142666,
 -0.19423365592956543, -0.0023350692354142666, -0.19423365592956543, -0.0023350692354142666, -0.19423365592956543,
 0.2305300086736679, 0.38786837458610535, 0.4220612943172455, 0.32225272059440613, 0.12013117969036102,
 -0.12013106793165207, -0.32225263118743896, -0.4220612943172455, -0.3878684639930725, -0.2305300533771515,
 0.12786521017551422, -0.027865217998623848, 0.12786521017551422, -0.027865217998623848, 0.12786521017551422,
 -0.027865217998623848, 0.12786521017551422, -0.027865217998623848, 0.12786521017551422, -0.027865217998623848,
 0.12786521017551422, -0.027865217998623848, 0.12786521017551422, -0.027865217998623848, 0.12786521017551422,
 -0.027865217998623848, 0.12786521017551422, -0.027865217998623848, 0.12786521017551422, -0.027865217998623848,
 0.32225269079208374, 0.4220612943172455, 0.23053009808063507, -0.12013106793165207, -0.38786837458610535,
 -0.3878684639930725, -0.1201312467455864, 0.23052984476089478, 0.4220612645149231, 0.3222527503967285,
 0.0688016414642334, -0.06217051297426224, 0.0688016414642334, -0.06217051297426224, 0.0688016414642334,
 -0.06217051297426224, 0.0688016414642334, -0.06217051297426224, 0.0688016414642334, -0.06217051297426224,
 0.0688016414642334, -0.06217051297426224, 0.0688016414642334, -0.06217051297426224, 0.0688016414642334,
 -0.06217051297426224, 0.0688016414642334, -0.06217051297426224, 0.0688016414642334, -0.06217051297426224,
 0.38786837458610535, 0.32225272059440613, -0.12013106793165207, -0.4220612943172455, -0.2305300533771515,
 0.23052984476089478, 0.4220612943172455, 0.12013130635023117, -0.322252482175827, -0.3878684341907501,
 0.07783990353345871, 0.022160079330205917, 0.07783990353345871, 0.022160079330205917, 0.07783990353345871,
 0.022160079330205917, 0.07783990353345871, 0.022160079330205917, 0.07783990353345871, 0.022160079330205917,
 0.07783990353345871, 0.022160079330205917, 0.07783990353345871, 0.022160079330205917, 0.07783990353345871,
 0.022160079330205917, 0.07783990353345871, 0.022160079330205917, 0.07783990353345871, 0.022160079330205917,
 0.4220612943172455, 0.12013117969036102, -0.38786837458610535, -0.2305300533771515, 0.32225269079208374,
 0.3222527503967285, -0.23052994906902313, -0.3878684341907501, 0.12013107538223267, 0.4220612943172455,
 0.03962681442499161, 0.011163842864334583, 0.03962681442499161, 0.011163842864334583, 0.03962681442499161,
 0.011163842864334583, 0.03962681442499161, 0.011163842864334583, 0.03962681442499161, 0.011163842864334583,
 0.03962681442499161, 0.011163842864334583, 0.03962681442499161, 0.011163842864334583, 0.03962681442499161,
 0.011163842864334583, 0.03962681442499161, 0.011163842864334583, 0.03962681442499161, 0.011163842864334583]

# P: FLOAT[2, 30], 60 value(s)
INIT_P_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

# Grel: FLOAT[3, 2, 2], 12 value(s)
INIT_GREL_2 = [1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.5, 1.5, 0.0]

# C: FLOAT[10, 3], 30 value(s)
INIT_C_3 = [0.7785810828208923, 0.8145315647125244, 0.273438960313797, 9.492019653320312, -13.79129409790039, -0.7166793346405029,
 520.5447387695312, -30524.658203125, 87.55093383789062, -6.206857204437256, 0.0, 0.5173301100730896, 0.0, 0.0, 0.0,
 -0.07011885195970535, -0.0013661086559295654, 0.2281186431646347, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0]

# LC: FLOAT[2, 2], 4 value(s)
INIT_LC_4 = [1.0, 1.0, 1.0, -1.0]


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
        # 0: Einsum inputs=74 outputs=1
        _node(
            'Einsum',
            ['P', 'U', 'LC', 'U', 'P', 'input', 'P', 'U', 'U', 'P', 'LC', 'Grel', 'LC', 'Grel', 'LC', 'LC',
             'Grel', 'LC', 'LC', 'U', 'P', 'U', 'P', 'input', 'P', 'U', 'P', 'U', 'U', 'P', 'LC', 'Grel', 'LC',
             'P', 'U', 'U', 'P', 'U', 'P', 'input', 'P', 'U', 'P', 'U', 'U', 'P', 'LC', 'Grel', 'LC', 'P', 'U',
             'U', 'P', 'U', 'P', 'input', 'P', 'U', 'P', 'U', 'C', 'U', 'P', 'LC', 'Grel', 'LC', 'P', 'U', 'U',
             'P', 'input', 'U', 'P', 'C'],
            ['output'],
            name='',
            domain='',
            equation='hC,dC,hf,dA,fA,nayA,Vy,Uy,UB,gB,gV,qgh,JL,qLJ,ML,JM,qMK,KJ,KK,Uw,Vw,dx,fx,nawx,ex,bx,Tw,Sw,SD,iD,iT,qij,je,jE,bE,Su,Tu,bv,ev,nauv,Zv,Yv,Ru,Qu,QF,kF,kR,qkl,lZ,lG,YG,Qs,Rs,Yt,Zt,nast,Xt,Wt,Ps,Os,aq,WI,pI,pX,qmp,mP,mH,OH,Or,Pr,narc,Wc,Xc,oq->norc',
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
            _tensor('U', TensorProto.FLOAT, (5, 30), INIT_U_0),
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_1),
            _tensor('Grel', TensorProto.FLOAT, (3, 2, 2), INIT_GREL_2),
            _tensor('C', TensorProto.FLOAT, (10, 3), INIT_C_3),
            _tensor('LC', TensorProto.FLOAT, (2, 2), INIT_LC_4),
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
