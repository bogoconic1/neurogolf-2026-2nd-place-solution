from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task284'
TASK_NUM = 284
KAGGLE = {'score': 19.732142, 'date': '2026-07-15'}
MEMORY_BYTES = 52
PARAMS = 142
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task284_terminal_matrix_axis_recover'
OPSETS = [('', 15)]


# nonbg_c: FLOAT[2, 10], 20 value(s)
INIT_NONBG_C_0 = [1.0, -2000000.125, -2000000.125, -2000000.125, -2000000.125, -2000000.125, -2000000.125, -2000000.125, -2000000.125,
 -2000000.125, 0.0, 1000000.0625, 1000000.0625, 1000000.0625, 1000000.0625, 1000000.0625, 1000000.0625, 1000000.0625,
 1000000.0625, 1000000.0625]

# coord_basis: FLOAT[2, 30], 60 value(s)
INIT_COORD_BASIS_1 = [0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.002041163155809045,
 -0.22012192010879517, -0.06140875816345215, 0.32908499240875244, -0.29150527715682983, 0.01605190709233284, 0.0,
 0.009999999776482582, 0.019999999552965164, 0.029999999329447746, 0.03999999910593033, 0.05000000074505806,
 0.05999999865889549, 0.07000000029802322, 0.07999999821186066, 0.09000000357627869, 0.10000000149011612,
 0.10999999940395355, 0.11999999731779099, 0.12999999523162842, 0.14000000059604645, 0.15000000596046448,
 0.1599999964237213, 0.17000000178813934, 0.18000000715255737, 0.1899999976158142, 0.20000000298023224,
 0.20999999344348907, 0.2199999988079071, 0.23000000417232513, -0.9055981040000916, -0.3552119731903076,
 -0.6244771480560303, 0.40463510155677795, -0.2856300175189972, -0.9866464734077454]

# cross_mode: FLOAT[2, 2, 2], 8 value(s)
INIT_CROSS_MODE_2 = [1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 1.0, 0.0]

# pow_exp: FLOAT[2], 2 value(s)
INIT_POW_EXP_3 = [0.5, 0.0]

# source_B: FLOAT[2, 2], 4 value(s)
INIT_SOURCE_B_4 = [-0.5, 1.0, -0.5, -1.0]

# poly_W2: FLOAT[2, 2, 5, 2], 40 value(s)
INIT_POLY_W2_5 = [-3.0204296308511402e-06, -4.51240822485488e-07, 3.826657484751195e-05, 4.060475475853309e-06, 7.188790186773986e-05,
 7.734314749541227e-06, -2.3577953470521607e-05, -2.7794596917374292e-06, 3.036833959413343e-06, 3.8303562632791e-07,
 3.1622775509276835e-07, -3.1622775509276835e-07, -1.5811387754638417e-07, -1.5811387754638417e-07,
 -6.324555101855367e-07, 0.0, -1.1067971854572534e-06, 1.5811387754638417e-07, -1.5811388038855512e-06,
 3.1622775509276835e-07, -8.077620805124752e-06, -3.366617988831422e-07, -2.0618752387235872e-05,
 -2.1940909391560126e-06, 1.1063118108722847e-05, 1.0399410257377895e-06, -7.594539965793956e-06,
 -8.487132845402812e-07, 3.5640211990539683e-06, -1.566958189869183e-07, 2.854112324257585e-07, 2.3125968695580923e-08,
 -1.397459072904894e-07, 1.4118133329077409e-08, 4.2604759187270247e-07, 1.8154178604845583e-08, 2.1915863612775865e-07,
 1.5117670670861116e-08, -7.547342306679639e-08, -5.6561066941185345e-09]

# main_coord_factor: FLOAT[2, 2, 2], 8 value(s)
INIT_MAIN_COORD_FACTOR_6 = [10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0, 0.0]


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
        # 0: Einsum inputs=30 outputs=1
        _node(
            'Einsum',
            ['coord_basis', 'coord_basis', 'input', 'nonbg_c', 'coord_basis', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'cross_mode',
             'source_B', 'cross_mode', 'cross_mode', 'source_B', 'cross_mode', 'coord_basis', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'coord_basis',
             'coord_basis', 'input', 'nonbg_c', 'coord_basis', 'coord_basis', 'source_B', 'cross_mode'],
            ['dist2_f'],
            name='dist2_f',
            domain='',
            equation='ti,ki,ndij,xd,uj,lj,tpb,utb,kpf,lkf,zab,gm,mzg,qef,oB,Bqo,aA,eE,rpa,sra,vpe,yve,rh,vh,nchw,xc,sw,yw,CD,DxC->n',
        ),
        # 1: Pow inputs=2 outputs=1
        _node(
            'Pow',
            ['dist2_f', 'pow_exp'],
            ['half_feat'],
            name='half_feat_pow',
            domain='',
        ),
        # 2: Einsum inputs=36 outputs=1
        _node(
            'Einsum',
            ['coord_basis', 'coord_basis', 'input', 'nonbg_c', 'coord_basis', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'cross_mode',
             'source_B', 'cross_mode', 'cross_mode', 'source_B', 'cross_mode', 'coord_basis', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'main_coord_factor', 'coord_basis',
             'coord_basis', 'input', 'nonbg_c', 'coord_basis', 'coord_basis', 'source_B', 'cross_mode',
             'pow_exp', 'pow_exp', 'pow_exp', 'pow_exp', 'pow_exp', 'pow_exp'],
            ['orient_f'],
            name='orient_f',
            domain='',
            equation='ti,ki,ndij,xd,uj,lj,tpb,utb,kpf,lkf,zab,gm,mzg,qef,oF,Fqo,aA,eE,rpa,sra,vpe,yve,rh,vh,nchw,xc,sw,yw,GH,HxG,B,B,C,C,D,D->p',
        ),
        # 3: Einsum inputs=11 outputs=1
        _node(
            'Einsum',
            ['nonbg_c', 'coord_basis', 'input', 'coord_basis', 'main_coord_factor', 'main_coord_factor',
             'orient_f', 'cross_mode', 'source_B', 'cross_mode', 'cross_mode'],
            ['terminal_cross'],
            name='terminal_feat',
            domain='',
            equation='xc,ah,nchw,bw,aof,baf,o,UfV,de,exd,RSf->RS',
        ),
        # 4: Einsum inputs=15 outputs=1
        _node(
            'Einsum',
            ['nonbg_c', 'coord_basis', 'input', 'coord_basis', 'main_coord_factor', 'main_coord_factor',
             'orient_f', 'main_coord_factor', 'main_coord_factor', 'source_B', 'cross_mode', 'cross_mode',
             'source_B', 'cross_mode', 'cross_mode'],
            ['cross_proj'],
            name='cross_feat',
            domain='',
            equation='xc,ah,nchw,bw,apf,baf,o,pox,spx,de,exd,UfV,gi,ixg,qCf->qC',
        ),
        # 5: Einsum inputs=72 outputs=1
        _node(
            'Einsum',
            ['cross_proj', 'main_coord_factor', 'main_coord_factor', 'coord_basis', 'coord_basis', 'cross_proj',
             'coord_basis', 'main_coord_factor', 'main_coord_factor', 'coord_basis', 'coord_basis',
             'coord_basis', 'main_coord_factor', 'main_coord_factor', 'cross_proj', 'cross_proj', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'coord_basis', 'coord_basis', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'terminal_cross', 'poly_W2', 'source_B', 'poly_W2',
             'poly_W2', 'source_B', 'terminal_cross', 'main_coord_factor', 'main_coord_factor', 'coord_basis',
             'coord_basis', 'poly_W2', 'poly_W2', 'pow_exp', 'source_B', 'terminal_cross', 'main_coord_factor',
             'main_coord_factor', 'coord_basis', 'coord_basis', 'source_B', 'terminal_cross',
             'main_coord_factor', 'main_coord_factor', 'coord_basis', 'coord_basis', 'main_coord_factor',
             'source_B', 'cross_mode', 'orient_f', 'orient_f', 'input', 'coord_basis', 'input', 'coord_basis',
             'main_coord_factor', 'main_coord_factor', 'source_B', 'source_B', 'cross_mode', 'source_B',
             'nonbg_c', 'half_feat', 'cross_mode', 'half_feat', 'terminal_cross', 'source_B', 'source_B'],
            ['output'],
            name='output',
            domain='',
            equation='aC,bZC,dbC,dw,bh,aE,mw,mkE,kZE,kh,sw,qh,sqF,qZF,pF,pG,xw,xvG,vZG,vh,Kw,Jh,KJL,JoL,IL,apjr,gI,uijI,uijM,gM,MQ,OoQ,POQ,Pw,Oh,uijV,uijR,u,gR,RU,SoU,TSU,Tw,Sh,gV,VY,WoY,XWY,Xw,Wh,Zoi,AB,BiA,o,o,...Dhw,nt,...ctH,yH,nof,ynf,ff,ff,Nef,gN,Nc,N,irl,l,eu,ue,zz->...chw',
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
            _tensor('nonbg_c', TensorProto.FLOAT, (2, 10), INIT_NONBG_C_0),
            _tensor('coord_basis', TensorProto.FLOAT, (2, 30), INIT_COORD_BASIS_1),
            _tensor('cross_mode', TensorProto.FLOAT, (2, 2, 2), INIT_CROSS_MODE_2),
            _tensor('pow_exp', TensorProto.FLOAT, (2,), INIT_POW_EXP_3),
            _tensor('source_B', TensorProto.FLOAT, (2, 2), INIT_SOURCE_B_4),
            _tensor('poly_W2', TensorProto.FLOAT, (2, 2, 5, 2), INIT_POLY_W2_5),
            _tensor('main_coord_factor', TensorProto.FLOAT, (2, 2, 2), INIT_MAIN_COORD_FACTOR_6),
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
