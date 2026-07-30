from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task363'
TASK_NUM = 363
KAGGLE = {'score': 19.67699, 'date': '2026-07-15'}
MEMORY_BYTES = 86
PARAMS = 119
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task363_packed2_factored'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# W: INT64[2, 10], 20 value(s)
INIT_W_1 = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]

# K: INT64[4, 2], 8 value(s)
INIT_K_2 = [-1, -2, -1, -1, 1, 0, 0, 1]

# U: INT64[2, 30], 60 value(s)
INIT_U_3 = [1, 2, 4, 8, 16, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4,
 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# CASES: STRING[30], 30 value(s)
INIT_CASES_4 = ['4977651903177932272,8491440267354769474,-2816786923789763024,-8506200240529934400',
 '-30227943263576320,132602077771760642,209324592600961664,-1263269973435909120',
 '954833430170828800,-1476898671332163582,8611234788433982976,-7903197372774088704',
 '4107423599822897152,-8214847199645794302,4995739851673427824,6196051930259652608',
 '37163501675872128,-74327003351743998,-762599317813984896,-1220144494075183616',
 '162578835693501568,-325157656338100734,-1148986213014205840,-7102858802385915392',
 '5908874041460260352,6629001493723744258,-6485927562794793824,-2594919187754256384',
 '4578470116951326720,-8940613305400033278,-4956621638669963392,-9206340517208522752',
 '4662735204132012032,9130289742464876546,-7826843869200657536,-8908096064389840896',
 '4654692678355644552,9137358716998270434,4093689952577203816,-8965332077285228000',
 '72095848506081280,-144191697011671038,-3542229608123653184,-8214336193514373120',
 '793401643927338528,2736652613125670786,3421946975273975184,9083865764544379008',
 '2414091677573578752,-4828182426377256958,-397433591346322064,-3962194937037979648',
 '426813629187209160,-815215819658026782,-1542449801437597528,4905135119783616288',
 '238690982315302912,-477381962617094142,-3933064686896570912,1432129403230912512',
 '2325160812185911296,-4650319562787520510,-2399784845682768032,-6657504960147619840',
 '3153989512665230848,-6307979010825500670,1435103669035250416,477193016786794496',
 '5310113264169122048,8331415210874117122,-4817099146214058816,-4527910896901561344',
 '1677731701830189056,-2815031448375918590,5779022891345040816,-7839938400118374400',
 '7210293270786408256,4026170739174146818,1417455548901061696,5935886291961117952',
 '6710329530772357088,5170587451705524354,1779895243563644896,-1290313373845553280',
 '36211074927689728,-72422148848746494,-108520735313105152,-8429644833978580992',
 '641129618402674688,-1282259215309479934,-765154689051302912,3846777731911917568',
 '13531900190654464,-27063800381308926,9165887956825323504,-8502036169926639616',
 '2655217372863922176,-5310434745224527870,-365737587260391936,-2515439852128829440',
 '2313034403720200192,-4620431611324792830,-2381900162579825280,-4562903207809908736',
 '-3456456770371715264,-4609539508814216446,-8211335412690519488,4199526698938531072',
 '4232893836373704704,-8462406157518700542,-5504321313080379424,6954000459761385472',
 '4685936397448248960,9075293903864202754,-4811702631341697024,9092624871202486784',
 '1772834976956397296,-3545665517173791678,2765231483072329136,-8404786035136996416']


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
        # 0: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['LIKE'],
            ['rng_f'],
            name='',
            domain='',
            dtype=10,
            high=28.605682373046875,
            low=-26.08678436279297,
            seed=5402863.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng_f'],
            ['case_i'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'case_i'],
            ['case_string'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_string'],
            ['parts', 'lengths'],
            name='',
            domain='',
            delimiter=',',
            maxsplit=3,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='',
            domain='',
            to=7,
        ),
        # 5: Einsum inputs=22 outputs=1
        _node(
            'Einsum',
            ['packed', 'K', 'K', 'K', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'K', 'K', 'K', 'K',
             'K', 'K', 'W'],
            ['output'],
            name='',
            domain='',
            equation='bm,ms,mt,mt,tr,tr,tr,tr,tr,tr,tr,tr,tr,tr,uc,au,du,eu,fu,gu,hu,so->borc',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT64, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('LIKE', TensorProto.FLOAT16, (1,), INIT_LIKE_0),
            _tensor('W', TensorProto.INT64, (2, 10), INIT_W_1),
            _tensor('K', TensorProto.INT64, (4, 2), INIT_K_2),
            _tensor('U', TensorProto.INT64, (2, 30), INIT_U_3),
            _tensor('CASES', TensorProto.STRING, (30,), INIT_CASES_4),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 4]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 4]),
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
