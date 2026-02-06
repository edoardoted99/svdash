import pytest
from kg_parser.parser import InternalKG, KGNode, KGEdge
from kg_parser.diff import compute_diff, OpType, AtomicOp

def test_diff_add_node():
    kg1 = InternalKG(nodes=[], edges=[])
    kg2 = InternalKG(nodes=[KGNode(id="n1", types=["T1"])], edges=[])
    
    diff = compute_diff(kg1, kg2)
    assert len(diff) == 1
    assert diff[0].op_type == OpType.ADD_NODE
    assert diff[0].target_id == "n1"

def test_diff_remove_node():
    kg1 = InternalKG(nodes=[KGNode(id="n1", types=["T1"])], edges=[])
    kg2 = InternalKG(nodes=[], edges=[])
    
    diff = compute_diff(kg1, kg2)
    assert len(diff) == 1
    assert diff[0].op_type == OpType.REMOVE_NODE
    assert diff[0].target_id == "n1"

def test_diff_update_prop():
    kg1 = InternalKG(nodes=[KGNode(id="n1", types=["T1"], properties={"p1": "v1"})], edges=[])
    kg2 = InternalKG(nodes=[KGNode(id="n1", types=["T1"], properties={"p1": "v2"})], edges=[])
    
    diff = compute_diff(kg1, kg2)
    assert len(diff) == 1
    assert diff[0].op_type == OpType.UPDATE_PROP
    assert diff[0].target_id == "n1"
    assert diff[0].payload['property'] == "p1"
    assert diff[0].payload['value'] == "v2"

def test_diff_retype():
    kg1 = InternalKG(nodes=[KGNode(id="n1", types=["T1"])], edges=[])
    kg2 = InternalKG(nodes=[KGNode(id="n1", types=["T2"])], edges=[])
    
    diff = compute_diff(kg1, kg2)
    assert len(diff) == 1
    assert diff[0].op_type == OpType.RETYPE
    assert diff[0].target_id == "n1"
    assert set(diff[0].payload) == {"T2"}

def test_diff_add_edge():
    kg1 = InternalKG(nodes=[KGNode("n1"), KGNode("n2")], edges=[])
    kg2 = InternalKG(nodes=[KGNode("n1"), KGNode("n2")], edges=[KGEdge("n1", "n2", "p1")])
    
    diff = compute_diff(kg1, kg2)
    assert len(diff) == 1
    assert diff[0].op_type == OpType.ADD_EDGE
    assert diff[0].payload.source_id == "n1"
    assert diff[0].payload.target_id == "n2"

def test_diff_remove_edge():
    kg1 = InternalKG(nodes=[KGNode("n1"), KGNode("n2")], edges=[KGEdge("n1", "n2", "p1")])
    kg2 = InternalKG(nodes=[KGNode("n1"), KGNode("n2")], edges=[])
    
    diff = compute_diff(kg1, kg2)
    assert len(diff) == 1
    assert diff[0].op_type == OpType.REMOVE_EDGE
    assert diff[0].payload.predicate == "p1"
