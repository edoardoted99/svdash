from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Any
from .parser import InternalKG

class OpType(Enum):
    ADD_NODE = "ADD_NODE"
    REMOVE_NODE = "REMOVE_NODE"
    ADD_EDGE = "ADD_EDGE"
    REMOVE_EDGE = "REMOVE_EDGE"
    UPDATE_PROP = "UPDATE_PROP"
    RETYPE = "RETYPE"

@dataclass(frozen=True)
class AtomicOp:
    """
    Represents an atomic operation to transform a KG.
    """
    op_type: OpType
    target_id: str # ID of the node or edge (source_id for edge, or tuple key representation)
    payload: Any = None # meaningful data for the operation (e.g., new property value, new types)

def compute_diff(kg_old: InternalKG, kg_new: InternalKG) -> List[AtomicOp]:
    """
    Computes the difference between two KGs, returning a list of atomic operations
    to transform kg_old into kg_new.
    """
    ops: List[AtomicOp] = []

    # Map nodes by ID for easier lookup
    old_nodes = {node.id: node for node in kg_old.nodes}
    new_nodes = {node.id: node for node in kg_new.nodes}

    # 1. Node Additions and Removals
    for nid in new_nodes:
        if nid not in old_nodes:
            ops.append(AtomicOp(OpType.ADD_NODE, nid, payload=asdict(new_nodes[nid])))
    
    for nid in old_nodes:
        if nid not in new_nodes:
            ops.append(AtomicOp(OpType.REMOVE_NODE, nid))

    # 2. Node Updates (for nodes present in both)
    common_ids = set(old_nodes.keys()) & set(new_nodes.keys())
    
    for nid in common_ids:
        old_node = old_nodes[nid]
        new_node = new_nodes[nid]
        
        # 2a. Retype
        if set(old_node.types) != set(new_node.types):
            ops.append(AtomicOp(OpType.RETYPE, nid, payload=new_node.types))
        
        # 2b. Property Updates
        all_props = set(old_node.properties.keys()) | set(new_node.properties.keys())
        for prop in all_props:
            old_val = old_node.properties.get(prop)
            new_val = new_node.properties.get(prop)
            
            if old_val != new_val:
                # We treat property changes as UPDATE_PROP.
                # If new_val is None (removed), payload could reflect that implementation detail.
                # Here we assume payload=(prop_name, new_value)
                ops.append(AtomicOp(OpType.UPDATE_PROP, nid, payload={'property': prop, 'value': new_val, 'old_value': old_val}))

    # 3. Edge Additions and Removals
    # Represent edges as tuples to compare sets
    old_edges = {(e.source_id, e.target_id, e.predicate): e for e in kg_old.edges}
    new_edges = {(e.source_id, e.target_id, e.predicate): e for e in kg_new.edges}
    
    for edge_key in new_edges:
        if edge_key not in old_edges:
            # target_id for edge op is source_id usually, but we need to identify the edge uniquely
            # Let's use source_id as target_id and payload contains full edge info
            ops.append(AtomicOp(OpType.ADD_EDGE, edge_key[0], payload=asdict(new_edges[edge_key])))
            
    for edge_key in old_edges:
        if edge_key not in new_edges:
            ops.append(AtomicOp(OpType.REMOVE_EDGE, edge_key[0], payload=asdict(old_edges[edge_key])))

    return ops
