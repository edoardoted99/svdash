from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pyld import jsonld

# --- Dataclasses ---

@dataclass(frozen=True)
class KGNode:
    """
    Represents a node in the internal Knowledge Graph.
    """
    id: str
    types: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class KGEdge:
    """
    Represents an edge (relationship) between two nodes in the internal Knowledge Graph.
    """
    source_id: str
    target_id: str
    predicate: str

@dataclass
class InternalKG:
    """
    Represents the internal Knowledge Graph structure.
    """
    nodes: List[KGNode] = field(default_factory=list)
    edges: List[KGEdge] = field(default_factory=list)

    def get_node(self, node_id: str) -> Optional[KGNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

# --- Parser Functions ---

def parse_jsonld(data: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> InternalKG:
    """
    Parses a JSON-LD Schema.org dictionary into an InternalKG.
    Expands the JSON-LD to normalize structure, then extracts nodes and edges.
    """
    # Expand the JSON-LD to resolve context and normalize structure
    expanded = jsonld.expand(data, options)
    
    nodes: List[KGNode] = []
    edges: List[KGEdge] = []
    
    # Map to store temporary node data before creating frozen dataclasses
    node_data_map: Dict[str, Dict[str, Any]] = {}

    def process_element(element: Dict[str, Any]):
        node_id = element.get('@id')
        if not node_id:
            # Skip nodes without ID for now, or generate one if strictly needed
            # For this implementation, we assume significant nodes have IDs
            return

        node_types = element.get('@type', [])
        # Ensure types is a list
        if isinstance(node_types, str):
            node_types = [node_types]
        
        if node_id not in node_data_map:
             node_data_map[node_id] = {
                'id': node_id,
                'types': node_types,
                'properties': {}
            }
        
        # Process properties
        for key, value in element.items():
            if key.startswith('@'):
                continue
            
            # Helper to process values
            def extract_refs(val):
                if isinstance(val, dict):
                    if '@id' in val:
                        # It's a reference (edge)
                        edges.append(KGEdge(source_id=node_id, target_id=val['@id'], predicate=key))
                    # Check for nested definitions (blank nodes or embedded) - simplified for now
                    # recursive processing could go here
                elif isinstance(val, list):
                    for item in val:
                        extract_refs(item)
            
            # If value is a literal, add to properties. If it points to a node, add edge.
            # In expanded JSON-LD, values are usually arrays of dicts with @value or @id
            if isinstance(value, list):
                literals = []
                for item in value:
                    if isinstance(item, dict):
                         if '@value' in item:
                             literals.append(item['@value'])
                         elif '@id' in item:
                             edges.append(KGEdge(source_id=node_id, target_id=item['@id'], predicate=key))
                
                if literals:
                     # Store single value if one, else list
                    node_data_map[node_id]['properties'][key] = literals[0] if len(literals) == 1 else literals

    # Root of expanded can be list or dict
    if isinstance(expanded, list):
        for item in expanded:
            process_element(item)
    elif isinstance(expanded, dict):
        process_element(expanded)
        
    # Create KGNode objects
    for nid, data in node_data_map.items():
        nodes.append(KGNode(id=data['id'], types=data['types'], properties=data['properties']))

    return InternalKG(nodes=nodes, edges=edges)

def serialize_to_jsonld(kg: InternalKG, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Serializes an InternalKG back to a JSON-LD dictionary.
    Compacts the output using the Schema.org context.
    """
    graph_nodes = []
    
    # Reconstruct the graph-like structure (flattened)
    for node in kg.nodes:
        node_obj = {
            '@id': node.id,
            '@type': node.types,
        }
        # Add properties
        for k, v in node.properties.items():
            node_obj[k] = v
        
        # Add edges (outgoing)
        for edge in kg.edges:
            if edge.source_id == node.id:
                if edge.predicate not in node_obj:
                     node_obj[edge.predicate] = []
                
                # Retrieve existing list or ensure it is one (tho we initialize it above)
                if not isinstance(node_obj[edge.predicate], list):
                     node_obj[edge.predicate] = [node_obj[edge.predicate]] # Should not happen based on logic above
                
                node_obj[edge.predicate].append({'@id': edge.target_id})

        # Cleanup single item lists for edges if desired, but pyld handles compacting well
        graph_nodes.append(node_obj)

    doc = {
        '@graph': graph_nodes
    }
    
    context = {"@context": "https://schema.org"}
    
    if options is None:
        options = {}
    
    if 'documentLoader' not in options:
        from .loader import get_local_document_loader
        options['documentLoader'] = get_local_document_loader()
        
    compacted = jsonld.compact(doc, context, options)
    return compacted
