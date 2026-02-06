import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from .parser import InternalKG, KGNode

@dataclass(frozen=True)
class ValidationError:
    """
    Represents a single validation error in the Knowledge Graph.
    """
    message: str
    node_id: Optional[str] = None
    property_name: Optional[str] = None

@dataclass
class ValidationResult:
    """
    Result of a KG validation process.
    """
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)

class SchemaOrgValidator:
    """
    Validator for InternalKG against Schema.org ontology.
    """
    
    def __init__(self, schema_path: str = "kg_parser/schemaorg-current-https.jsonld"):
        self.schema_file = schema_path
        self.ontology: Dict[str, Any] = {}
        self.loaded = False
        
        # Cache for quick lookups
        self.valid_types: Set[str] = set()
        self.property_domains: Dict[str, List[str]] = {} # prop -> list of valid types
        self.property_ranges: Dict[str, List[str]] = {} # prop -> list of expected types
        self.parent_types: Dict[str, List[str]] = {} # type -> list of direct parent types

    def load_schema_org(self):
        """
        Loads Schema.org ontology from local file.
        """
        if self.loaded:
            return

        if not os.path.exists(self.schema_file):
             # Try absolute path or relative to project root if default fails
             base_dir = os.path.dirname(os.path.dirname(__file__)) # kg_transform/
             potential_path = os.path.join(base_dir, self.schema_file)
             if os.path.exists(potential_path):
                 self.schema_file = potential_path
             else:
                 raise FileNotFoundError(f"Schema.org ontology file not found at {self.schema_file}")

        with open(self.schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._process_ontology(data)
            self.loaded = True

    def _process_ontology(self, data: Dict[str, Any]):
        """
        Processes the raw JSON-LD ontology into internal structures for validation.
        Uses jsonld expansion to ensure canonical URIs.
        """
        from pyld import jsonld
        
        # Expand uses the context inside the data to normalize to full URIs
        try:
            expanded_list = jsonld.expand(data)
        except Exception as e:
            # Fallback if expansion fails (e.g. context issues), though it shouldn't for valid file
            print(f"Warning: Ontology expansion failed: {e}. Using raw graph.")
            expanded_list = data.get("@graph", [])

        self.ontology = data # Keep raw data referenced if needed
        
        for item in expanded_list:
            item_id = item.get("@id")
            item_types = item.get("@type", [])
            
            if not item_id:
                continue
                
            # Normalize type to list
            if isinstance(item_types, str):
                item_types = [item_types]
            
            # Store valid types (Classes)
            # Check for rdfs:Class full URI
            # rdfs is http://www.w3.org/2000/01/rdf-schema#
            if "http://www.w3.org/2000/01/rdf-schema#Class" in item_types:
                # print(f"DEBUG: Found class {item_id}")
                self.valid_types.add(item_id)
                
                # Handle subClassOf
                parents = item.get("http://www.w3.org/2000/01/rdf-schema#subClassOf", [])
                if isinstance(parents, dict): parents = [parents]
                
                parent_ids = [p.get("@id") for p in parents if isinstance(p, dict) and "@id" in p]
                if parent_ids:
                    self.parent_types[item_id] = parent_ids
            
            # Store Properties
            # rdf:Property is http://www.w3.org/1999/02/22-rdf-syntax-ns#Property
            if "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property" in item_types:
                # domainIncludes
                # schema is defined as https://schema.org/ in the file
                # so properties are https://schema.org/domainIncludes
                # Check both http and https to be safe, or rely on file context
                
                # In expanded JSON-LD, properties are values of keys
                # The key in the item dict IS the property URI
                
                domains = item.get("https://schema.org/domainIncludes", [])
                if not domains:
                    domains = item.get("http://schema.org/domainIncludes", [])

                # In expanded form, value is list of dicts with @id
                domain_ids = [d.get("@id") for d in domains if isinstance(d, dict) and "@id" in d]
                self.property_domains[item_id] = domain_ids

                # rangeIncludes
                ranges = item.get("https://schema.org/rangeIncludes", [])
                if not ranges:
                     ranges = item.get("http://schema.org/rangeIncludes", [])

                range_ids = [r.get("@id") for r in ranges if isinstance(r, dict) and "@id" in r]
                self.property_ranges[item_id] = range_ids

    def _get_all_types(self, node_types: List[str]) -> Set[str]:
        """
        Returns a set of all types for the node, including parent types.
        """
        all_types = set(node_types)
        queue = list(node_types)
        visited = set(node_types)
        
        while queue:
            current_type = queue.pop(0)
            if current_type in self.parent_types:
                for parent in self.parent_types[current_type]:
                    if parent not in visited:
                        visited.add(parent)
                        all_types.add(parent)
                        queue.append(parent)
        return all_types

    def validate_kg(self, kg: InternalKG) -> ValidationResult:
        """
        Validates the InternalKG against the loaded ontology.
        """
        if not self.loaded:
            self.load_schema_org()
            
        errors: List[ValidationError] = []
        
        for node in kg.nodes:
            # 1. Validate Types
            for node_type in node.types:
                # Schema.org uses http://schema.org/Type usually, assuming expansion
                # Check directly and maybe normalized
                if node_type not in self.valid_types:
                    # Try to see if it matches if we ensure schema.org prefix
                    # Ideally the ontology uses full URIs. The downloaded jsonld usually creates ids like "http://schema.org/Person"
                    errors.append(ValidationError(
                        message=f"Unknown type: {node_type}",
                        node_id=node.id
                    ))

            # 2. Validate Properties (Domain)
            # Check if properties are allowed for this node's types
            for prop_uri, _ in node.properties.items():
                if prop_uri.startswith("@"): continue # skip json-ld keywords
                
                # If we don't know the property, maybe warn or error? 
                # Strict mode: error. Lenient: skip.
                # Let's assume strict check if property is in our ontology
                if prop_uri in self.property_domains:
                     allowed_domains = self.property_domains[prop_uri]
                     # If allowed_domains is empty, maybe it applies to everything or it's not defined well?
                     # Ideally we check intersection of node.types and allowed_domains
                     # But we need to handle inheritance (subClassOf).
                     # For MVP: simple check if ANY type matches. Inheritence requires more complex graph traversal of ontology.
                     
                     # Simple check:
                     has_domain_match = False
                     if not allowed_domains:
                         has_domain_match = True # No restrictions
                     else:
                        # Check against full hierarchy
                        all_node_types = self._get_all_types(node.types)
                        for ntype in all_node_types:
                            if ntype in allowed_domains:
                                has_domain_match = True
                                break
                     
                     if not has_domain_match:
                         # For error message, show direct types for clarity
                         errors.append(ValidationError(
                             message=f"Property {prop_uri} not allowed for types {node.types}",
                             node_id=node.id,
                             property_name=prop_uri
                         ))
                
            # 3. Edge Validation (Range) - implicitly properties too
            pass # Similar complexity for range.

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
