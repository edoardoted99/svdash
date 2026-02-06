import pytest
from kg_parser.parser import parse_jsonld, serialize_to_jsonld, InternalKG, KGNode, KGEdge

# Mock Document Loader for PyLD
def mock_document_loader(url, options={}):
    if url == "https://schema.org":
        return {
            'contextUrl': None,
            'documentUrl': url,
            'document': {
                "@context": {
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                    "name": "http://schema.org/name",
                    "worksFor": {"@id": "http://schema.org/worksFor", "@type": "@id"},
                    "Person": "http://schema.org/Person",
                    "Organization": "http://schema.org/Organization"
                }
            }
        }
    raise Exception(f"URL {url} not mocked")

@pytest.fixture
def options():
    return {'documentLoader': mock_document_loader}

# Sample Data
@pytest.fixture
def sample_jsonld():
    return {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@id": "http://example.org/alice",
                "@type": "Person",
                "name": "Alice",
                "worksFor": {
                    "@id": "http://example.org/acme"
                }
            },
            {
                "@id": "http://example.org/acme",
                "@type": "Organization",
                "name": "Acme Corp"
            }
        ]
    }

def test_parse_jsonld_nodes(sample_jsonld, options):
    kg = parse_jsonld(sample_jsonld, options=options)
    
    # Check nodes
    assert len(kg.nodes) == 2
    
    alice = kg.get_node("http://example.org/alice")
    assert alice is not None
    assert "http://schema.org/Person" in alice.types # pyld expands to full URIs
    # With the simple context above, "name" maps to http://schema.org/name
    assert alice.properties["http://schema.org/name"] == "Alice"

    acme = kg.get_node("http://example.org/acme")
    assert acme is not None
    assert "http://schema.org/Organization" in acme.types
    assert acme.properties["http://schema.org/name"] == "Acme Corp"

def test_parse_jsonld_edges(sample_jsonld, options):
    kg = parse_jsonld(sample_jsonld, options=options)
    
    # Check edges
    assert len(kg.edges) == 1
    edge = kg.edges[0]
    assert edge.source_id == "http://example.org/alice"
    assert edge.target_id == "http://example.org/acme"
    assert edge.predicate == "http://schema.org/worksFor"

def test_round_trip(sample_jsonld, options):
    kg = parse_jsonld(sample_jsonld, options=options)
    reconstructed = serialize_to_jsonld(kg, options=options)
    
    kg2 = parse_jsonld(reconstructed, options=options)
    
    assert len(kg2.nodes) == len(kg.nodes)
    assert len(kg2.edges) == len(kg.edges)
    
    alice = kg2.get_node("http://example.org/alice")
    assert alice is not None
    assert alice.properties["http://schema.org/name"] == "Alice"
