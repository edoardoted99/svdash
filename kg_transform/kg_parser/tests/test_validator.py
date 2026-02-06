import pytest
from unittest.mock import MagicMock, patch
from kg_parser.parser import InternalKG, KGNode
from kg_parser.validator import SchemaOrgValidator, ValidationResult

class MockValidator(SchemaOrgValidator):
    def __init__(self, schema_path="dummy_path"):
        super().__init__(schema_path)
        
    def load_schema_org(self):
        # Mock loading by manually setting up valid types/properties
        self.loaded = True
        self.valid_types = {
            "http://schema.org/Person", 
            "http://schema.org/Organization"
        }
        self.property_domains = {
            "http://schema.org/worksFor": ["http://schema.org/Person"],
            "http://schema.org/name": [] # Allowed for everyone in this mock
        }
        self.property_ranges = {}

@pytest.fixture
def validator():
    return MockValidator()

def test_validate_valid_kg(validator):
    kg = InternalKG(
        nodes=[
            KGNode(id="http://example.org/alice", types=["http://schema.org/Person"], properties={"http://schema.org/name": "Alice"}),
            KGNode(id="http://example.org/acme", types=["http://schema.org/Organization"], properties={"http://schema.org/name": "Acme"})
        ],
        edges=[]
    )
    result = validator.validate_kg(kg)
    assert result.is_valid
    assert len(result.errors) == 0

def test_validate_invalid_type(validator):
    kg = InternalKG(
        nodes=[
            KGNode(id="http://example.org/bob", types=["http://schema.org/Alien"], properties={})
        ],
        edges=[]
    )
    result = validator.validate_kg(kg)
    assert not result.is_valid
    assert len(result.errors) == 1
    assert "Unknown type" in result.errors[0].message

def test_validate_invalid_domain(validator):
    # worksFor is only for Person
    kg = InternalKG(
        nodes=[
            KGNode(id="http://example.org/acme", types=["http://schema.org/Organization"], properties={"http://schema.org/worksFor": "something"})
        ],
        edges=[]
    )
    result = validator.validate_kg(kg)
    assert not result.is_valid
    assert len(result.errors) == 1
    assert "Property http://schema.org/worksFor not allowed" in result.errors[0].message
