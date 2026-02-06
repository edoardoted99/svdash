import os
import json
import logging

logger = logging.getLogger(__name__)

def get_local_document_loader():
    """
    Returns a pyld document loader that uses the local schema.org file.
    """
    # Path relative to this file: ./schemaorg-current-https.jsonld
    base_dir = os.path.dirname(__file__)
    schema_path = os.path.join(base_dir, "schemaorg-current-https.jsonld")
    
    def loader(url, options={}):
        if "schema.org" in url:
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    data = json.load(f)
                
                # Inject @vocab to support simple term expansion if missing
                if "@context" in data:
                    if "@vocab" not in data["@context"]:
                        data["@context"]["@vocab"] = "https://schema.org/"
                
                return {
                    'contextUrl': None,
                    'documentUrl': url,
                    'document': data
                }
            else:
                 logger.warning(f"Local schema file not found at {schema_path}, falling back to default loader (network)")
        
        # Fallback to default loader (or error if no default installed/configured)
        # PyLD's default behavior is to use requests if installed.
        # But we want to avoid network if possible or if requests is missing.
        from pyld.jsonld import requests_document_loader
        return requests_document_loader()(url, options)

    return loader
