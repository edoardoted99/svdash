import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import os
from pyld import jsonld
from kg_parser.parser import parse_jsonld, serialize_to_jsonld, InternalKG
from kg_parser.validator import SchemaOrgValidator
from kg_parser.diff import compute_diff, AtomicOp
from kg_parser.loader import get_local_document_loader

from .provider import OllamaProvider
from .prompts import build_messages

logger = logging.getLogger(__name__)

# Loader function removed, imported from kg_parser.loader

@dataclass
class TransformResult:
    """
    Result of a KG transformation process.
    """
    kg_output: Optional[InternalKG]
    diff: List[AtomicOp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

async def transform_kg(prompt: str, kg_input_data: Dict[str, Any], model: Optional[str] = None) -> TransformResult:
    """
    Orchestrates the transformation of a Knowledge Graph using an LLM.
    
    1. Parse and validate input KG
    2. Build messages
    3. Call OllamaProvider
    4. Extract and Validate Output
    5. Compute Diff
    6. Save to DB (Placeholder)
    7. Return Result
    """
    provider = OllamaProvider()
    validator = SchemaOrgValidator()
    
    # 1. Parse Input
    try:
        loader = get_local_document_loader()
        options = {"documentLoader": loader}
        kg_input = parse_jsonld(kg_input_data, options=options)
        # Optional: Validate input? The prompt implies we trust input or just parse it.
        # But let's assume valid JSON-LD structure at least.
    except Exception as e:
        logger.error(f"Failed to parse input KG: {e}")
        return TransformResult(kg_output=None, error=f"Input parsing error: {str(e)}")

    # 2. Build Messages
    messages = build_messages(prompt, kg_input_data)
    
    # Retry Loop
    max_retries = 3
    retry_count = 0
    kg_output = None
    
    while retry_count < max_retries:
        # 3. Call Provider
        try:
            llm_response = await provider.call(messages, model=model)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return TransformResult(kg_output=None, error=f"LLM provider error: {str(e)}")

        content = llm_response.content
        
        # 4. Extract JSON
        json_content = _extract_json(content)
        if not json_content:
            error_msg = "Could not find valid JSON object in response."
            _add_error_feedback(messages, content, error_msg)
            retry_count += 1
            continue
            
        try:
            output_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON syntax: {e}"
            _add_error_feedback(messages, content, error_msg)
            retry_count += 1
            continue

        # 5. Parse and Validate Output
        try:
            # We assume the LLM follows instruction to strictly output JSON-LD
            # We might want to pass options if needed, but default is fine
            candidate_kg = parse_jsonld(output_data, options=options)
            
            validation_res = validator.validate_kg(candidate_kg)
            if validation_res.is_valid:
                kg_output = candidate_kg
                break # Success!
            else:
                # Construct error feedback from validation errors
                error_lines = [f"- Node {e.node_id}: {e.message}" for e in validation_res.errors[:5]] # Limit to 5
                error_msg = "Validation failed:\n" + "\n".join(error_lines)
                _add_error_feedback(messages, content, error_msg)
                retry_count += 1
                
        except Exception as e:
             error_msg = f"Error parsing output JSON-LD: {str(e)}"
             _add_error_feedback(messages, content, error_msg)
             retry_count += 1

    if not kg_output:
        return TransformResult(kg_output=None, error="Max retries exceeded. Could not generate valid KG.")

    # 6. Compute Diff
    diff_ops = compute_diff(kg_input, kg_output)
    
    # 7. Save to DB
    transformation_id = None
    try:
        from core.models import KnowledgeGraph, Transformation, AtomicOperation
        from asgiref.sync import sync_to_async

        # We need to run DB operations in sync context if we are in async function, using sync_to_async
        # Or better, wrap the whole save block
        
        @sync_to_async
        def save_results():
            # Save Input KG (if not already existing? For now, always new for simplicity or we could check hash)
            # Assuming we save a new snapshot every time for full history
            input_kg_obj = KnowledgeGraph.objects.create(data=serialize_to_jsonld(kg_input))
            
            # Save Output KG
            output_kg_obj = KnowledgeGraph.objects.create(data=serialize_to_jsonld(kg_output))
            
            # Save Transformation
            transformation = Transformation.objects.create(
                prompt=prompt,
                input_kg=input_kg_obj,
                output_kg=output_kg_obj,
                model_used=llm_response.model or (model or "unknown")
            )
            
            # Save Operations
            ops_to_create = []
            for op in diff_ops:
                ops_to_create.append(AtomicOperation(
                    transformation=transformation,
                    op_type=op.op_type.value,
                    target_id=op.target_id,
                    payload=op.payload
                ))
            AtomicOperation.objects.bulk_create(ops_to_create)
            return transformation.id

        transformation_id = await save_results()

    except ImportError:
        logger.warning("Core models not found, skipping DB save.")
    except Exception as e:
        logger.error(f"Failed to save transformation to DB: {e}")
        # We don't fail the whole request, just log error

    # 8. Return Result
    result = TransformResult(
        kg_output=kg_output,
        diff=diff_ops,
        metadata={
            "model": llm_response.model,
            "prompt_tokens": llm_response.prompt_tokens,
            "completion_tokens": llm_response.completion_tokens,
            "retries": retry_count,
            "transformation_id": str(transformation_id) if transformation_id else None
        }
    )
    return result

def _extract_json(text: str) -> Optional[str]:
    """
    Extracts the first JSON object string from the text.
    Handles markdown code blocks.
    """
    text = text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```"):
        # Find first newline
        try:
            start_idx = text.index("\n") + 1
            # Find last ```
            end_idx = text.rindex("```")
            text = text[start_idx:end_idx]
        except ValueError:
            pass # Fallback to raw text search

    # Find first { and last }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return text[start:end]
    except ValueError:
        return None

def _add_error_feedback(messages: List[Dict[str, str]], assistant_content: str, error_msg: str):
    """
    Appends the assistant's invalid response and the system's error feedback to messages.
    """
    messages.append({"role": "assistant", "content": assistant_content})
    messages.append({
        "role": "user", 
        "content": f"The output was invalid. Error: {error_msg}\nPlease correct it and return ONLY valid JSON-LD."
    })
