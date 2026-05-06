import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tempfile
import shutil

# Add parent dir to path to import ingest_doc and src
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import ingest_doc
from datag_bridge import DataGBridge

app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChunkRequest(BaseModel):
    text: str

class Cell(BaseModel):
    content: str
    description: str

class SaveRequest(BaseModel):
    source_table: str
    cells: List[Cell]

class QACell(BaseModel):
    question: str
    answer: str
    source_table: str

class SuggestRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/namespaces")
def get_namespaces():
    """
    Return the valid source_table options the UI should present.
    Fixed namespaces plus any product namespaces already in the corpus.
    """
    bridge = DataGBridge.get()
    fixed = [
        {"value": "identity",     "label": "Sales identity / persona",
         "hint": "How Tee speaks — tone, values, confidence"},
        {"value": "reasoning",    "label": "Objection handling",
         "hint": "Counter-arguments, pushback responses, persuasion logic"},
        {"value": "conversation", "label": "General knowledge",
         "hint": "Industry context, company background, supporting detail"},
    ]
    products = []
    if bridge.ready:
        seen = set()
        for c in bridge.substrate.methodology_cells.values():
            t = getattr(c, 'source_table', '')
            if t.startswith('meth_product_'):
                slug = t[len('meth_product_'):]
                if slug not in seen:
                    seen.add(slug)
                    products.append({
                        "value": f"product_{slug}",
                        "label": f"Product: {slug.replace('_', ' ').title()}",
                        "hint":  "Product/service content — returned word-for-word",
                    })
    return {"fixed": fixed, "products": sorted(products, key=lambda x: x["value"])}


@app.get("/products")
def get_products():
    meth_dir = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')
    if not os.path.exists(meth_dir):
        return []
    return ingest_doc.list_known_products(meth_dir)

@app.post("/extract")
async def extract_content(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = ingest_doc.load_file(tmp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/chunk")
def chunk_content(req: ChunkRequest):
    try:
        chunks = ingest_doc.chunk_text(req.text)
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

SIM_DUPLICATE = 0.85   # blocked — too close to an existing cell
SIM_NEAR_DUPE = 0.70   # warning — similar but allowed through


def _check_duplicate(bridge, description: str, source_table: str):
    """
    Returns (status, sim, existing_description) where status is:
      'duplicate'  — sim >= SIM_DUPLICATE, block the save
      'near_dupe'  — sim >= SIM_NEAR_DUPE, warn but allow
      'ok'         — safe to save

    Only checks cells in the same source_table namespace — cross-product
    similarity is expected and should never block a save.
    """
    hits = bridge.top_cells(description, k=20)
    full_table = f"meth_{source_table}" if not source_table.startswith('meth_') else source_table
    same_ns = [(s, cid, c) for s, cid, c in hits
               if getattr(c, 'source_table', '') == full_table]
    if not same_ns:
        return 'ok', 0.0, ''
    sim, _, cell = same_ns[0]
    existing_desc = (getattr(cell, 'meta', {}) or {}).get('description', '')
    if sim >= SIM_DUPLICATE:
        return 'duplicate', round(float(sim), 3), existing_desc
    if sim >= SIM_NEAR_DUPE:
        return 'near_dupe', round(float(sim), 3), existing_desc
    return 'ok', round(float(sim), 3), existing_desc


@app.post("/save")
def save_memory(req: SaveRequest):
    try:
        bridge = DataGBridge.get()
        if not bridge.ready:
            raise HTTPException(status_code=500, detail="Tee's substrate failed to load")

        effective_table = req.source_table

        saved = []
        duplicates = []
        warnings = []

        for cell in req.cells:
            status, sim, existing = _check_duplicate(bridge, cell.description, effective_table)

            if status == 'duplicate':
                duplicates.append({
                    "description": cell.description,
                    "sim": sim,
                    "existing": existing,
                })
                continue

            cid = bridge.save_cell(
                description=cell.description,
                content=cell.content,
                source_table=effective_table,
                confidence=0.95,
                energy=150.0,
            )
            entry = {"cell_id": cid, "description": cell.description}

            if status == 'near_dupe':
                entry["warning"] = f"Similar to existing cell (sim={sim}): \"{existing}\""
                warnings.append(entry)
            else:
                saved.append(entry)

        return {
            "status": "success",
            "saved": len(saved),
            "saved_cells": saved,
            "duplicates": duplicates,
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa")
def save_qa(req: QACell):
    """Save a single Q&A pair directly as a cell. Same path as save_cell()."""
    try:
        bridge = DataGBridge.get()
        if not bridge.ready:
            raise HTTPException(status_code=500, detail="Tee's substrate failed to load")

        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question is required")
        if not req.answer.strip():
            raise HTTPException(status_code=400, detail="answer is required")

        valid_tables = {'identity', 'reasoning', 'conversation'}
        table = req.source_table.strip()
        if not table:
            raise HTTPException(status_code=400, detail="source_table is required")
        if table not in valid_tables and not table.startswith('product_'):
            raise HTTPException(status_code=400,
                detail=f"source_table must be one of {sorted(valid_tables)} or 'product_<name>'")

        status, sim, existing = _check_duplicate(bridge, req.question.strip(), table)

        if status == 'duplicate':
            raise HTTPException(status_code=409, detail={
                "reason": "duplicate",
                "sim": sim,
                "existing": existing,
                "message": f"Too similar to existing cell (sim={sim}): \"{existing}\"",
            })

        cid = bridge.save_cell(
            description=req.question.strip(),
            content=req.answer.strip(),
            source_table=table,
            confidence=0.95,
            energy=150.0,
        )

        result = {"status": "saved", "cell_id": cid}
        if status == 'near_dupe':
            result["warning"] = f"Similar to existing cell (sim={sim}): \"{existing}\""
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggest")
def suggest_similar(req: SuggestRequest):
    """
    Return top-K existing cells similar to a question.
    Used by the UI before saving to show what's already covered.
    """
    try:
        bridge = DataGBridge.get()
        if not bridge.ready:
            raise HTTPException(status_code=500, detail="Tee's substrate failed to load")

        if not req.question.strip():
            return {"hits": []}

        hits = bridge.top_cells(req.question.strip(), k=req.top_k)
        return {
            "hits": [
                {
                    "sim": round(float(s), 3),
                    "cell_id": cid,
                    "description": (getattr(c, 'meta', {}) or {}).get('description', ''),
                    "content": (getattr(c, 'content', '') or '')[:200],
                    "source_table": getattr(c, 'source_table', ''),
                }
                for s, cid, c in hits
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Return corpus size and namespace breakdown."""
    try:
        bridge = DataGBridge.get()
        if not bridge.ready:
            raise HTTPException(status_code=500, detail="Tee's substrate failed to load")

        cells = bridge.substrate.methodology_cells
        by_table: dict = {}
        for c in cells.values():
            t = getattr(c, 'source_table', 'unknown')
            by_table[t] = by_table.get(t, 0) + 1

        return {"total": len(cells), "by_table": by_table}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
