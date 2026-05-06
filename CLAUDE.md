# SalesTee — Project Index

## Purpose
A deterministic, zero-cost, offline sales retrieval agent with voice. Built on the DataG neuronal architecture.

## Tech Stack
- **Architecture**: Retrieval over neuronal substrate (128-dim DNA vectors).
- **Core Logic**: Python (datag_bridge.py, ingest_doc.py).
- **Storage**: JSON files in `data_store/methodology/`.
- **Models**: Custom transformer encoder/decoder in `models/`.

## Key Files
- `datag_bridge.py`: Singleton bridge to the substrate. Handles encoding/decoding.
- `ingest_doc.py`: CLI tool for feeding docs into memory.
- `seed_product_voxi.py`: Example product cell seeding.
- `src/living_cell.py`: Core data structure for a "cell".

## Guidelines for Claude
- **Deterministic**: Always use `bridge.save_cell` for ingestion.
- **Concise**: Focus on the (description, content) pair structure.
- **Ignore Data**: Do not attempt to read files in `data_store/` or `models/` unless explicitly asked; they are too large.
