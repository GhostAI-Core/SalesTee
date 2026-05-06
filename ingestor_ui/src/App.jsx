import React, { useState, useEffect } from 'react';
import {
  Upload,
  FileText,
  Type,
  CheckCircle2,
  ChevronRight,
  Trash2,
  Edit3,
  Save,
  ArrowLeft,
  Search,
  Database,
  BrainCircuit,
  MessageSquare,
  Sparkles,
  Layers,
  ClipboardList,
  AlertCircle,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const CONTENT_TYPES = [
  { 
    id: 'product', 
    label: 'Product / Service', 
    icon: <Database className="w-6 h-6" />, 
    desc: 'Pricing, features, exact steps. Tee will not paraphrase.' 
  },
  { 
    id: 'identity', 
    label: 'Identity / Persona', 
    icon: <BrainCircuit className="w-6 h-6" />, 
    desc: 'Shapes HOW Tee speaks – tone, values, persona.' 
  },
  { 
    id: 'reasoning', 
    label: 'Objection Handling', 
    icon: <Search className="w-6 h-6" />, 
    desc: 'Counter-arguments and persuasion logic.' 
  },
  { 
    id: 'conversation', 
    label: 'General Knowledge', 
    icon: <MessageSquare className="w-6 h-6" />, 
    desc: 'Context, industry facts, company background.' 
  },
  { 
    id: 'custom', 
    label: 'Custom Category', 
    icon: <Sparkles className="w-6 h-6" />, 
    desc: 'You define the category name.' 
  },
];

// ── Bulk import parser ────────────────────────────────────────────────────────
// Accepts the NotebookLM CELLS = [...] output and returns parsed cells or an error.
// Valid tuple shapes:
//   ("question", "answer")
//   ("question", "answer", "product_slug")
function parseCellsPython(raw) {
  const errors = [];
  const parsed = [];

  // Strip everything before the first '[' (the CELLS = part)
  const bracketStart = raw.indexOf('[');
  const bracketEnd   = raw.lastIndexOf(']');
  if (bracketStart === -1 || bracketEnd === -1) {
    return { ok: false, errors: ['No list found. Paste the full CELLS = [...] block.'], cells: [] };
  }
  const inner = raw.slice(bracketStart + 1, bracketEnd);

  // Split on tuple boundaries — match (...) groups
  const tupleRe = /\(\s*("""[\s\S]*?"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*,\s*("""[\s\S]*?"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')(?:\s*,\s*("""[\s\S]*?"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))?\s*\)/g;

  let match;
  let count = 0;
  while ((match = tupleRe.exec(inner)) !== null) {
    count++;
    const unquote = (s) => s.replace(/^"""|"""$|^"|"$|^'|'$/g, '').replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\'/g, "'");
    const question = unquote(match[1]).trim();
    const answer   = unquote(match[2]).trim();
    const slug     = match[3] ? unquote(match[3]).trim() : null;

    if (!question) { errors.push(`Tuple ${count}: question is empty`); continue; }
    if (!answer)   { errors.push(`Tuple ${count}: answer is empty`); continue; }
    if (question.length < 5)  { errors.push(`Tuple ${count}: question too short — "${question}"`); continue; }
    if (answer.length < 5)    { errors.push(`Tuple ${count}: answer too short — "${answer}"`); continue; }

    parsed.push({ question, answer, slug, id: Math.random() });
  }

  if (count === 0) {
    return { ok: false, errors: ['No tuples found. Check the format matches the example.'], cells: [] };
  }

  return { ok: errors.length === 0, errors, cells: parsed };
}

export default function App() {
  const [step, setStep] = useState(1);
  const [inputMode, setInputMode] = useState('document'); // 'document' | 'bulk'
  const [bulkRaw, setBulkRaw] = useState('');
  const [bulkResult, setBulkResult] = useState(null); // { ok, errors, cells }
  const [loading, setLoading] = useState(false);
  const [rawText, setRawText] = useState('');
  const [contentType, setContentType] = useState(null);
  const [productName, setProductName] = useState('');
  const [customCategory, setCustomCategory] = useState('');
  const [chunks, setChunks] = useState([]);
  const [knownProducts, setKnownProducts] = useState([]);
  const [saveResult, setSaveResult] = useState(null); // { saved, duplicates, warnings }

  useEffect(() => {
    if (step === 3 && contentType?.id === 'product') {
      fetchKnownProducts();
    }
  }, [step]);

  const fetchKnownProducts = async () => {
    try {
      const res = await axios.get(`${API_BASE}/products`);
      setKnownProducts(res.data);
    } catch (e) {
      console.error("Failed to fetch products", e);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/extract`, formData);
      setRawText(res.data.text);
      setStep(2);
    } catch (e) {
      alert("Failed to extract file content");
    } finally {
      setLoading(false);
    }
  };

  const processChunks = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/chunk`, { text: rawText });
      setChunks(res.data.chunks.map(content => ({ content, description: '', id: Math.random() })));
      setStep(4);
    } catch (e) {
      alert("Chunking failed");
    } finally {
      setLoading(false);
    }
  };

  const saveToMemory = async () => {
    setLoading(true);
    try {
      const valid = chunks.filter(c => c.content && c.description);

      // Bulk import path: cells may carry per-cell slugs — group and save per namespace
      const hasMixedSlugs = valid.some(c => c._slug) &&
        new Set(valid.map(c => c._slug || '__default__')).size > 1;

      const aggregated = { saved: 0, saved_cells: [], duplicates: [], warnings: [] };

      if (hasMixedSlugs) {
        const groups = {};
        for (const c of valid) {
          const t = c._slug ? `product_${c._slug}` : (contentType?.id === 'product' ? `product_${productName}` : contentType?.id || 'conversation');
          if (!groups[t]) groups[t] = [];
          groups[t].push({ content: c.content, description: c.description });
        }
        for (const [source_table, cells] of Object.entries(groups)) {
          const res = await axios.post(`${API_BASE}/save`, { source_table, cells });
          aggregated.saved += res.data.saved;
          aggregated.saved_cells.push(...(res.data.saved_cells || []));
          aggregated.duplicates.push(...(res.data.duplicates || []));
          aggregated.warnings.push(...(res.data.warnings || []));
        }
      } else {
        const sourceTable = contentType?.id === 'product' ? `product_${productName}` :
                           contentType?.id === 'custom' ? customCategory :
                           contentType?.id || 'conversation';
        const res = await axios.post(`${API_BASE}/save`, {
          source_table: sourceTable,
          cells: valid.map(c => ({ content: c.content, description: c.description }))
        });
        aggregated.saved = res.data.saved;
        aggregated.saved_cells = res.data.saved_cells || [];
        aggregated.duplicates = res.data.duplicates || [];
        aggregated.warnings = res.data.warnings || [];
      }

      setSaveResult(aggregated);
      setStep(5);
    } catch (e) {
      alert("Save failed: " + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  const handleBulkParse = () => {
    const result = parseCellsPython(bulkRaw);
    setBulkResult(result);
  };

  const handleBulkProceed = () => {
    // Convert parsed cells to chunks and jump to type selection.
    // The slug from the tuple (if present) will pre-fill productName.
    const slugs = [...new Set(bulkResult.cells.map(c => c.slug).filter(Boolean))];
    if (slugs.length === 1) {
      setProductName(slugs[0]);
      setContentType(CONTENT_TYPES.find(t => t.id === 'product'));
    }
    setChunks(
      bulkResult.cells.map(c => ({
        id: c.id,
        description: c.question,
        content: c.answer,
        _slug: c.slug,
      }))
    );
    setStep(slugs.length === 1 ? 4 : 2); // skip to review if product auto-detected
  };

  const updateChunk = (id, field, value) => {
    setChunks(chunks.map(c => c.id === id ? { ...c, [field]: value } : c));
  };

  const removeChunk = (id) => {
    setChunks(chunks.filter(c => c.id !== id));
  };

  const renderStep = () => {
    switch(step) {
      case 1:
        return (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="animate-fade-in">
            <h2 className="text-3xl font-bold mb-6">Source Material</h2>

            {/* Mode tabs */}
            <div className="flex gap-2 mb-8 p-1 glass rounded-xl w-fit">
              <button
                className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-all ${inputMode === 'document' ? 'bg-primary-color text-white' : 'text-dim hover:text-white'}`}
                onClick={() => setInputMode('document')}
              >
                <Upload className="w-4 h-4" /> Document
              </button>
              <button
                className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-all ${inputMode === 'bulk' ? 'bg-primary-color text-white' : 'text-dim hover:text-white'}`}
                onClick={() => { setInputMode('bulk'); setBulkResult(null); }}
              >
                <ClipboardList className="w-4 h-4" /> Bulk Import (CELLS)
              </button>
            </div>

            {inputMode === 'document' && (
              <>
                <div className="upload-zone mb-8" onClick={() => document.getElementById('fileInput').click()}>
                  <Upload className="upload-icon mx-auto" />
                  <p className="text-xl font-medium">Drag & Drop or Click to Upload</p>
                  <p className="text-dim mt-2">Supports PDF, DOCX, TXT, MD</p>
                  <input type="file" id="fileInput" className="hidden" onChange={handleFileUpload} />
                </div>
                <div className="relative">
                  <div className="absolute inset-0 flex items-center"><span className="w-full border-t border-glass-border"></span></div>
                  <div className="relative flex justify-center text-xs uppercase"><span className="bg-bg-color px-2 text-dim">Or paste text</span></div>
                </div>
                <textarea
                  className="textarea-input"
                  placeholder="Paste text here..."
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                />
                {rawText && (
                  <div className="mt-8 flex justify-end">
                    <button className="btn btn-primary" onClick={() => setStep(2)}>
                      Next: Select Type <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </>
            )}

            {inputMode === 'bulk' && (
              <>
                <div className="glass p-5 rounded-xl mb-5 text-sm text-dim border border-glass-border">
                  <p className="font-semibold text-white mb-1">Expected format (NotebookLM output):</p>
                  <pre className="text-xs overflow-x-auto text-primary-color">
{`CELLS = [
    ("what is the setup process", "The setup is five steps...", "voxi"),
    ("can it handle multiple calls", "Yes. The system scales...", "voxi"),
]`}
                  </pre>
                  <p className="mt-2">The third element (product slug) is optional but recommended. All fields are validated before you proceed.</p>
                </div>

                <textarea
                  className="textarea-input font-mono text-sm"
                  placeholder="Paste CELLS = [...] output here..."
                  value={bulkRaw}
                  onChange={(e) => { setBulkRaw(e.target.value); setBulkResult(null); }}
                  style={{ minHeight: '220px' }}
                />

                {bulkRaw && !bulkResult && (
                  <div className="mt-4 flex justify-end">
                    <button className="btn btn-primary" onClick={handleBulkParse}>
                      Validate Format <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                )}

                {bulkResult && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6 space-y-4">
                    {/* Error list */}
                    {bulkResult.errors.length > 0 && (
                      <div className="glass p-4 rounded-xl border border-error/40 space-y-2">
                        <div className="flex items-center gap-2 font-semibold text-error">
                          <AlertCircle className="w-4 h-4" /> {bulkResult.errors.length} validation issue{bulkResult.errors.length > 1 ? 's' : ''}
                        </div>
                        <ul className="text-sm text-dim list-disc list-inside space-y-1">
                          {bulkResult.errors.map((e, i) => <li key={i}>{e}</li>)}
                        </ul>
                      </div>
                    )}

                    {/* Success summary */}
                    {bulkResult.cells.length > 0 && (
                      <div className="glass p-4 rounded-xl border border-success/40">
                        <div className="flex items-center gap-2 font-semibold text-success mb-3">
                          <CheckCircle2 className="w-4 h-4" /> {bulkResult.cells.length} cells ready to review
                        </div>
                        <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar pr-2">
                          {bulkResult.cells.map((c, i) => (
                            <div key={c.id} className="text-xs text-dim flex gap-2">
                              <span className="text-primary-color font-mono w-5 shrink-0">{i + 1}.</span>
                              <span className="truncate">{c.question}</span>
                              {c.slug && <span className="shrink-0 px-1.5 py-0.5 rounded bg-white/10 text-white/60">{c.slug}</span>}
                            </div>
                          ))}
                        </div>
                        <div className="mt-4 flex justify-end">
                          <button className="btn btn-primary" onClick={handleBulkProceed}>
                            Review & Save <ChevronRight className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </>
            )}
          </motion.div>
        );
      case 2:
        return (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <button className="btn btn-outline mb-6" onClick={() => setStep(1)}><ArrowLeft className="w-4 h-4" /> Back</button>
            <h2 className="text-3xl font-bold mb-6">Content Type</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {CONTENT_TYPES.map(type => (
                <div 
                  key={type.id} 
                  className={`glass type-card ${contentType?.id === type.id ? 'selected' : ''}`}
                  onClick={() => setContentType(type)}
                >
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-lg bg-white/5">{type.icon}</div>
                    <div>
                      <h4 className="font-bold">{type.label}</h4>
                      <p className="text-dim text-sm">{type.desc}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            {contentType && (
              <div className="mt-8 flex justify-end">
                <button className="btn btn-primary" onClick={() => setStep(3)}>
                  Next: Details <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </motion.div>
        );
      case 3:
        return (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <button className="btn btn-outline mb-6" onClick={() => setStep(2)}><ArrowLeft className="w-4 h-4" /> Back</button>
            <h2 className="text-3xl font-bold mb-6">Details</h2>
            
            {contentType.id === 'product' && (
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-dim mb-2">Product Name</label>
                  <input 
                    type="text" 
                    className="textarea-input !min-h-0" 
                    placeholder="e.g. VOXI Pro"
                    value={productName}
                    onChange={(e) => setProductName(e.target.value)}
                  />
                </div>
                {knownProducts.length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-dim mb-2">Existing Products</label>
                    <div className="flex flex-wrap gap-2">
                      {knownProducts.map(p => (
                        <button 
                          key={p} 
                          className="px-3 py-1 rounded-full bg-white/5 border border-glass-border hover:border-primary-color text-sm"
                          onClick={() => setProductName(p)}
                        >
                          {p}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {contentType.id === 'custom' && (
              <div>
                <label className="block text-sm font-medium text-dim mb-2">Category Name</label>
                <input 
                  type="text" 
                  className="textarea-input !min-h-0" 
                  placeholder="e.g. competitor_intel"
                  value={customCategory}
                  onChange={(e) => setCustomCategory(e.target.value)}
                />
              </div>
            )}

            <div className="mt-8 flex justify-end">
              <button className="btn btn-primary" onClick={processChunks} disabled={loading}>
                {loading ? 'Processing...' : 'Review Chunks'} <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        );
      case 4:
        return (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
             <div className="flex justify-between items-center mb-6">
               <h2 className="text-3xl font-bold">Review Chunks ({chunks.length})</h2>
               <div className="flex gap-4">
                  <button className="btn btn-outline" onClick={() => setStep(3)}>Back</button>
                  <button className="btn btn-primary" onClick={saveToMemory} disabled={loading}>
                    <CheckCircle2 className="w-4 h-4" /> {loading ? 'Saving...' : 'Confirm & Save'}
                  </button>
               </div>
             </div>
             
             <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-4 custom-scrollbar">
                {chunks.map((chunk) => (
                  <div key={chunk.id} className="glass chunk-card animate-fade-in">
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex-1">
                        <label className="text-xs font-bold uppercase tracking-wider text-primary-color mb-2 block">Retrieval Question (DNA)</label>
                        <input 
                          type="text" 
                          className="textarea-input !min-h-0 !p-3 bg-white/5" 
                          placeholder="What question does this cell answer?"
                          value={chunk.description}
                          onChange={(e) => updateChunk(chunk.id, 'description', e.target.value)}
                        />
                      </div>
                      <button className="p-2 text-error hover:bg-error/10 rounded-lg mt-8" onClick={() => removeChunk(chunk.id)}>
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                    <div>
                      <label className="text-xs font-bold uppercase tracking-wider text-secondary-color mb-2 block">Content</label>
                      <textarea 
                        className="textarea-input !min-h-0 !p-3 h-32 text-sm" 
                        value={chunk.content}
                        onChange={(e) => updateChunk(chunk.id, 'content', e.target.value)}
                      />
                    </div>
                  </div>
                ))}
             </div>
          </motion.div>
        );
      case 5:
        return (
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex flex-col items-center justify-center py-12 text-center"
          >
            <div className="w-20 h-20 rounded-full bg-success/20 flex items-center justify-center mb-6 border-2 border-success">
              <CheckCircle2 className="w-10 h-10 text-success" />
            </div>
            <h2 className="text-4xl font-bold mb-2">Memory Updated</h2>
            <p className="text-dim text-lg mb-8">Tee can use new cells immediately in conversation.</p>

            {saveResult && (
              <div className="w-full max-w-xl space-y-4 text-left">

                {/* Saved */}
                <div className="glass p-4 rounded-xl border border-success/30">
                  <div className="flex items-center gap-2 font-semibold text-success mb-2">
                    <CheckCircle2 className="w-4 h-4" /> {saveResult.saved} cell{saveResult.saved !== 1 ? 's' : ''} saved
                  </div>
                  {saveResult.saved_cells.length > 0 && (
                    <ul className="text-xs text-dim space-y-1 max-h-32 overflow-y-auto custom-scrollbar pr-1">
                      {saveResult.saved_cells.map((c, i) => (
                        <li key={i} className="truncate">✓ {c.description}</li>
                      ))}
                    </ul>
                  )}
                </div>

                {/* Warnings — near-dupes that went through */}
                {saveResult.warnings.length > 0 && (
                  <div className="glass p-4 rounded-xl border border-yellow-500/30">
                    <div className="flex items-center gap-2 font-semibold text-yellow-400 mb-2">
                      <AlertCircle className="w-4 h-4" /> {saveResult.warnings.length} near-duplicate{saveResult.warnings.length !== 1 ? 's' : ''} (saved with warning)
                    </div>
                    <ul className="text-xs text-dim space-y-2 max-h-40 overflow-y-auto custom-scrollbar pr-1">
                      {saveResult.warnings.map((c, i) => (
                        <li key={i}>
                          <span className="text-white">↑ {c.description}</span>
                          <br />
                          <span className="text-yellow-400/70">{c.warning}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Duplicates — blocked */}
                {saveResult.duplicates.length > 0 && (
                  <div className="glass p-4 rounded-xl border border-error/30">
                    <div className="flex items-center gap-2 font-semibold text-error mb-2">
                      <AlertCircle className="w-4 h-4" /> {saveResult.duplicates.length} duplicate{saveResult.duplicates.length !== 1 ? 's' : ''} blocked
                    </div>
                    <ul className="text-xs text-dim space-y-2 max-h-40 overflow-y-auto custom-scrollbar pr-1">
                      {saveResult.duplicates.map((d, i) => (
                        <li key={i}>
                          <span className="text-white line-through opacity-50">{d.description}</span>
                          <br />
                          <span className="text-error/70">sim={d.sim} — already covered by: "{d.existing}"</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <button className="btn btn-primary mt-10 px-8" onClick={() => window.location.reload()}>
              Ingest Another
            </button>
          </motion.div>
        );
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="logo">
          <Layers className="w-8 h-8" />
          <span>SalesTee</span>
        </div>
        
        <nav className="nav-links">
          <div className={`nav-item ${step === 1 ? 'active' : ''}`}><Upload className="w-5 h-5" /> 1. Upload</div>
          <div className={`nav-item ${step === 2 ? 'active' : ''}`}><Type className="w-5 h-5" /> 2. Content Type</div>
          <div className={`nav-item ${step === 3 ? 'active' : ''}`}><Edit3 className="w-5 h-5" /> 3. Details</div>
          <div className={`nav-item ${step === 4 ? 'active' : ''}`}><CheckCircle2 className="w-5 h-5" /> 4. Review</div>
        </nav>

        <div className="mt-auto glass p-6 rounded-2xl">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-primary-color/20 rounded-lg"><BrainCircuit className="w-5 h-5 text-primary-color" /></div>
            <h5 className="font-bold">Substrate Status</h5>
          </div>
          <div className="space-y-2 text-sm text-dim">
            <div className="flex justify-between"><span>Status</span> <span className="text-success">Online</span></div>
            <div className="flex justify-between"><span>Memory</span> <span>1.2 GB</span></div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <AnimatePresence mode="wait">
          {renderStep()}
        </AnimatePresence>
      </main>
    </div>
  );
}
