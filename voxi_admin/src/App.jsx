import { useState, useEffect } from 'react'
import './App.css'

const API = ''

export default function App() {
  const [misses,   setMisses]   = useState([])   // [{query, count, product, top_sim}]
  const [selected, setSelected] = useState(null) // index into misses
  const [answer,   setAnswer]   = useState('')
  const [table,    setTable]    = useState('product_voxi')
  const [status,   setStatus]   = useState(null) // {ok, msg}
  const [saving,   setSaving]   = useState(false)
  const [filter,   setFilter]   = useState('')

  useEffect(() => { loadMisses() }, [])

  async function loadMisses() {
    try {
      const res  = await fetch(`${API}/misses?top_n=100`)
      const data = await res.json()
      setMisses(data.misses || [])
    } catch {
      setMisses([])
    }
  }

  function selectMiss(i) {
    setSelected(i)
    setAnswer('')
    setStatus(null)
  }

  async function submitAnswer() {
    if (!answer.trim() || selected === null) return
    setSaving(true)
    setStatus(null)

    const question = misses[selected].query
    try {
      const res  = await fetch(`${API}/misses/answer`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          question,
          answer: answer.trim(),
          source_table: table,
        }),
      })
      const data = await res.json()
      if (res.ok) {
        setStatus({ ok: true, msg: `Saved — ${data.cell_id}` })
        setAnswer('')
        // Remove from list
        setMisses(m => m.filter((_, i) => i !== selected))
        setSelected(null)
      } else {
        const detail = data.detail
        const msg = typeof detail === 'object'
          ? `Duplicate (sim=${detail.sim}): "${detail.existing}"`
          : detail
        setStatus({ ok: false, msg })
      }
    } catch (e) {
      setStatus({ ok: false, msg: 'Network error — is the backend running?' })
    } finally {
      setSaving(false)
    }
  }

  const visible = filter.trim()
    ? misses.filter(m => m.query.toLowerCase().includes(filter.toLowerCase()))
    : misses

  const selectedMiss = selected !== null ? misses[selected] : null

  return (
    <div className="shell">
      <header className="topbar">
        <span className="logo">VOXI Admin</span>
        <span className="tagline">Tee Learning Panel</span>
        <button className="refresh-btn" onClick={loadMisses} title="Refresh">↻</button>
      </header>

      <div className="body">
        {/* ── Left panel — miss list ──────────────────────────────── */}
        <aside className="miss-panel">
          <div className="panel-header">
            <span className="panel-title">Unanswered questions</span>
            <span className="badge">{misses.length}</span>
          </div>
          <input
            className="filter-input"
            placeholder="Filter…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
          <ul className="miss-list">
            {visible.length === 0 && (
              <li className="empty">No missed questions yet.</li>
            )}
            {visible.map((m, i) => {
              const realIdx = misses.indexOf(m)
              return (
                <li
                  key={i}
                  className={`miss-item ${selected === realIdx ? 'active' : ''}`}
                  onClick={() => selectMiss(realIdx)}
                >
                  <span className="miss-num">{i + 1}</span>
                  <span className="miss-query">{m.query}</span>
                  <span className="miss-meta">
                    {m.count > 1 && <span className="count-pill">×{m.count}</span>}
                    <span className={`sim-pill ${m.top_sim < 0.1 ? 'red' : 'yellow'}`}>
                      {(m.top_sim * 100).toFixed(0)}%
                    </span>
                  </span>
                </li>
              )
            })}
          </ul>
        </aside>

        {/* ── Right panel — answer editor ─────────────────────────── */}
        <section className="answer-panel">
          {!selectedMiss ? (
            <div className="empty-state">
              <p>Select a question on the left to write Tee's answer.</p>
            </div>
          ) : (
            <div className="editor">
              <div className="question-block">
                <label className="field-label">Question</label>
                <p className="question-text">{selectedMiss.query}</p>
                {selectedMiss.product && (
                  <span className="product-tag">{selectedMiss.product}</span>
                )}
              </div>

              <div className="answer-block">
                <label className="field-label" htmlFor="answer">
                  Tee's answer <span className="hint">(will be spoken word-for-word)</span>
                </label>
                <textarea
                  id="answer"
                  className="answer-area"
                  placeholder="Type the answer Tee should give…"
                  value={answer}
                  onChange={e => setAnswer(e.target.value)}
                  rows={6}
                  autoFocus
                />
              </div>

              <div className="table-block">
                <label className="field-label" htmlFor="table">Namespace</label>
                <select
                  id="table"
                  className="table-select"
                  value={table}
                  onChange={e => setTable(e.target.value)}
                >
                  <option value="product_voxi">product_voxi (product knowledge)</option>
                  <option value="reasoning">reasoning (objection handling)</option>
                  <option value="identity">identity (who Tee is)</option>
                  <option value="conversation">conversation (general)</option>
                </select>
              </div>

              {status && (
                <div className={`status-bar ${status.ok ? 'ok' : 'err'}`}>
                  {status.msg}
                </div>
              )}

              <div className="action-row">
                <button
                  className="submit-btn"
                  onClick={submitAnswer}
                  disabled={saving || !answer.trim()}
                >
                  {saving ? 'Saving…' : 'Add to Tee ✓'}
                </button>
                <button
                  className="skip-btn"
                  onClick={() => {
                    setSelected(null)
                    setAnswer('')
                    setStatus(null)
                  }}
                >
                  Skip
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
