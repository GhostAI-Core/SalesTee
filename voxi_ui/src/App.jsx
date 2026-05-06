import { useState, useRef, useEffect } from 'react'
import './App.css'

const API = ''

const STATES = {
  IDLE:      'idle',
  LOADING:   'loading',   // fetching opening line
  LISTENING: 'listening', // recording mic
  THINKING:  'thinking',  // awaiting API response
  SPEAKING:  'speaking',  // TTS playing
}

export default function App() {
  const [phase,       setPhase]       = useState(STATES.IDLE)
  const [transcript,  setTranscript]  = useState([])  // [{role:'tee'|'you', text}]
  const [statusText,  setStatusText]  = useState('')
  const [usedCells,   setUsedCells]   = useState([])
  const [history,     setHistory]     = useState([])
  const [product]                     = useState('voxi')

  const mediaRef      = useRef(null)
  const chunksRef     = useRef([])
  const transcriptEnd = useRef(null)

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [transcript])

  // ── Start session ────────────────────────────────────────────────────────────
  async function startSession() {
    setPhase(STATES.LOADING)
    setTranscript([])
    setHistory([])
    setUsedCells([])

    try {
      const res  = await fetch(`${API}/chat/open/${product}`)
      const data = await res.json()
      const opening = data.opening || "Hi, I'm Tee. What would you like to know?"
      addLine('tee', opening)
      await speakText(opening)
      listenLoop()
    } catch (e) {
      setStatusText('Could not reach Tee. Is the backend running?')
      setPhase(STATES.IDLE)
    }
  }

  // ── Mic recording (MediaRecorder → Whisper via /transcribe) ─────────────────
  async function listenLoop() {
    setPhase(STATES.LISTENING)
    setStatusText('Listening…')

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setStatusText('Microphone access denied.')
      setPhase(STATES.IDLE)
      return
    }

    const recorder = new MediaRecorder(stream)
    chunksRef.current = []
    mediaRef.current  = recorder

    recorder.ondataavailable = e => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop())
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      await processAudio(blob)
    }

    // Use VAD-style silence detection: stop after 1.8s of silence via
    // AudioContext analyser, max 15s hard cap
    await vadRecord(recorder, stream, 1800, 15000)
  }

  async function vadRecord(recorder, stream, silenceMs, maxMs) {
    const ctx      = new AudioContext()
    const source   = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    source.connect(analyser)

    const buf         = new Uint8Array(analyser.frequencyBinCount)
    let   silenceStart = null
    let   speechSeen   = false
    const startTime    = Date.now()

    recorder.start(100)

    await new Promise(resolve => {
      const check = () => {
        analyser.getByteFrequencyData(buf)
        const rms = buf.reduce((a, b) => a + b, 0) / buf.length

        if (rms > 8) {
          speechSeen  = true
          silenceStart = null
        } else if (speechSeen) {
          if (!silenceStart) silenceStart = Date.now()
          else if (Date.now() - silenceStart > silenceMs) {
            resolve()
            return
          }
        }

        if (Date.now() - startTime > maxMs) { resolve(); return }
        requestAnimationFrame(check)
      }
      requestAnimationFrame(check)
    })

    recorder.stop()
    ctx.close()
  }

  // ── Send audio to Whisper STT endpoint ──────────────────────────────────────
  async function processAudio(blob) {
    setPhase(STATES.THINKING)
    setStatusText('Thinking…')

    // Transcribe via backend
    let userText
    try {
      const form = new FormData()
      form.append('file', blob, 'recording.webm')
      const res  = await fetch(`${API}/transcribe`, { method: 'POST', body: form })
      const data = await res.json()
      userText = (data.text || '').trim()
    } catch {
      setStatusText('Transcription failed.')
      setPhase(STATES.IDLE)
      return
    }

    if (!userText) {
      listenLoop()
      return
    }

    addLine('you', userText)

    // Chat
    let teeText
    try {
      const res  = await fetch(`${API}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          message:    userText,
          product,
          history,
          used_cells: usedCells,
        }),
      })
      const data = await res.json()
      teeText = data.response

      setHistory(h => [...h, { user: userText, tee: teeText }])
      if (data.cell_id) setUsedCells(u => [...u.slice(-5), data.cell_id])
    } catch {
      setStatusText('No response from Tee.')
      setPhase(STATES.IDLE)
      return
    }

    addLine('tee', teeText)
    await speakText(teeText)
    listenLoop()
  }

  // ── TTS via Web Speech API (no backend needed) ───────────────────────────────
  function speakText(text) {
    return new Promise(resolve => {
      setPhase(STATES.SPEAKING)
      setStatusText('Tee is speaking…')
      const utt = new SpeechSynthesisUtterance(text)
      utt.lang  = 'en-US'
      utt.rate  = 1.05
      utt.onend = resolve
      utt.onerror = resolve
      window.speechSynthesis.speak(utt)
    })
  }

  function addLine(role, text) {
    setTranscript(t => [...t, { role, text }])
  }

  function endSession() {
    window.speechSynthesis.cancel()
    mediaRef.current?.stop()
    setPhase(STATES.IDLE)
    setStatusText('')
  }

  const isActive = phase !== STATES.IDLE

  return (
    <div className="shell">
      <header className="topbar">
        <span className="logo">VOXI</span>
        <span className="tagline">powered by Tee</span>
      </header>

      <main className="stage">
        <div className={`orb-wrap ${phase}`}>
          <div className="orb-ring ring1" />
          <div className="orb-ring ring2" />
          <div className="orb-ring ring3" />
          <button
            className={`orb ${phase}`}
            onClick={isActive ? endSession : startSession}
            aria-label={isActive ? 'End session' : 'Talk to Tee'}
          >
            {phase === STATES.IDLE     && <MicIcon />}
            {phase === STATES.LOADING  && <SpinIcon />}
            {phase === STATES.LISTENING && <WaveIcon />}
            {phase === STATES.THINKING  && <SpinIcon />}
            {phase === STATES.SPEAKING  && <SpeakIcon />}
          </button>
        </div>

        <p className="status-label">
          {phase === STATES.IDLE
            ? 'Tap to talk to Tee'
            : statusText}
        </p>
      </main>

      {transcript.length > 0 && (
        <div className="transcript-panel">
          <div className="transcript-scroll">
            {transcript.map((line, i) => (
              <div key={i} className={`line ${line.role}`}>
                <span className="speaker">{line.role === 'tee' ? 'Tee' : 'You'}</span>
                <span className="text">{line.text}</span>
              </div>
            ))}
            <div ref={transcriptEnd} />
          </div>
        </div>
      )}
    </div>
  )
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="2" width="6" height="13" rx="3"/>
      <path d="M5 10a7 7 0 0014 0"/>
      <line x1="12" y1="19" x2="12" y2="22"/>
      <line x1="9" y1="22" x2="15" y2="22"/>
    </svg>
  )
}
function WaveIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <line x1="2" y1="12" x2="2" y2="12"/>
      <line x1="6" y1="8" x2="6" y2="16"/>
      <line x1="10" y1="5" x2="10" y2="19"/>
      <line x1="14" y1="8" x2="14" y2="16"/>
      <line x1="18" y1="10" x2="18" y2="14"/>
      <line x1="22" y1="12" x2="22" y2="12"/>
    </svg>
  )
}
function SpeakIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
      <path d="M15.54 8.46a5 5 0 010 7.07"/>
      <path d="M19.07 4.93a10 10 0 010 14.14"/>
    </svg>
  )
}
function SpinIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="spin">
      <path d="M12 2a10 10 0 110 20A10 10 0 0112 2z" opacity="0.25"/>
      <path d="M12 2a10 10 0 0110 10"/>
    </svg>
  )
}
