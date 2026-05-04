import { useState, useEffect } from 'react';
import OrganismLattice from './components/OrganismLattice';
import { NeuralComparison } from './components/NeuralComparison';
import './App.css';

interface AssimilationStatus {
    methodology_cells: number;
    avg_connections: number;
    phase: string;
    phase_2_ready: boolean;
    avg_similarity: number;
    interactions_total: number;
    phase_2_status: string;
}

const App = () => {
    const API_BASE = window.location.origin;

    const [status, setStatus] = useState<AssimilationStatus | null>({
        methodology_cells: 0,
        avg_connections: 0,
        phase: "BOOTING...",
        phase_2_ready: false,
        avg_similarity: 0,
        interactions_total: 0,
        phase_2_status: "Initializing neural weights..."
    });
    const [metrics, setMetrics] = useState<any>({});
    const [view, setView] = useState<'lattice' | 'comparison'>('lattice');
    const [logs, setLogs] = useState<string[]>([]);

    useEffect(() => {
        const poll = setInterval(async () => {
            try {
                const [sRes, mRes] = await Promise.all([
                    fetch(`${API_BASE}/assimilation_status`).then(r => r.json()),
                    fetch(`${API_BASE}/telemetry`).then(r => r.json())
                ]);
                setStatus(sRes);
                setMetrics(mRes);
                if (mRes.activity_log) setLogs(mRes.activity_log);
            } catch (e) {
                console.warn("Neural Signal Interrupted");
            }
        }, 1000);
        return () => clearInterval(poll);
    }, [API_BASE]);

    return (
        <div className="dashboard-root">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-full flex items-center justify-center neon-border" style={{backgroundColor: 'rgba(0, 255, 180, 0.1)'}}>
                        <span className="neon-text animate-pulse">⚡</span>
                    </div>
                    <div>
                        <h1 className="text-xl font-extrabold tracking-tighter" style={{margin: 0}}>DATAG <span className="opacity-40" style={{fontSize: '10px'}}>PRIME</span></h1>
                        <div className="text-10px text-neon font-mono tracking-widest" style={{fontSize: '10px'}}>NEURAL_STATUS: ACTIVE</div>
                    </div>
                </div>

                <div className="mt-8">
                    <label className="stat-label">ACTIVE PHASE</label>
                    <div className="p-5 rounded-2xl" style={{backgroundColor: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)'}}>
                        <div className="neon-text font-bold text-lg mb-1">{status?.phase || "BOOTING..."}</div>
                        <div className="text-xs opacity-50 leading-relaxed font-light">{status?.phase_2_status || "Initializing neural weights..."}</div>
                    </div>
                </div>

                <div className="flex-1 overflow-hidden flex flex-col mt-4">
                    <label className="stat-label">ACTIVITY LOG</label>
                    <div className="flex-1 overflow-y-auto custom-scrollbar font-mono text-10px" style={{fontSize: '10px', color: 'rgba(255,255,255,0.4)'}}>
                        {logs.slice(0, 50).map((log, i) => (
                            <div key={i} className="mb-1 opacity-70 hover:opacity-100 transition-opacity" style={{borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px'}}>
                                {log}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="flex flex-col gap-3 mt-auto">
                    <button
                        onClick={() => setView('lattice')}
                        className={`tab-button ${view === 'lattice' ? 'active' : ''}`}
                    >
                        CORE LATTICE
                    </button>
                    <button
                        onClick={() => setView('comparison')}
                        className={`tab-button ${view === 'comparison' ? 'active' : ''}`}
                    >
                        SHADOW FEED
                    </button>
                </div>
            </aside>

            {/* Main Content Area */}
            <div className="content-area">
                {/* Header Stats */}
                <header className="header">
                    <div className="flex gap-12">
                        <div className="stat-group">
                            <div className="stat-label">DNA ENTROPY</div>
                            <div className="text-2xl font-extrabold font-mono" style={{color: 'rgba(255,255,255,0.9)'}}>{(metrics.entropy || "0.00").toString().slice(0, 6)}</div>
                        </div>
                        <div className="stat-group">
                            <div className="stat-label">CELL COUNT</div>
                            <div className="text-2xl font-extrabold font-mono" style={{color: 'rgba(255,255,255,0.9)'}}>{metrics.total_records || 3833}</div>
                        </div>
                        <div className="stat-group">
                            <div className="stat-label">INDEPENDENCE</div>
                            <div className={`text-2xl font-extrabold font-mono ${metrics.independence >= 0.8 ? 'text-neon' : ''}`} style={{color: metrics.independence >= 0.8 ? '' : 'rgba(255,255,255,0.9)'}}>
                                {(metrics.independence * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-4 px-4 py-2 rounded-2xl" style={{backgroundColor: 'rgba(0, 255, 180, 0.05)', border: '1px solid var(--neon-dim)'}}>
                        <div className="w-2 h-2 rounded-full bg-neon animate-pulse shadow-lg"></div>
                        <div className="text-10px font-bold tracking-widest text-neon" style={{fontSize: '10px'}}>DATA_G_REALTIME_SYNC (STREAMING_OK)</div>
                    </div>
                </header>

                <main className="main-viewport">
                    {view === 'lattice' ? (
                        <div className="canvas-container">
                            <OrganismLattice
                                active={true}
                                anomalyCount={metrics.anomaly_count || 0}
                                activeAdapters={metrics.active_adapters || 0}
                                totalCells={metrics.total_records || 3833}
                                similarity={status?.avg_similarity || 0}
                            />
                            <div className="absolute" style={{top: '2.5rem', left: '2.5rem', pointerEvents: 'none'}}>
                                <h2 className="text-2xl font-bold tracking-widest opacity-80 uppercase" style={{margin: 0}}>Neural Organism</h2>
                                <p className="text-xs font-light mt-2" style={{color: 'rgba(0, 255, 180, 0.6)'}}>Simulating Methodology Cells...</p>
                            </div>
                        </div>
                    ) : (
                        <NeuralComparison />
                    )}
                </main>
            </div>
        </div>
    );
};

export default App;
