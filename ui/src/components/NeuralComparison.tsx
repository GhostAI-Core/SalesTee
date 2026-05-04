import { useState, useEffect } from 'react';

interface ComparisonData {
    teacher: any;
    student: any;
    ia_truth: any;
    similarity: number;
    filename: string;
    merit_win: boolean;
}

export const NeuralComparison = () => {
    const [data, setData] = useState<ComparisonData | null>(null);

    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/api/comparison');
                const d = await res.json();
                if (d && d.teacher) setData(d);
            } catch (e) {}
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    if (!data || !data.teacher) return (
        <div className="empty-state">
            <div className="icon">📥</div>
            <div className="label">Awaiting Neural Signal...</div>
        </div>
    );

    const fields = [
        { label: 'Name', key: 'name' },
        { label: 'Email', key: 'email' },
        { label: 'Location', key: 'location' },
        { label: 'Summary', key: 'summary' },
        { label: 'Industry', key: 'industry' },
        { label: 'Skills', key: 'skills', list: true },
        { label: 'Experience', key: 'experience', list: true },
        { label: 'Timeline', key: 'experience_timeline', list: true },
        { label: 'Education', key: 'education', list: true }
    ];

    const getStatus = (t: any, s: any) => {
        if (!s || s === "Not specified" || s === "") return { label: 'MISSING', className: 'status-missing' };
        const tStr = JSON.stringify(t).toLowerCase();
        const sStr = JSON.stringify(s).toLowerCase();
        if (tStr === sStr) return { label: 'MATCHED', className: 'status-matched' };
        if (sStr.length > tStr.length && sStr.includes(tStr.replace(/["']/g, ''))) return { label: 'RESOLUTION+', className: 'status-resolution' };
        return { label: 'DIVERGED', className: 'status-diverged' };
    };

    return (
        <div className="shadow-feed-container">
            <div className="feed-header-info">
                <div className="title-group">
                    <h2 className="feed-title">Shadow Feed <span className="separator">/</span> <span className="filename">{data.filename}</span></h2>
                    <div className="feed-subtitle">Neural Convergence Monitoring</div>
                </div>
                <div className="similarity-group">
                    <div className="stat-label">Similarity Index</div>
                    <div className={`similarity-value ${data.similarity >= 0.8 ? 'text-neon' : 'text-amber'}`}>
                        {(data.similarity * 100).toFixed(1)}%
                    </div>
                </div>
            </div>

            <div className="column-headers">
                <div className="header-label teacher">Teacher (Ground Truth)</div>
                <div className="header-label student">Student (IA Model)</div>
                <div className="header-label datag">DataG (Neural Memory)</div>
            </div>

            <div className="feed-content custom-scrollbar">
                {fields.map(field => {
                    const t = data.teacher[field.key];
                    const s = data.student[field.key];
                    const d = data.ia_truth[field.key];
                    const status = getStatus(t, s);

                    return (
                        <div key={field.key} className="feed-row">
                            <div className="row-meta">
                                <label className="field-label">{field.label}</label>
                                <div className={`status-badge ${status.className}`}>
                                    {status.label}
                                </div>
                            </div>
                            
                            <div className="cells-grid">
                                <div className="feed-cell teacher-cell">
                                    {field.list ? (Array.isArray(t) ? t.map((item: any, i: number) => <div key={i} className="list-item">- {typeof item === 'object' ? JSON.stringify(item) : item}</div>) : '[]') : String(t)}
                                </div>
                                <div className="feed-cell student-cell">
                                    {field.list ? (Array.isArray(s) ? s.map((item: any, i: number) => <div key={i} className="list-item">- {typeof item === 'object' ? JSON.stringify(item) : item}</div>) : '[]') : String(s)}
                                </div>
                                <div className="feed-cell datag-cell">
                                    {field.list ? (Array.isArray(d) ? d.map((item: any, i: number) => <div key={i} className="list-item">- {typeof item === 'object' ? JSON.stringify(item) : item}</div>) : '[]') : String(d)}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {data.merit_win && (
                <div className="merit-banner">
                    <div className="merit-title">Merrit Evolution Detected</div>
                    <div className="merit-desc">Student resolution exceeds teacher. DataG is synthesizing organic memory.</div>
                </div>
            )}
        </div>
    );
};
