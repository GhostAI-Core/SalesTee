import React, { useEffect, useRef, useMemo } from 'react';

interface LatticeProps {
    active?: boolean;
    anomalyCount?: number;
    totalCells?: number;
    activeAdapters?: number;
    similarity?: number;
}

interface Node {
    id: string;
    r: number;
    theta: number;
    phi: number;
    thetaSpeed: number;
    phiSpeed: number;
    color: string;
    isAnomaly: boolean;
    isActiveNode: boolean;
    size: number;
    projX?: number;
    projY?: number;
    projZ?: number;
}

const OrganismLattice: React.FC<LatticeProps> = ({ 
    active = false, 
    anomalyCount = 0, 
    activeAdapters = 0, 
    similarity = 0 
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number>(0);
    const mouseRef = useRef<{ x: number; y: number } | null>(null);
    const camRef = useRef({ 
        zoom: 1.0, 
        rotX: 0, 
        rotY: 0, 
        isDragging: false, 
        lastX: 0, 
        lastY: 0 
    });

    const nodes = useMemo(() => {
        const items: Node[] = [];
        for (let i = 0; i < 3833; i++) {
            const isCore = i < 200;
            const r = isCore ? Math.random() * 300 : Math.random() * 1800;
            const theta = Math.random() * 2 * Math.PI;
            const phi = Math.acos((Math.random() * 2) - 1);

            items.push({
                id: `node_${i}`,
                r: r,
                theta: theta,
                phi: phi,
                thetaSpeed: (isCore ? 0.008 : 0.002) * (Math.random() - 0.5),
                phiSpeed: (isCore ? 0.005 : 0.001) * (Math.random() - 0.5),
                color: '#ffffff',
                isAnomaly: false,
                isActiveNode: false,
                size: isCore ? 2 + Math.random() * 3 : 0.8 + Math.random() * 1.5,
            });
        }
        return items;
    }, []);

    useEffect(() => {
        let baseColor = '#00ffb4';
        if (similarity >= 0.9) baseColor = '#fbbf24';
        else if (similarity >= 0.5) baseColor = '#8b5cf6';

        for (let i = 0; i < nodes.length; i++) {
            nodes[i].isAnomaly = (i < anomalyCount);
            nodes[i].isActiveNode = (!nodes[i].isAnomaly && i < (anomalyCount + activeAdapters));

            if (nodes[i].isAnomaly) nodes[i].color = '#ff4444';
            else if (nodes[i].isActiveNode) nodes[i].color = baseColor;
            else nodes[i].color = 'rgba(255, 255, 255, 0.2)'; 
        }
    }, [anomalyCount, activeAdapters, nodes, similarity]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const resize = () => {
            canvas.width = canvas.offsetWidth * window.devicePixelRatio;
            canvas.height = canvas.offsetHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        resize();
        window.addEventListener('resize', resize);

        const handleMouseMove = (e: MouseEvent) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            mouseRef.current = { x, y };

            const cam = camRef.current;
            if (cam.isDragging) {
                cam.rotY += (x - cam.lastX) * 0.005;
                cam.rotX += (y - cam.lastY) * 0.005;
                cam.rotX = Math.max(-1.5, Math.min(1.5, cam.rotX));
                cam.lastX = x;
                cam.lastY = y;
            }
        };

        const handleMouseDown = (e: MouseEvent) => {
            const rect = canvas.getBoundingClientRect();
            const cam = camRef.current;
            cam.isDragging = true;
            cam.lastX = e.clientX - rect.left;
            cam.lastY = e.clientY - rect.top;
            canvas.style.cursor = 'grabbing';
        };

        const handleMouseUp = () => {
            camRef.current.isDragging = false;
            canvas.style.cursor = 'default';
        };

        const handleMouseLeave = () => {
            mouseRef.current = null;
            camRef.current.isDragging = false;
            canvas.style.cursor = 'default';
        };

        const handleWheel = (e: WheelEvent) => {
            const cam = camRef.current;
            const zoomDelta = e.deltaY * -0.001;
            cam.zoom = Math.min(Math.max(0.1, cam.zoom + zoomDelta), 5.0);
        };

        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mouseup', handleMouseUp);
        canvas.addEventListener('mouseleave', handleMouseLeave);
        canvas.addEventListener('wheel', handleWheel, { passive: false });

        let t = 0;
        const draw = () => {
            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;
            const cx = w / 2;
            const cy = h / 2;
            ctx.clearRect(0, 0, w, h);
            t += 0.005;

            const cam = camRef.current;

            ctx.save();
            ctx.translate(cx, cy);
            ctx.scale(cam.zoom, cam.zoom);
            ctx.translate(-cx, -cy);

            for (const n of nodes) {
                n.theta += n.thetaSpeed * (active ? 2.5 : 1);
                n.phi += n.phiSpeed * (active ? 2.5 : 1);
                if (n.phi < 0 || n.phi > Math.PI) n.phiSpeed *= -1;

                const nx = n.r * Math.sin(n.phi) * Math.cos(n.theta);
                const ny = n.r * Math.cos(n.phi);
                const nz = n.r * Math.sin(n.phi) * Math.sin(n.theta);

                const cy1 = Math.cos(cam.rotX);
                const sy1 = Math.sin(cam.rotX);
                const y1 = ny * cy1 - nz * sy1;
                const z1 = ny * sy1 + nz * cy1;

                const cy2 = Math.cos(cam.rotY + (t * 0.2));
                const sy2 = Math.sin(cam.rotY + (t * 0.2));
                const px = nx * cy2 + z1 * sy2;
                const pz = -nx * sy2 + z1 * cy2;
                const py = y1;

                n.projX = px;
                n.projY = py;
                n.projZ = pz;
            }

            const sorted = [...nodes].sort((a, b) => (a.projZ || 0) - (b.projZ || 0));

            // Background Ripples for active mode
            if (active) {
                const baseCol = '#00ffb4';
                for (let r = 0; r < 3; r++) {
                    const rippleT = (t * 2 + r * 2.1) % 6.28;
                    const haloR = 250 + rippleT * 80;
                    const alpha = Math.max(0, 1 - (rippleT / 6.28));
                    ctx.beginPath();
                    ctx.strokeStyle = baseCol;
                    ctx.globalAlpha = alpha * 0.2;
                    ctx.arc(cx, cy, haloR, 0, Math.PI * 2);
                    ctx.stroke();
                }
            }

            // Connection Lines
            ctx.lineWidth = 0.5;
            const activeNodesList = sorted.filter(n => n.isActiveNode);
            for (let i = 0; i < activeNodesList.length; i++) {
                for (let j = i + 1; j < Math.min(i + 5, activeNodesList.length); j++) {
                    const n1 = activeNodesList[i];
                    const n2 = activeNodesList[j];
                    const dx = (n1.projX || 0) - (n2.projX || 0);
                    const dy = (n1.projY || 0) - (n2.projY || 0);
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 300) {
                        ctx.globalAlpha = (1 - dist / 300) * 0.2;
                        ctx.strokeStyle = n1.color;
                        ctx.beginPath();
                        ctx.moveTo(cx + (n1.projX || 0), cy + (n1.projY || 0));
                        ctx.lineTo(cx + (n2.projX || 0), cy + (n2.projY || 0));
                        ctx.stroke();
                    }
                }
            }

            for (const n of sorted) {
                const depthScale = Math.max(0.1, 0.6 + ((n.projZ || 0) / 1800));
                const r = n.size * depthScale;
                ctx.globalAlpha = n.isActiveNode ? 1.0 : (n.isAnomaly ? 0.8 : 0.2);

                if (n.isActiveNode || n.isAnomaly) {
                    const pulse = 1 + Math.sin(t * 5 + (n.projX || 0) * 0.01) * 0.2;
                    const grad = ctx.createRadialGradient(cx + (n.projX || 0), cy + (n.projY || 0), 0, cx + (n.projX || 0), cy + (n.projY || 0), r * 5 * pulse);
                    grad.addColorStop(0, n.color);
                    grad.addColorStop(1, 'transparent');
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(cx + (n.projX || 0), cy + (n.projY || 0), r * 5 * pulse, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = n.color;
                    ctx.beginPath();
                    ctx.arc(cx + (n.projX || 0), cy + (n.projY || 0), r, 0, Math.PI * 2);
                    ctx.fill();
                } else {
                    ctx.fillStyle = '#ffffff';
                    ctx.beginPath();
                    ctx.arc(cx + (n.projX || 0), cy + (n.projY || 0), r, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            ctx.restore();
            animRef.current = requestAnimationFrame(draw);
        };
        draw();
        
        return () => {
            cancelAnimationFrame(animRef.current);
            window.removeEventListener('resize', resize);
            canvas.removeEventListener('mousemove', handleMouseMove);
            canvas.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mouseup', handleMouseUp);
            canvas.removeEventListener('mouseleave', handleMouseLeave);
            canvas.removeEventListener('wheel', handleWheel);
        };
    }, [nodes, active, similarity]);

    return <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />;
};

export default OrganismLattice;
