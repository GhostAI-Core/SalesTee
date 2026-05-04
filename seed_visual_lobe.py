"""
Visual Lobe Seeder
==================
One-shot script. Seeds meth_visual_* cells with domain-appropriate content
covering: scene description, spatial reasoning, diagram/chart reading,
UI layout, color/texture/shape, image analysis patterns, visual metaphor.

Run from DataG root: python3 seed_visual_lobe.py
"""

import os
import sys
import hashlib
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "src"))
os.chdir(_THIS)

SEEDS = [
    # ── Scene description ──────────────────────────────────────────────────────
    "The image shows a wide open field at dusk, horizon split by a line of trees. Long shadows stretch left.",
    "A cluttered desk: monitor on the left, notebook open at center, coffee cup top-right, cables trailing down.",
    "Overhead view of a city grid. Blocks are rectangular, streets form a lattice, a diagonal avenue cuts top-left to bottom-right.",
    "Close-up of a circuit board. Green substrate, silver traces radiating from a central IC, capacitors arranged in rows.",
    "A face in three-quarter profile, light source upper-left, shadow fills the right cheek, eyes tracking slightly downward.",
    "Rain-streaked window. Drops elongate into streaks toward the bottom. City lights blur into vertical smears behind the glass.",
    "A staircase descending left. Each step casts a triangular shadow. The vanishing point sits at the bottom-center.",
    "Dense forest canopy from below. Leaves overlap, light breaks through in scattered bright patches between dark gaps.",
    "Satellite image: coastline curves gently southeast, shallow water appears turquoise fading to deep blue offshore.",
    "A busy market stall. Produce arranged in columns by color — reds left, greens center, yellows right.",

    # ── Spatial reasoning ─────────────────────────────────────────────────────
    "Object A is left of object B. Object C is above object B. Therefore C is above and right of A.",
    "The red shape is inside the blue rectangle. The blue rectangle is inside the green circle.",
    "Depth cue: the smaller figure appears farther away because it overlaps nothing and sits higher in frame.",
    "Two objects partially overlap. The front object occludes the rear one along its left edge.",
    "Rotate 90 degrees clockwise: top becomes right, right becomes bottom, bottom becomes left, left becomes top.",
    "Mirror along vertical axis: left features move right, right features move left. Up/down unchanged.",
    "The arrow points toward the upper-right quadrant, approximately 45 degrees from horizontal.",
    "Three boxes stacked: bottom is largest, middle medium, top smallest. Center of mass is in the lower third.",
    "The path goes: forward 3 steps, turn right, forward 2 steps, turn left, forward 1 step.",
    "Foreground objects are sharp and large. Background objects are blurred and small — depth of field effect.",

    # ── Diagram and chart reading ──────────────────────────────────────────────
    "Bar chart: y-axis is frequency, x-axis shows months. January peak, July trough, December secondary peak.",
    "Line graph: two series cross at month 4. Before the cross, series A is higher. After, series B dominates.",
    "Pie chart with 4 slices: largest slice (~40%) labeled Revenue, smallest (~8%) labeled Other.",
    "Flow diagram: start node → decision diamond → two branches (yes/no) → merge → end node.",
    "Scatter plot: positive correlation. Points cluster along a rising diagonal. Two outliers upper-left.",
    "Venn diagram: three overlapping circles. The center intersection represents elements common to all three sets.",
    "Network diagram: central hub node connects to 6 leaf nodes. Two leaf nodes also connect to each other.",
    "Gantt chart: tasks A, B, C on y-axis. Time on x-axis. B starts after A ends. C overlaps both.",
    "Heatmap: darker cells indicate higher values. Top-right quadrant is consistently darker than bottom-left.",
    "Histogram: right-skewed distribution. Most values cluster near zero, long tail extends rightward.",

    # ── UI and layout reading ──────────────────────────────────────────────────
    "Navigation bar at top. Hero image spans full width below. Three-column feature grid below that.",
    "Left sidebar for navigation, main content panel center, right sidebar for metadata. Standard three-panel layout.",
    "Modal dialog overlays dimmed background. Close button top-right, primary action bottom-right, cancel bottom-left.",
    "Card component: image top, title below image, subtitle below title, action button bottom-right of card.",
    "Form layout: labels left-aligned, input fields right of labels, submit button centered below last field.",
    "Breadcrumb trail at page top: Home > Category > Subcategory > Current Page.",
    "Tab interface: four tabs across top, active tab visually distinct (underline/fill), content panel below.",
    "Progress stepper: five steps in a row. Steps 1–3 complete (filled), step 4 active, step 5 incomplete.",
    "Dropdown expanded: list of options extends downward below trigger button, selected item highlighted.",
    "Toast notification: bottom-right corner, brief message, auto-dismisses after 3 seconds.",

    # ── Color, texture, shape ──────────────────────────────────────────────────
    "Warm colors (red, orange, yellow) advance visually. Cool colors (blue, green, violet) recede.",
    "Complementary colors sit opposite on the wheel: red/green, blue/orange, yellow/violet.",
    "High contrast edge: sharp luminance boundary between adjacent regions signals an object edge.",
    "Texture gradient: fine texture in foreground becoming coarser toward background indicates recession.",
    "Regular repeating pattern — grid, stripe, or checkerboard — signals a manufactured surface.",
    "Smooth gradient from dark to light across a curved surface indicates a convex 3D form under directional light.",
    "Broken or irregular edges suggest organic material. Straight, crisp edges suggest manufactured objects.",
    "Silhouette: shape alone, no internal detail. Recognition relies entirely on outer boundary contour.",
    "Saturation drop in a region indicates shadow. Hue typically shifts slightly cooler in shadow areas.",
    "Transparency: object behind partially visible. Color is a blend of foreground and background colors.",

    # ── Image analysis patterns ────────────────────────────────────────────────
    "Describe what you see: identify the dominant object, its position, its relation to other elements.",
    "Identify the light source direction by observing which faces of 3D objects are bright versus shadowed.",
    "Count distinct regions by grouping pixels of similar color and texture into coherent areas.",
    "Detect anomalies: locate regions that differ in color, texture, or pattern from surrounding context.",
    "Track motion blur: streaked objects indicate direction and speed of movement relative to shutter speed.",
    "Estimate scale by comparing unknown object to a known reference object in the same frame.",
    "Identify symmetry axes: fold the image mentally — regions that mirror each other define the axis.",
    "Read emotion from facial geometry: raised brows, downturned mouth, eye narrowing each map to affect states.",
    "Detect text regions by looking for high-frequency horizontal patterns in small rectangular areas.",
    "Assess image quality: sharpness of edges, presence of noise, compression artifacts, dynamic range.",

    # ── Visual metaphor and analogy ────────────────────────────────────────────
    "A branching tree structure maps onto hierarchy: trunk is root, branches are children, leaves are terminals.",
    "A funnel shape represents filtering or narrowing: wide input at top, constrained output at bottom.",
    "A bridge in an image implies connection between two otherwise separated regions or concepts.",
    "A magnifying glass over a region signals: zoom in, examine closely, this area is significant.",
    "A clock or hourglass in context signals time pressure, deadline, or temporal passage.",
    "A broken chain link signals a failure in sequence or a severed dependency.",
    "An upward arrow on a graph always implies growth, increase, or positive direction — regardless of units.",
    "Overlapping circles in a diagram signal shared properties, intersection, or integration.",
    "A red X universally signals failure, error, or prohibition in UI and diagram contexts.",
    "A checkmark signals completion, correctness, or approval — green checkmark amplifies the positive signal.",
]


def run():
    from system import AgenticSystem
    from living_cell import LivingCell

    print("[VisualSeeder] Loading substrate...")
    substrate = AgenticSystem(hdc_dim=384, slim=True, skip_cells=True)
    print(f"[VisualSeeder] Substrate online — {len(substrate.methodology_cells):,} existing cells")

    stored = 0
    skipped = 0

    for text in SEEDS:
        uid = hashlib.md5(text.encode()).hexdigest()[:14]
        cell_id = f"meth_visual_{uid}"

        if cell_id in substrate.methodology_cells:
            skipped += 1
            continue

        dna = np.array(substrate.hdc.encode(text), dtype=np.float32)
        norm = np.linalg.norm(dna)
        if norm > 1e-8:
            dna = dna / norm

        cell = LivingCell(
            cell_id,
            content=text,
            source_table="visual_seed",
            confidence=1.2,
            source="visual_seed",
        )
        cell.dna    = dna
        cell.anchor = text[:200]

        substrate.methodology_cells[cell_id] = cell
        substrate._persist_methodology_cell(cell)
        stored += 1

    print(f"[VisualSeeder] Done — {stored} new cells stored, {skipped} already existed.")
    print(f"[VisualSeeder] Visual pool now has {sum(1 for c in substrate.methodology_cells if c.startswith('meth_visual'))} cells total.")


if __name__ == "__main__":
    run()
