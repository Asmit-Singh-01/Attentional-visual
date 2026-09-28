const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");
const termLog = document.getElementById("termLog");
const startBtn = document.getElementById("startBtn");

// Resize Canvas
function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// Architectural Nodes Positions (Geometry shifted lower for mobile landscape height)
const nodes = [
    { name: "Input Token Matrix", x: 0.15, y: 0.50 },
    { name: "QK Multi-Head Proj", x: 0.35, y: 0.40 },
    { name: "Softmax Noise Engine", x: 0.55, y: 0.60 },
    { name: "Pythia-70M SVD Rank", x: 0.75, y: 0.40 },
    { name: "ETF Geometry Fix", x: 0.90, y: 0.50 }
];

// Stages Configuration (Total Duration = Exact 40 Seconds)
const stages = [
    {
        title: "🟢 STAGE 1: High-Dim Input Token Vector Ingestion",
        color: "#00FFA3",
        eq: "\\[ S_{ij} = \\frac{\\mathbf{q}_i^\\top \\mathbf{k}_j}{\\sqrt{d_{head}}}, \\quad \\frac{N}{d_{head}} = 0.5 \\le 1.0 \\]",
        metrics: "> Frobenius Error: 0.0012 | Capacity Load: 50% [STABLE]",
        activeNode: 0,
        logs: [
            "[INFO] Parsing sequence embeddings...",
            "[MATH] Constructing Q, K spaces...",
            "[METRIC] Orthogonality metric stable.",
            "[STATUS] Subspace alignment safe."
        ]
    },
    {
        title: "🟡 STAGE 2: Query-Key Subspace Capacity Limit",
        color: "#FFD700",
        eq: "\\[ \\text{rank}(\\mathbf{W}_Q\\mathbf{W}_K^\\top) = d_{head}, \\quad \\frac{N}{d_{head}} = 1.0 \\]",
        metrics: "> Load Boundary Reached | Critical Threshold",
        activeNode: 1,
        logs: [
            "[WARN] Approaching N/d_head = 1.0",
            "[CALC] Computing singular values SVD...",
            "[SPECTRUM] Tail eigenvalues begin to flatten.",
            "[ALERT] Transition boundary localized."
        ]
    },
    {
        title: "🔴 STAGE 3: Softmax Exponential Interference Chaos",
        color: "#FF0055",
        eq: "\\[ \\mathcal{D}_{KL}(A \\| A_{true}) \\sim \\mathcal{O}\\left(e^{\\gamma (N - d_{head})}\\right) \\]",
        metrics: "> Cross-Talk Noise Amplified | Entropy EXPLOSION",
        activeNode: 2,
        logs: [
            "[CRITICAL] Ratio N/d_head = 1.85 > 1.0",
            "[MATH ERROR] Softmax explosion triggered!",
            "[ENTROPY] Cross-talk noise exploding.",
            "[SYSTEM] Context window lost."
        ]
    },
    {
        title: "🟣 STAGE 4: Pythia-70M SVD Rank Truncation Proof",
        color: "#9900FF",
        eq: "\\[ \\mathbf{A} = \\mathbf{U} \\mathbf{\\Sigma} \\mathbf{V}^\\top, \\quad \\sigma_i \\to 0 \\text{ for } i > d_{head} \\]",
        metrics: "> Effective Rank: 13/16 | Singular Floor Hit | Truncated",
        activeNode: 3,
        logs: [
            "[INSPECT] EleutherAI/Pythia-70M Layer 8...",
            "[SVD] Decomposing head matrix...",
            "[RESULT] Singular values hit zero line.",
            "[PROOF] Standard tricks cannot shift bound."
        ]
    },
    {
        title: "⚡ STAGE 5: ETF-Aware Initialization & Context Recovery",
        color: "#00E5FF",
        eq: "\\[ \\mathbf{M}^* = \\sqrt{\\frac{K}{K-1}} \\left( \\mathbf{I}_K - \\frac{1}{K}\\mathbf{1}\\mathbf{1}^\\top \\right) \\]",
        metrics: "> Reconstruction Fidelity: 99.89% | Zero Interference",
        activeNode: 4,
        logs: [
            "[ENG] Injecting ETF Initialization...",
            "[MATH] Maximizing feature pairwise angles...",
            "[RECOVERY] Dynamic Head Allocation rescales...",
            "[SUCCESS] Window restored. ZERO context loss."
        ]
    }
];

let isRunning = false;
let currentStage = 0;
let progress = 0; // 0 to 1 packet movement
let frameCount = 0;

// Render Loop (60 FPS GPU Canvas)
function drawLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;

    // 1. Draw Multi-Layer Circuit Connections
    ctx.strokeStyle = "#1a1a3a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i < nodes.length - 1; i++) {
        ctx.moveTo(nodes[i].x * w, nodes[i].y * h);
        ctx.lineTo(nodes[i + 1].x * w, nodes[i + 1].y * h);
    }
    ctx.stroke();

    // 2. Draw Nodes
    nodes.forEach((node, idx) => {
        const nx = node.x * w;
        const ny = node.y * h;
        const isActive = idx === stages[currentStage].activeNode;

        ctx.fillStyle = isActive ? stages[currentStage].color : "#0a0a1a";
        ctx.strokeStyle = isActive ? "#ffffff" : "#333366";
        ctx.lineWidth = isActive ? 3 : 1.5;

        ctx.beginPath();
        ctx.arc(nx, ny, isActive ? 18 : 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Node Title (Text positioned below node)
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px Courier New";
        ctx.fillText(node.name, nx - 40, ny + 35); // Text pushed DOWN
    });

    // 3. Draw Live Moving Packet Trajectory (Point A to B)
    if (isRunning && currentStage < nodes.length - 1) {
        const p1 = nodes[currentStage];
        const p2 = nodes[currentStage + 1];

        const px = (p1.x + (p2.x - p1.x) * progress) * w;
        const py = (p1.y + (p2.y - p1.y) * progress) * h;

        // Glowing Signal Packet
        ctx.shadowColor = stages[currentStage].color;
        ctx.shadowBlur = 15;
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0; // Reset
    }

    // 4. Draw Peak-Complexity Live Waveform at Bottom (Geometry shifted down)
    ctx.strokeStyle = stages[currentStage].color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x < w; x += 5) {
        let waveY = h * 0.85; // Signal line PUSHED DOWN to bottom
        if (currentStage === 2) {
            waveY += Math.sin(x * 0.1 + frameCount * 0.2) * 25 * (Math.random() - 0.5);
        } else {
            waveY += Math.sin(x * 0.02 + frameCount * 0.05) * 12;
        }
        if (x === 0) ctx.moveTo(x, waveY);
        else ctx.lineTo(x, waveY);
    }
    ctx.stroke();

    frameCount++;
    requestAnimationFrame(drawLoop);
}

// Termux Log Streamer Function
function addTermLog(text) {
    const p = document.createElement("p");
    p.innerText = text;
    termLog.appendChild(p);
    termLog.scrollTop = termLog.scrollHeight;
}

// Main 40-Second Automated Sequence Controller
startBtn.addEventListener("click", () => {
    if (isRunning) return;
    isRunning = true;
    currentStage = 0;
    termLog.innerHTML = "";

    function executeStage(index) {
        if (index >= stages.length) {
            isRunning = false;
            addTermLog("[FINISHED] Pipeline Execution Complete.");
            return;
        }

        currentStage = index;
        const stg = stages[index];

        // Update Math HUD
        document.getElementById("stageTitle").innerText = stg.title;
        document.getElementById("stageTitle").style.color = stg.color;
        document.getElementById("mathEquation").innerHTML = stg.eq;
        document.getElementById("liveMetrics").innerText = stg.metrics;
        MathJax.typesetPromise();

        // Stream Termux Logs
        stg.logs.forEach((log, i) => {
            setTimeout(() => addTermLog(log), i * 1500);
        });

        // Packet Travel Animation Duration (Exact 8 Seconds per stage, 40s total)
        let startTime = performance.now();
        let stageDuration = 8000; // 8 Seconds per stage

        function animatePacket(time) {
            let elapsed = time - startTime;
            progress = Math.min(elapsed / stageDuration, 1);

            if (elapsed < stageDuration) {
                requestAnimationFrame(animatePacket);
            } else {
                executeStage(index + 1);
            }
        }
        requestAnimationFrame(animatePacket);
    }

    executeStage(0);
});

// Start Canvas Draw
drawLoop();
            
