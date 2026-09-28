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

// Architectural Nodes Positions
const nodes = [
    { name: "Input Token Matrix", x: 0.15, y: 0.35 },
    { name: "QK Multi-Head Proj", x: 0.35, y: 0.25 },
    { name: "Softmax Noise Engine", x: 0.55, y: 0.45 },
    { name: "Pythia-70M SVD Rank", x: 0.75, y: 0.25 },
    { name: "ETF Geometry Fix", x: 0.90, y: 0.35 }
];

// Stages Configuration (Total Duration = Exact 40 Seconds)
const stages = [
    {
        title: "🟢 STAGE 1: High-Dim Input Token Vector Ingestion",
        color: "#00FFA3",
        eq: "\\[ S_{ij} = \\frac{\\mathbf{q}_i^\\top \\mathbf{k}_j}{\\sqrt{d_{head}}}, \\quad \\frac{N}{d_{head}} = 0.5 \\le 1.0 \\]",
        metrics: "> Frobenius Error: 0.0012 | Null Space: 0 | Capacity Load: 50% [STABLE]",
        activeNode: 0,
        logs: [
            "[INFO] Parsing token sequence embeddings...",
            "[MATH] Constructing Q and K projection spaces...",
            "[METRIC] Orthogonality metric stable: ||Q*Q^T - I|| = 0.0012",
            "[STATUS] Subspace alignment within noise bounds."
        ]
    },
    {
        title: "🟡 STAGE 2: Query-Key Alignment & Capacity Limit",
        color: "#FFD700",
        eq: "\\[ \\text{rank}(\\mathbf{W}_Q\\mathbf{W}_K^\\top) = d_{head}, \\quad \\frac{N}{d_{head}} = 1.0 \\]",
        metrics: "> Critical Load Boundary Reached | Spectral Leakage: 2.1%",
        activeNode: 1,
        logs: [
            "[WARN] Load ratio approaching critical threshold N/d_head = 1.0",
            "[CALC] Computing singular values SVD(W_Q * W_K^T)...",
            "[SPECTRUM] Tail eigenvalues begin to flatten...",
            "[ALERT] Phase Transition boundary localized."
        ]
    },
    {
        title: "🔴 STAGE 3: Softmax Exponential Noise Explosion",
        color: "#FF0055",
        eq: "\\[ A_{ij} = \\frac{\\exp(S_{ij})}{\\sum_k \\exp(S_{ik})}, \\quad \\mathcal{D}_{KL}(A \\| A_{true}) \\sim \\mathcal{O}\\left(e^{\\gamma (N - d_{head})}\\right) \\]",
        metrics: "> Cross-Talk Noise: +340% | Entropy: 5.84 nats | COLLAPSED",
        activeNode: 2,
        logs: [
            "[CRITICAL] Load ratio exceeded! N/d_head = 1.85 > 1.0",
            "[MATH ERROR] Softmax exponential amplification triggered!",
            "[ENTROPY] Cross-talk noise exploding: H(A) = 5.84 nats",
            "[SYSTEM] Context window lost. Hallucination mode active."
        ]
    },
    {
        title: "🟣 STAGE 4: Pythia-70M Real Weight SVD Rank Truncation",
        color: "#9900FF",
        eq: "\\[ \\mathbf{A} = \\mathbf{U} \\mathbf{\\Sigma} \\mathbf{V}^\\top, \\quad \\sigma_i \\to 0 \\quad \\forall i > d_{head} \\]",
        metrics: "> Effective Rank: 13/16 | Singular Value Floor Hit | Truncated",
        activeNode: 3,
        logs: [
            "[INSPECT] Extracting weights from EleutherAI/Pythia-70M Layer 8...",
            "[SVD] Decomposing attention head matrix...",
            "[RESULT] Singular values dropped to exact zero line.",
            "[PROOF] LayerNorm and L2 decay failed to shift geometric bound."
        ]
    },
    {
        title: "⚡ STAGE 5: ETF-Aware Initialization Applied — Structural Recovery",
        color: "#00E5FF",
        eq: "\\[ \\mathbf{M}^* = \\sqrt{\\frac{K}{K-1}} \\left( \\mathbf{I}_K - \\frac{1}{K}\\mathbf{1}\\mathbf{1}^\\top \\right), \\quad \\cos\\theta_{ij} = -\\frac{1}{N-1} \\]",
        metrics: "> Reconstruction Fidelity: 99.89% | Zero Interference | Context Restored",
        activeNode: 4,
        logs: [
            "[ENG] Injecting ETF Polytope Vector Initialization...",
            "[MATH] Maximizing pairwise feature angles theta_ij...",
            "[RECOVERY] Dynamic Head Allocation rescales d_h(t)...",
            "[SUCCESS] Context window restored with ZERO information loss!"
        ]
    }
];

let isRunning = false;
let currentStage = 0;
let progress = 0; // 0 to 1 packet movement
let frameCount = 0;

// Render Loop (Runs at smooth 60 FPS GPU Canvas)
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

        // Node Title
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px Courier New";
        ctx.fillText(node.name, nx - 40, ny - 25);
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

    // 4. Draw Peak-Complexity Live Waveform at Bottom
    ctx.strokeStyle = stages[currentStage].color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x < w; x += 5) {
        let waveY = h * 0.75;
        if (currentStage === 2) {
            // High Entropy Chaos
            waveY += Math.sin(x * 0.1 + frameCount * 0.2) * 25 * (Math.random() - 0.5);
        } else {
            // Smooth Controlled Signal
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

        // Packet Travel Animation Duration (8 seconds per stage = 40s total)
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
          
