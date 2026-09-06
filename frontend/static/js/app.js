// Multi-Modal Fetal Risk Assessment - Clinician Workstation Controller

let currentPatient = null;
let currentBrainData = null;
let currentCardiacData = null;
let fhrChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    initSidebarTabs();
    initDropzones();
    loadPatients();
    loadAssessmentsHistory();
    loadEvaluationMetrics();
    initHeraHeroAndFeatures();
    animateWatermarkCounters();

    document.getElementById("btn-register-patient")?.addEventListener("click", registerPatient);
    document.getElementById("btn-quick-demo")?.addEventListener("click", loadDemoCase);
    document.getElementById("btn-run-analysis")?.addEventListener("click", executeMultiModalAnalysis);
    document.getElementById("select-patient")?.addEventListener("change", onPatientSelect);
});

// Hera Maternal Theme Interactivity & Smooth Scrolling
function initHeraHeroAndFeatures() {
    document.getElementById("btn-hero-quick-demo")?.addEventListener("click", loadDemoCase);
    document.getElementById("btn-hero-scroll-input")?.addEventListener("click", () => {
        document.querySelector(".section-card")?.scrollIntoView({ behavior: "smooth" });
    });

    document.getElementById("card-feature-fusion")?.addEventListener("click", () => {
        document.querySelector(".bottom-action-bar")?.scrollIntoView({ behavior: "smooth" });
    });
    document.getElementById("card-feature-brain")?.addEventListener("click", () => {
        document.querySelector(".modality-card-brain")?.scrollIntoView({ behavior: "smooth" });
    });
    document.getElementById("card-feature-cardiac")?.addEventListener("click", () => {
        document.querySelector(".modality-card-cardiac")?.scrollIntoView({ behavior: "smooth" });
    });
    document.getElementById("card-feature-explain")?.addEventListener("click", () => {
        document.querySelector(".bottom-action-bar")?.scrollIntoView({ behavior: "smooth" });
    });
}

// Animated Giant Watermark Stat Counters
function animateWatermarkCounters() {
    const cards = document.querySelectorAll(".watermark-metric-card");
    cards.forEach((card, idx) => {
        const target = parseFloat(card.getAttribute("data-target"));
        const suffix = card.getAttribute("data-suffix") || "";
        const valEl = card.querySelector(".watermark-metric-value");
        if (!valEl || isNaN(target)) return;

        const duration = 1200 + idx * 200;
        const startTime = performance.now();

        function updateCount(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = target * easeOut;

            if (target % 1 !== 0) {
                valEl.textContent = current.toFixed(1) + suffix;
            } else {
                valEl.textContent = Math.round(current) + suffix;
            }

            if (progress < 1) {
                requestAnimationFrame(updateCount);
            }
        }
        requestAnimationFrame(updateCount);
    });
}

// Sidebar Tab Navigation
function initSidebarTabs() {
    const buttons = document.querySelectorAll(".sidebar-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            buttons.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");
            
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.style.display = "block";
            }
            if (targetId === "tab-history") loadAssessmentsHistory();
            if (targetId === "tab-eval") loadEvaluationMetrics();
        });
    });
}

// Drag & Drop Upload Handlers
function initDropzones() {
    // Brain Scan
    const brainZone = document.getElementById("brain-dropzone");
    const brainInput = document.getElementById("brain-file-input");
    brainZone?.addEventListener("click", () => brainInput.click());
    brainInput?.addEventListener("change", (e) => {
        if (e.target.files.length > 0) uploadBrainScan(e.target.files[0]);
    });

    // Cardiac Signal
    const cardiacZone = document.getElementById("cardiac-dropzone");
    const cardiacInput = document.getElementById("cardiac-file-input");
    cardiacZone?.addEventListener("click", () => cardiacInput.click());
    cardiacInput?.addEventListener("change", (e) => {
        if (e.target.files.length > 0) uploadCardiacSignal(e.target.files[0]);
    });
}

// Load Registered Patients
async function loadPatients() {
    try {
        const res = await fetch("/api/patients");
        const patients = await res.json();
        const select = document.getElementById("select-patient");
        if (!select) return;

        select.innerHTML = '<option value="">-- Select Registered Patient --</option>';
        patients.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.id;
            opt.textContent = `${p.patient_id} - ${p.name} (GW: ${p.gestational_week}w)`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to load patients:", err);
    }
}

// Register New Patient
async function registerPatient() {
    const id = document.getElementById("input-pat-id").value.trim();
    const name = document.getElementById("input-pat-name").value.trim();
    const age = document.getElementById("input-pat-age").value;
    const gw = document.getElementById("input-pat-gw").value;

    if (!name || !age || !gw) {
        alert("Please enter patient Name, Age, and Gestational Week.");
        return;
    }

    try {
        const res = await fetch("/api/patient", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: id, name, age, gestational_week: gw })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Registration failed");

        await loadPatients();
        document.getElementById("select-patient").value = data.patient.id;
        currentPatient = data.patient;
        updateActivePatientDisplay();
    } catch (err) {
        alert(err.message);
    }
}

function onPatientSelect(e) {
    const pId = e.target.value;
    if (!pId) {
        currentPatient = null;
        updateActivePatientDisplay();
        return;
    }
    fetch("/api/patients")
        .then(r => r.json())
        .then(patients => {
            currentPatient = patients.find(p => p.id == pId);
            updateActivePatientDisplay();
        });
}

function updateActivePatientDisplay() {
    const nameEl = document.getElementById("active-patient-name-display");
    const subEl = document.getElementById("active-patient-sub-display");
    if (!nameEl || !subEl) return;

    if (currentPatient) {
        nameEl.textContent = `${currentPatient.patient_id} - ${currentPatient.name}`;
        nameEl.style.color = "var(--coral-main)";
        subEl.textContent = `Age: ${currentPatient.age} yrs | Gestational Week: ${currentPatient.gestational_week}w`;
        subEl.style.color = "var(--navy-muted)";
    } else {
        nameEl.textContent = "No Patient Selected";
        nameEl.style.color = "var(--navy-dark)";
        subEl.textContent = "Please register or select a patient";
        subEl.style.color = "var(--text-dim)";
    }
    checkReadiness();
}

// Upload & Process Brain Scan
async function uploadBrainScan(file) {
    const statusEl = document.getElementById("brain-status");
    statusEl.textContent = "Processing Brain Scan...";
    statusEl.style.color = "#f59e0b";

    document.querySelectorAll(".image-box").forEach(b => b.classList.add("scanning"));

    const formData = new FormData();
    formData.append("file", file);
    if (currentPatient) formData.append("patient_id", currentPatient.id);

    try {
        const res = await fetch("/api/brain/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Brain processing failed");

        currentBrainData = data;
        displayBrainResults(data);
        statusEl.textContent = "Brain Features Ready";
        statusEl.style.color = "#10b981";
        checkReadiness();
    } catch (err) {
        statusEl.textContent = "Upload Failed";
        statusEl.style.color = "#ef4444";
        alert(`Brain scan error: ${err.message}`);
    } finally {
        document.querySelectorAll(".image-box").forEach(b => b.classList.remove("scanning"));
    }
}

function displayBrainResults(data) {
    document.getElementById("brain-result-viewer").style.display = "block";
    document.getElementById("img-brain-preprocessed").src = data.preprocessed_url;
    document.getElementById("img-brain-mapped").src = data.atlas_mapped_url;
    document.getElementById("img-brain-deform").src = data.deformation_field_url;

    const metricsDiv = document.getElementById("brain-metrics-body");
    const m = data.atlas_metrics;
    const q = data.preprocessing_metrics;
    metricsDiv.innerHTML = `
        <tr><td>Algorithm</td><td><strong style="color:#9333ea;">${m.algorithm}</strong></td></tr>
        <tr><td>Normalized Cross-Correlation (NCC)</td><td><strong>${m.normalized_cross_correlation}</strong></td></tr>
        <tr><td>Mean Displacement</td><td>${m.mean_displacement_pixels} px (Max: ${m.max_displacement_pixels} px)</td></tr>
        <tr><td>Atlas Mapping Status</td><td><span style="color:#10b981;font-weight:700;">&#10003; Aligned to Reference Atlas</span></td></tr>
    `;

    const featDiv = document.getElementById("brain-feature-vector");
    featDiv.textContent = `F_brain [dim=${data.feature_summary.dimension}]: [${data.feature_summary.sample_vector.join(", ")} ...]`;
}

// Upload & Process Cardiac Signal
async function uploadCardiacSignal(file) {
    const statusEl = document.getElementById("cardiac-status");
    statusEl.textContent = "Conditioning FHR...";
    statusEl.style.color = "#f59e0b";

    const formData = new FormData();
    formData.append("file", file);
    if (currentPatient) formData.append("patient_id", currentPatient.id);

    try {
        const res = await fetch("/api/cardiac/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Cardiac processing failed");

        currentCardiacData = data;
        displayCardiacResults(data);
        statusEl.textContent = "Cardiac Features Ready";
        statusEl.style.color = "#10b981";
        checkReadiness();
    } catch (err) {
        statusEl.textContent = "Upload Failed";
        statusEl.style.color = "#ef4444";
        alert(`Cardiac signal error: ${err.message}`);
    }
}

function displayCardiacResults(data) {
    document.getElementById("cardiac-result-viewer").style.display = "block";
    renderFhrChart(data.chart_data);

    const statsDiv = document.getElementById("cardiac-metrics-body");
    const s = data.stats;
    statsDiv.innerHTML = `
        <tr><td>Baseline FHR</td><td><strong style="color:var(--coral-main);">${s.baseline_fhr} bpm</strong> (Normal: 110-160)</td></tr>
        <tr><td>Short-Term Variability (STV)</td><td>${s.short_term_variability} bpm</td></tr>
        <tr><td>Long-Term Variability (LTV)</td><td>${s.long_term_variability} bpm</td></tr>
        <tr><td>Accelerations / Decelerations</td><td>${s.accelerations_count} acc / ${s.decelerations_count} dec</td></tr>
    `;

    const featDiv = document.getElementById("cardiac-feature-vector");
    featDiv.textContent = `F_cardiac [dim=${data.feature_summary.dimension}]: [${data.feature_summary.sample_vector.join(", ")} ...]`;
}

// Render Waveform Chart
function renderFhrChart(chartData) {
    const ctx = document.getElementById("chart-fhr")?.getContext("2d");
    if (!ctx) return;

    if (fhrChartInstance) fhrChartInstance.destroy();

    fhrChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: chartData.time,
            datasets: [
                {
                    label: "Conditioned FHR (bpm)",
                    data: chartData.processed,
                    borderColor: "#e06d64",
                    backgroundColor: "rgba(224, 109, 100, 0.12)",
                    borderWidth: 2.2,
                    pointRadius: 0,
                    tension: 0.2,
                    fill: true,
                },
                {
                    label: "Baseline FHR",
                    data: Array(chartData.time.length).fill(chartData.baseline),
                    borderColor: "#1e3a5f",
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 1000, easing: "easeOutQuart" },
            scales: {
                x: {
                    title: { display: true, text: "Time (seconds)", color: "#556885" },
                    grid: { color: "rgba(224, 109, 100, 0.1)" },
                    ticks: { color: "#556885", maxTicksLimit: 10 }
                },
                y: {
                    min: 50,
                    max: 200,
                    title: { display: true, text: "BPM", color: "#556885" },
                    grid: { color: "rgba(224, 109, 100, 0.1)" },
                    ticks: { color: "#556885" }
                }
            },
            plugins: {
                legend: { labels: { color: "#132742", font: { weight: "600" } } }
            }
        }
    });
}

function checkReadiness() {
    const btn = document.getElementById("btn-run-analysis");
    const titleEl = document.getElementById("action-bar-status-title");
    const descEl = document.getElementById("action-bar-status-desc");
    if (!btn) return;

    if (currentBrainData && currentCardiacData && currentPatient) {
        btn.classList.add("active-ready");
        titleEl.textContent = "Pipeline Fully Armed";
        titleEl.style.color = "var(--coral-main)";
        descEl.textContent = `Ready to perform multi-modal feature fusion for ${currentPatient.name} (GW: ${currentPatient.gestational_week}w).`;
    } else {
        btn.classList.remove("active-ready");
        titleEl.textContent = "Ready for Analysis";
        titleEl.style.color = "var(--navy-dark)";
        descEl.textContent = "Upload both modalities and register patient to run the multi-modal fetal risk assessment pipeline.";
    }
}

// Multi-Stage Diagnostic Animation Sequence
async function runDiagnosticAnimationSequence() {
    const modal = document.getElementById("diagnostic-modal");
    const progressFill = document.getElementById("diag-progress-fill");
    const statusText = document.getElementById("diag-status-text");
    modal.classList.add("active");

    const steps = [
        { id: "diag-step-1", text: "Isolating Cranial ROI & Standardizing Contrast...", progress: 20 },
        { id: "diag-step-2", text: "Solving Elastic B-Spline Deformable Atlas Registration...", progress: 40 },
        { id: "diag-step-3", text: "Brain CNN Extracting 128-D Spatial Representation...", progress: 60 },
        { id: "diag-step-4", text: "Conditioning CTG Waveform & Computing LSTM Attention...", progress: 80 },
        { id: "diag-step-5", text: "Executing Feature-Level Gated Attention Fusion...", progress: 95 },
        { id: "diag-step-6", text: "Multi-Layer Risk Assessment Synthesized!", progress: 100 },
    ];

    for (let i = 1; i <= 6; i++) {
        const el = document.getElementById(`diag-step-${i}`);
        el.className = "diagnostic-step";
        el.querySelector(".step-icon").textContent = i;
    }

    for (let i = 0; i < steps.length; i++) {
        const s = steps[i];
        const stepEl = document.getElementById(s.id);
        stepEl.classList.add("active");
        statusText.textContent = s.text;
        progressFill.style.width = `${s.progress}%`;

        await new Promise(r => setTimeout(r, 400));
        stepEl.classList.remove("active");
        stepEl.classList.add("done");
        stepEl.querySelector(".step-icon").innerHTML = "&#10003;";
    }

    await new Promise(r => setTimeout(r, 250));
    modal.classList.remove("active");
}

// Execute Multi-Modal Analysis
async function executeMultiModalAnalysis() {
    if (!currentBrainData || !currentCardiacData) {
        alert("Please upload and analyze both Brain Scan and Cardiac Signal files first.");
        return;
    }

    const patId = currentPatient ? currentPatient.id : (currentBrainData.patient_id || currentCardiacData.patient_id);
    if (!patId) {
        alert("Please register or select a patient before assessment.");
        return;
    }

    const btn = document.getElementById("btn-run-analysis");
    btn.disabled = true;

    try {
        const [apiResponse] = await Promise.all([
            fetch("/api/predict/multimodal", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    patient_id: patId,
                    brain_image_id: currentBrainData.db_id,
                    cardiac_signal_id: currentCardiacData.db_id,
                })
            }).then(r => r.json()),
            runDiagnosticAnimationSequence()
        ]);

        if (apiResponse.error) throw new Error(apiResponse.error);

        displayAssessmentResult(apiResponse);
    } catch (err) {
        alert("Assessment error: " + err.message);
    } finally {
        btn.disabled = false;
    }
}

// Animated Display of Assessment Results
function displayAssessmentResult(data) {
    const card = document.getElementById("assessment-result-card");
    card.style.display = "block";

    const circle = document.getElementById("radial-gauge-circle");
    const scoreVal = document.getElementById("risk-score-value");
    const targetPercent = data.risk_percent;
    const circumference = 440;

    let strokeColor = "#10b981"; // Normal
    if (data.prediction.toLowerCase() === "suspect") strokeColor = "#f59e0b";
    if (data.prediction.toLowerCase() === "pathological") strokeColor = "#ef4444";

    circle.style.stroke = strokeColor;
    const offset = circumference - (circumference * (targetPercent / 100));
    circle.style.strokeDashoffset = offset;

    let currentScore = 0;
    const duration = 1200;
    const startTime = performance.now();

    function updateScore(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = (targetPercent * easeOut).toFixed(1);
        scoreVal.textContent = `${current}%`;
        if (progress < 1) requestAnimationFrame(updateScore);
    }
    requestAnimationFrame(updateScore);

    const badge = document.getElementById("prediction-badge");
    badge.textContent = data.prediction;
    badge.className = "risk-badge";
    if (data.prediction.toLowerCase() === "normal") badge.classList.add("risk-normal");
    else if (data.prediction.toLowerCase() === "suspect") badge.classList.add("risk-suspect");
    else badge.classList.add("risk-pathological");

    document.getElementById("confidence-value").textContent = `Model Confidence: ${data.confidence}%`;

    const bInf = data.modality_influence.brain_imaging;
    const cInf = data.modality_influence.cardiac_signal;
    setTimeout(() => {
        document.getElementById("bar-brain-fill").style.width = `${bInf}%`;
        document.getElementById("bar-brain-val").textContent = `${bInf}%`;
        document.getElementById("bar-cardiac-fill").style.width = `${cInf}%`;
        document.getElementById("bar-cardiac-val").textContent = `${cInf}%`;
    }, 200);

    const probList = document.getElementById("class-prob-list");
    probList.innerHTML = "";
    for (const [cls, prob] of Object.entries(data.probabilities)) {
        const chip = document.createElement("div");
        chip.className = "prob-chip";
        chip.innerHTML = `<span>${cls}</span><strong>${(prob * 100).toFixed(1)}%</strong>`;
        probList.appendChild(chip);
    }

    card.scrollIntoView({ behavior: "smooth" });
}

// 1-Click Load Pre-Packaged Demo Case
async function loadDemoCase() {
    try {
        document.querySelectorAll(".image-box").forEach(b => b.classList.add("scanning"));

        const [demoData] = await Promise.all([
            fetch("/api/demo/load", { method: "POST" }).then(r => r.json()),
            runDiagnosticAnimationSequence()
        ]);

        if (demoData.error) throw new Error(demoData.error);

        await loadPatients();
        currentPatient = demoData.patient;
        document.getElementById("select-patient").value = demoData.patient.id;
        updateActivePatientDisplay();

        currentBrainData = demoData.brain;
        displayBrainResults(demoData.brain);
        document.getElementById("brain-status").textContent = "Brain Features Ready";
        document.getElementById("brain-status").style.color = "#10b981";

        currentCardiacData = demoData.cardiac;
        displayCardiacResults(demoData.cardiac);
        document.getElementById("cardiac-status").textContent = "Cardiac Features Ready";
        document.getElementById("cardiac-status").style.color = "#10b981";

        checkReadiness();
        displayAssessmentResult(demoData.assessment);

    } catch (err) {
        alert("Demo loading failed: " + err.message);
    } finally {
        document.querySelectorAll(".image-box").forEach(b => b.classList.remove("scanning"));
    }
}

// Load Assessment History
async function loadAssessmentsHistory() {
    const tbody = document.getElementById("history-table-body");
    if (!tbody) return;
    try {
        const res = await fetch("/api/assessments");
        const list = await res.json();
        if (list.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:2.5rem;">No risk assessments recorded yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = list.map(a => `
            <tr>
                <td><strong>${a.assessment_id}</strong></td>
                <td>${a.patient_code || "N/A"} (${a.patient_name || "N/A"})</td>
                <td>${a.gestational_week ? a.gestational_week + " wks" : "N/A"}</td>
                <td><strong>${(a.risk_score * 100).toFixed(1)}%</strong></td>
                <td><span class="risk-badge risk-${a.prediction.toLowerCase()}">${a.prediction}</span></td>
                <td>${(a.confidence * 100).toFixed(1)}%</td>
                <td>${a.date}</td>
            </tr>
        `).join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="color:#ef4444;">Failed to load history: ${err.message}</td></tr>`;
    }
}

// Load Evaluation Metrics
async function loadEvaluationMetrics() {
    const container = document.getElementById("eval-content-container");
    if (!container) return;

    try {
        const res = await fetch("/api/evaluation");
        const data = await res.json();

        if (data.status === "uncalibrated") {
            container.innerHTML = `
                <div class="section-card" style="text-align:center;padding:2.5rem;">
                    <h3 style="color:#f59e0b;margin-bottom:0.75rem;">Model Evaluation In Progress</h3>
                    <p style="color:#cbd5e1;max-width:600px;margin:0 auto;">${data.message}</p>
                </div>
            `;
            return;
        }

        const cmRows = data.confusion_matrix.map((row, i) => `
            <tr>
                <th>True ${data.classes[i]}</th>
                ${row.map(val => `<td><strong style="color:var(--coral-main);">${val}</strong></td>`).join("")}
            </tr>
        `).join("");

        container.innerHTML = `
            <div class="section-card">
                <div class="card-header-flex">
                    <div class="card-main-title">Authentic Multi-Modal Test Split Metrics</div>
                    <span style="font-size:0.8rem;color:#10b981;font-weight:700;">Evaluated on Held-Out Test Set (N=${data.sample_count})</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:1.2rem;margin-bottom:1.75rem;text-align:center;">
                    <div style="padding:1.4rem;background:#ffffff;border:1px solid rgba(224,109,100,0.25);border-radius:1rem;box-shadow:var(--card-shadow);">
                        <span style="font-size:0.75rem;color:var(--navy-muted);font-weight:700;text-transform:uppercase;">Accuracy</span>
                        <h2 style="color:var(--coral-main);margin-top:0.35rem;font-size:2rem;font-weight:800;">${(data.accuracy * 100).toFixed(2)}%</h2>
                    </div>
                    <div style="padding:1.4rem;background:#ffffff;border:1px solid rgba(16,185,129,0.25);border-radius:1rem;box-shadow:var(--card-shadow);">
                        <span style="font-size:0.75rem;color:var(--navy-muted);font-weight:700;text-transform:uppercase;">Precision (Macro)</span>
                        <h2 style="color:#10b981;margin-top:0.35rem;font-size:2rem;font-weight:800;">${(data.precision_macro * 100).toFixed(2)}%</h2>
                    </div>
                    <div style="padding:1.4rem;background:#ffffff;border:1px solid rgba(245,158,11,0.25);border-radius:1rem;box-shadow:var(--card-shadow);">
                        <span style="font-size:0.75rem;color:var(--navy-muted);font-weight:700;text-transform:uppercase;">Recall (Macro)</span>
                        <h2 style="color:#f59e0b;margin-top:0.35rem;font-size:2rem;font-weight:800;">${(data.recall_macro * 100).toFixed(2)}%</h2>
                    </div>
                    <div style="padding:1.4rem;background:#ffffff;border:1px solid rgba(147,51,234,0.25);border-radius:1rem;box-shadow:var(--card-shadow);">
                        <span style="font-size:0.75rem;color:var(--navy-muted);font-weight:700;text-transform:uppercase;">F1-Score (Macro)</span>
                        <h2 style="color:#9333ea;margin-top:0.35rem;font-size:2rem;font-weight:800;">${(data.f1_macro * 100).toFixed(2)}%</h2>
                    </div>
                </div>

                <h4 style="margin-bottom:0.75rem;color:var(--navy-dark);font-weight:800;">Confusion Matrix</h4>
                <table class="metrics-table" style="max-width:520px;margin-bottom:1.5rem;">
                    <thead>
                        <tr>
                            <th></th>
                            ${data.classes.map(c => `<th>Pred ${c}</th>`).join("")}
                        </tr>
                    </thead>
                    <tbody>${cmRows}</tbody>
                </table>
                <p style="font-size:0.8rem;color:var(--navy-muted);">${data.note}</p>
            </div>
        `;
    } catch (err) {
        container.innerHTML = `<p style="color:#ef4444;">Failed to load evaluation: ${err.message}</p>`;
    }
}
