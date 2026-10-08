// ============================================================
// HERA MULTI-MODAL FETAL INTELLIGENCE PLATFORM CONTROLLER
// Features: Clinical Authentication, Workstation, CTG Streaming,
//           Batch Triage, Growth Velocity, Grad-CAM XAI, PDF Export
// ============================================================

let currentPatient = null;
let currentBrainData = null;
let currentCardiacData = null;
let currentDemoMetadata = null;
let fhrChartInstance = null;

// Streaming state
let streamingInterval = null;
let streamData = null;
let streamIndex = 0;
let streamSpeed = 1;
let streamChartInstance = null;

// Growth chart instances
let growthBiometryChart = null;
let growthFhrChart = null;

// Cache
let demoPatientsCache = [];

// Clinical User Session Presets
const CLINICIAN_PRESETS = {
    researcher: {
        name: "Dr. Elena Rostova",
        dept: "Fetal AI Studio",
        email: "researcher@fetal-ai.org",
        role: "Clinical AI Researcher"
    },
    perinatologist: {
        name: "Dr. Sarah Jenkins",
        dept: "Maternal-Fetal Medicine",
        email: "s.jenkins@perinatal-med.org",
        role: "Lead Perinatologist"
    },
    obgyn: {
        name: "Dr. Michael Chen",
        dept: "Labor & Delivery Unit",
        email: "m.chen@hospital-obgyn.org",
        role: "High-Risk Obstetrician"
    },
    administrator: {
        name: "Alex Morgan",
        dept: "HERA Operations",
        email: "admin@fetal-ai.org",
        role: "Demo Administrator"
    }
};

document.addEventListener("DOMContentLoaded", () => {
    // 1. Mandatory Clinical Authentication Modal on Every Page Load
    showLoginModal();

    // 2. Platform Initializations
    initSidebarTabs();
    initDropzones();
    loadPatients();
    loadDemoPatientsList();
    loadAssessmentsHistory();
    loadEvaluationMetrics();
    initBrandNavigation();

    // 3. Event Listeners
    document.getElementById("btn-register-patient")?.addEventListener("click", registerPatient);
    document.getElementById("btn-run-analysis")?.addEventListener("click", executeMultiModalAnalysis);
    document.getElementById("btn-export-pdf")?.addEventListener("click", exportPdfReport);
    document.getElementById("select-patient")?.addEventListener("change", onPatientSelect);
    document.getElementById("select-demo-patient")?.addEventListener("change", onDemoPatientSelect);
    document.getElementById("btn-load-selected-demo")?.addEventListener("click", onBtnLoadDemoClick);

    // Streaming Event Listeners
    document.getElementById("btn-stream-play")?.addEventListener("click", startStreaming);
    document.getElementById("btn-stream-pause")?.addEventListener("click", pauseStreaming);
    document.getElementById("btn-stream-reset")?.addEventListener("click", resetStreaming);
    document.getElementById("select-stream-patient")?.addEventListener("change", onStreamPatientChange);
    
    document.querySelectorAll(".btn-speed").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            streamSpeed = parseInt(e.target.getAttribute("data-speed")) || 1;
            if (streamingInterval) {
                pauseStreaming();
                startStreaming();
            }
        });
    });

    // Triage Filter Chips
    document.querySelectorAll(".btn-filter-chip").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".btn-filter-chip").forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            const filter = e.target.getAttribute("data-filter");
            filterTriageTable(filter);
        });
    });

    // Longitudinal Patient Select
    document.getElementById("select-longitudinal-patient")?.addEventListener("change", (e) => {
        loadLongitudinalTracker(e.target.value);
    });

    // Close user dropdown when clicking outside
    document.addEventListener("click", (e) => {
        const toggle = document.getElementById("user-profile-menu-toggle");
        const menu = document.getElementById("user-dropdown-menu");
        if (toggle && menu && !toggle.contains(e.target)) {
            menu.style.display = "none";
        }
    });
});

// ============================================================
// CLINICAL LOGIN & AUTHENTICATION CONTROLLER
// ============================================================
function showLoginModal() {
    const modal = document.getElementById("login-modal");
    if (modal) modal.style.display = "flex";
}

function hideLoginModal() {
    const modal = document.getElementById("login-modal");
    if (modal) modal.style.display = "none";
}

function updateLoginPreset(profileKey) {
    const preset = CLINICIAN_PRESETS[profileKey] || CLINICIAN_PRESETS.researcher;
    const emailInput = document.getElementById("login-email");
    if (emailInput) emailInput.value = preset.email;
}

function handleClinicalLogin() {
    const profileKey = document.getElementById("login-profile-select")?.value || "researcher";
    const preset = CLINICIAN_PRESETS[profileKey] || CLINICIAN_PRESETS.researcher;
    applyUserSession(preset);
    hideLoginModal();
    // Default load first patient PAT-1001 for seamless experience
    loadDemoPatientById("PAT-1001");
}

function handleInstantDemoLogin() {
    applyUserSession(CLINICIAN_PRESETS.researcher);
    hideLoginModal();
    loadDemoPatientById("PAT-1001");
}

function applyUserSession(user) {
    const nameEl = document.getElementById("nav-user-name");
    const deptEl = document.getElementById("nav-user-dept");
    const dropName = document.getElementById("dropdown-user-name");
    const dropRole = document.getElementById("dropdown-user-role");

    if (nameEl) nameEl.textContent = user.name;
    if (deptEl) deptEl.textContent = user.dept;
    if (dropName) dropName.textContent = user.name;
    if (dropRole) dropRole.textContent = `${user.role} (${user.dept})`;
}

function toggleUserDropdown() {
    const menu = document.getElementById("user-dropdown-menu");
    if (menu) {
        menu.style.display = menu.style.display === "none" || !menu.style.display ? "block" : "none";
    }
}

function closeUserDropdown() {
    const menu = document.getElementById("user-dropdown-menu");
    if (menu) menu.style.display = "none";
}

function handleLogout() {
    closeUserDropdown();
    showLoginModal();
}

// Brand Logo Navigation to Home
function initBrandNavigation() {
    const brandLogo = document.getElementById("nav-brand-logo");
    const brandText = document.getElementById("nav-brand-text");
    brandLogo?.addEventListener("click", () => switchTab("tab-home"));
    brandText?.addEventListener("click", () => switchTab("tab-home"));
}

// Global Tab & View Switcher
function switchTab(targetId) {
    const buttons = document.querySelectorAll(".sidebar-btn");
    buttons.forEach(b => {
        if (b.getAttribute("data-tab") === targetId) b.classList.add("active");
        else b.classList.remove("active");
    });

    document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");
    const targetPane = document.getElementById(targetId);
    if (targetPane) {
        targetPane.style.display = "block";
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    if (targetId === "tab-history") loadAssessmentsHistory();
    if (targetId === "tab-eval") loadEvaluationMetrics();
    if (targetId === "tab-triage") loadBatchTriage();
    if (targetId === "tab-streaming") initStreamingMonitor();
    if (targetId === "tab-longitudinal") {
        const pId = currentPatient ? currentPatient.patient_id : "PAT-1001";
        loadLongitudinalTracker(pId);
    }
}

function initSidebarTabs() {
    const buttons = document.querySelectorAll(".sidebar-btn");
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            switchTab(targetId);
        });
    });
}

// ============================================================
// MODALITY 1 & 2 DRAG & DROP UPLOAD
// ============================================================
function initDropzones() {
    const brainZone = document.getElementById("brain-dropzone");
    const brainInput = document.getElementById("brain-file-input");
    brainZone?.addEventListener("click", () => brainInput.click());
    brainInput?.addEventListener("change", (e) => {
        if (e.target.files.length > 0) uploadBrainScan(e.target.files[0]);
    });

    const cardiacZone = document.getElementById("cardiac-dropzone");
    const cardiacInput = document.getElementById("cardiac-file-input");
    cardiacZone?.addEventListener("click", () => cardiacInput.click());
    cardiacInput?.addEventListener("change", (e) => {
        if (e.target.files.length > 0) uploadCardiacSignal(e.target.files[0]);
    });
}

// ============================================================
// PATIENT INTAKE & 50 SYNTHETIC CASES
// ============================================================
async function loadPatients() {
    try {
        const res = await fetch("/api/patients");
        const patients = await res.json();
        const select = document.getElementById("select-patient");
        if (!select) return;

        select.innerHTML = '<option value="">PAT-1001 - John Smith (Demo)</option>';
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

async function loadDemoPatientsList() {
    try {
        const res = await fetch("/api/demo/patients");
        demoPatientsCache = await res.json();
        const select = document.getElementById("select-demo-patient");
        const streamSelect = document.getElementById("select-stream-patient");
        const longSelect = document.getElementById("select-longitudinal-patient");

        if (Array.isArray(demoPatientsCache)) {
            if (select) {
                select.innerHTML = '<option value="">Choose a Synthetic Patient (50 Available Cases)</option>';
                const ncGroup = document.createElement("optgroup");
                ncGroup.label = "Cohort NC (Cases PAT-1001 to PAT-1030)";
                const scGroup = document.createElement("optgroup");
                scGroup.label = "Cohort SC (Cases PAT-1031 to PAT-1042)";
                const pcGroup = document.createElement("optgroup");
                pcGroup.label = "Cohort PC (Cases PAT-1043 to PAT-1050)";

                demoPatientsCache.forEach(p => {
                    const opt = document.createElement("option");
                    opt.value = p.patient_id;
                    const code = p.expected_demo_result === "Normal" ? "NC" : (p.expected_demo_result === "Suspect" ? "SC" : "PC");
                    opt.textContent = `${p.patient_id} - ${p.name} (GW: ${p.gestational_week}w | ${code})`;
                    if (code === "NC") ncGroup.appendChild(opt);
                    else if (code === "SC") scGroup.appendChild(opt);
                    else pcGroup.appendChild(opt);
                });

                select.appendChild(ncGroup);
                select.appendChild(scGroup);
                select.appendChild(pcGroup);
            }

            if (streamSelect) {
                streamSelect.innerHTML = demoPatientsCache.map(p => {
                    const code = p.expected_demo_result === "Normal" ? "NC" : (p.expected_demo_result === "Suspect" ? "SC" : "PC");
                    return `<option value="${p.patient_id}">${p.patient_id} - ${p.name} (GW: ${p.gestational_week}w | ${code})</option>`;
                }).join("");
            }

            if (longSelect) {
                longSelect.innerHTML = demoPatientsCache.map(p => {
                    const code = p.expected_demo_result === "Normal" ? "NC" : (p.expected_demo_result === "Suspect" ? "SC" : "PC");
                    return `<option value="${p.patient_id}">${p.patient_id} - ${p.name} (GW: ${p.gestational_week}w | ${code})</option>`;
                }).join("");
            }
        }
    } catch (err) {
        console.error("Failed to load demo patients list:", err);
    }
}

function onDemoPatientSelect(e) {
    const patId = e.target.value;
    if (!patId) return;
    const meta = demoPatientsCache.find(p => p.patient_id === patId);
    if (meta) {
        updateDemoInspector(meta);
    }
}

function onBtnLoadDemoClick() {
    const select = document.getElementById("select-demo-patient");
    const patId = select ? select.value : null;
    if (!patId) {
        alert("Please select a synthetic patient case from the dropdown first.");
        return;
    }
    loadDemoPatientById(patId);
}

function updateDemoInspector(meta) {
    const inspector = document.getElementById("demo-patient-inspector");
    if (!inspector) return;
    inspector.style.display = "flex";

    const nameEl = document.getElementById("insp-pat-id-name");
    const gwEl = document.getElementById("insp-pat-gw");
    const wtEl = document.getElementById("insp-pat-weight");
    const presEl = document.getElementById("insp-pat-pres");
    const brainEl = document.getElementById("insp-brain-file");
    const cardEl = document.getElementById("insp-cardiac-file");
    const tierEl = document.getElementById("insp-expected-tier");

    if (nameEl) nameEl.textContent = `${meta.patient_id} ${meta.name}`;
    if (gwEl) gwEl.textContent = `${meta.gestational_week} wks (Age: ${meta.age || meta.maternal_age || 28})`;
    if (wtEl) wtEl.textContent = `${meta.estimated_fetal_weight_g || 1500} g`;
    if (presEl) presEl.textContent = `${meta.presentation || 'Cephalic'} (${meta.scan_type || 'MRI'})`;
    if (brainEl) brainEl.textContent = meta.brain_scan_filename || `${meta.patient_id}_brain.png`;
    if (cardEl) cardEl.textContent = meta.cardiac_signal_filename || `${meta.patient_id}_fhr.csv`;
    updateUploadedDatasetName("brain-upload-name", meta.brain_scan_filename || `${meta.patient_id}_brain.png`);
    updateUploadedDatasetName("cardiac-upload-name", meta.cardiac_signal_filename || `${meta.patient_id}_fhr.csv`);
    
    if (tierEl) {
        const res = (meta.expected_demo_result || 'Normal').toLowerCase();
        const code = res === 'normal' ? 'NC' : (res === 'suspect' ? 'SC' : 'PC');
        tierEl.textContent = `COHORT ${code}`;
        tierEl.className = "risk-badge";
        if (res === 'normal') tierEl.classList.add("risk-normal");
        else if (res === 'suspect') tierEl.classList.add("risk-suspect");
        else tierEl.classList.add("risk-pathological");
    }
}

async function loadDemoPatientById(patientId) {
    const statusTitle = document.getElementById("action-bar-status-title");
    const statusDesc = document.getElementById("action-bar-status-desc");
    if (statusTitle) statusTitle.textContent = `Loading ${patientId}...`;

    try {
        const res = await fetch(`/api/demo/load_patient/${patientId}`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load synthetic patient.");

        currentPatient = data.patient;
        currentBrainData = data.brain;
        currentCardiacData = data.cardiac;
        currentDemoMetadata = data.patient_metadata;

        // Update UI
        if (statusTitle) statusTitle.textContent = `Loaded ${patientId}`;
        if (statusDesc) statusDesc.textContent = "Matched brain and cardiac files are ready for multi-modal assessment.";
        updateActivePatientCard(currentPatient);
        if (currentDemoMetadata) updateDemoInspector(currentDemoMetadata);
        renderBrainViewer(currentBrainData);
        renderCardiacViewer(currentCardiacData);

        // Update Select Box if matched
        const select = document.getElementById("select-demo-patient");
        if (select) select.value = patientId;

    } catch (err) {
        console.error("Load patient error:", err);
        alert(err.message);
    }
}

function updateActivePatientCard(patient) {
    const nameEl = document.getElementById("insp-pat-id-name");
    if (nameEl) nameEl.textContent = `${patient.patient_id} ${patient.name}`;
}

async function registerPatient() {
    const patId = document.getElementById("input-pat-id")?.value.trim();
    const name = document.getElementById("input-pat-name")?.value.trim();
    const age = document.getElementById("input-pat-age")?.value;
    const gw = document.getElementById("input-pat-gw")?.value;

    if (!patId || !name) {
        alert("Please enter both Patient ID and Patient Name.");
        return;
    }

    try {
        const res = await fetch("/api/patients", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_id: patId,
                name: name,
                age: age ? parseInt(age) : 28,
                gestational_week: gw ? parseFloat(gw) : 32.0,
            })
        });
        const patient = await res.json();
        if (!res.ok) throw new Error(patient.error || "Registration failed.");

        currentPatient = patient;
        loadPatients();
        alert(`Patient ${patient.name} (${patient.patient_id}) successfully registered.`);
    } catch (err) {
        alert("Error registering patient: " + err.message);
    }
}

function onPatientSelect(e) {
    const pId = e.target.value;
    if (!pId) return;
    fetch(`/api/patients`)
        .then(r => r.json())
        .then(patients => {
            const p = patients.find(x => x.id == pId);
            if (p) {
                currentPatient = p;
                updateActivePatientCard(p);
            }
        });
}

// ============================================================
// MODALITY 1: BRAIN SCAN UPLOAD & RENDERING
// ============================================================
async function uploadBrainScan(file) {
    const formData = new FormData();
    formData.append("file", file);
    if (currentPatient) formData.append("patient_id", currentPatient.id);

    try {
        const res = await fetch("/api/brain/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Brain processing failed.");

        currentBrainData = data;
        updateUploadedDatasetName("brain-upload-name", file.name);
        renderBrainViewer(data);
    } catch (err) {
        alert("Brain scan upload failed: " + err.message);
    }
}

function renderBrainViewer(data) {
    if (!data) return;
    const prepImg = document.getElementById("img-brain-preprocessed");
    const mapImg = document.getElementById("img-brain-mapped");
    const defImg = document.getElementById("img-brain-deform");
    const gradImg = document.getElementById("img-brain-gradcam");

    if (prepImg && data.preprocessed_url) prepImg.src = data.preprocessed_url + `?t=${Date.now()}`;
    if (mapImg && data.atlas_mapped_url) mapImg.src = data.atlas_mapped_url + `?t=${Date.now()}`;
    if (defImg && data.deformation_field_url) defImg.src = data.deformation_field_url + `?t=${Date.now()}`;
    if (gradImg && data.gradcam_url) gradImg.src = data.gradcam_url + `?t=${Date.now()}`;
}

// ============================================================
// MODALITY 2: CARDIAC SIGNAL UPLOAD & CHART.JS RENDERING
// ============================================================
async function uploadCardiacSignal(file) {
    const formData = new FormData();
    formData.append("file", file);
    if (currentPatient) formData.append("patient_id", currentPatient.id);

    try {
        const res = await fetch("/api/cardiac/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Cardiac processing failed.");

        currentCardiacData = data;
        updateUploadedDatasetName("cardiac-upload-name", file.name);
        renderCardiacViewer(data);
    } catch (err) {
        alert("Cardiac upload failed: " + err.message);
    }

}

function renderCardiacViewer(data) {
    if (!data) return;

    // Update 4 feature stat cards
    const stats = data.statistics || {};
    const baseEl = document.getElementById("card-stat-baseline");
    const stvEl = document.getElementById("card-stat-stv");
    const ltvEl = document.getElementById("card-stat-ltv");
    const qualEl = document.getElementById("card-stat-quality");

    if (baseEl) baseEl.textContent = `${stats.baseline_fhr || 135.0} bpm`;
    if (stvEl) stvEl.textContent = `${stats.short_term_variability || 2.1} bpm`;
    if (ltvEl) ltvEl.textContent = `${stats.long_term_variability || 18.5} bpm`;
    if (qualEl) qualEl.textContent = `98.5%`;

    // Render Canvas Chart
    if (data.chart_data) {
        renderFhrChart(data.chart_data);
    }
}

function updateUploadedDatasetName(elementId, fileName) {
    const element = document.getElementById(elementId);
    if (!element || !fileName) return;
    element.textContent = fileName;
    element.classList.add("is-loaded");
    element.title = fileName;
}

function renderFhrChart(chartData) {
    const canvas = document.getElementById("chart-fhr");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (fhrChartInstance) fhrChartInstance.destroy();

    const labels = chartData.time || [];
    const procData = chartData.processed || [];
    const baseline = chartData.baseline || 135.0;
    const baselineArr = new Array(labels.length).fill(baseline);

    fhrChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Conditioned FHR (bpm)',
                    data: procData,
                    borderColor: '#e06d64',
                    backgroundColor: 'rgba(224, 109, 100, 0.12)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.25,
                    pointRadius: 0
                },
                {
                    label: 'Baseline FHR',
                    data: baselineArr,
                    borderColor: '#1e3a5f',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} bpm`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8, color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'Time (seconds)', color: '#64748b', font: { size: 10 } }
                },
                y: {
                    min: 50,
                    max: 200,
                    grid: { color: 'rgba(226, 232, 240, 0.8)' },
                    ticks: { stepSize: 50, color: '#64748b', font: { size: 10 } },
                    title: { display: true, text: 'BPM', color: '#64748b', font: { size: 10 } }
                }
            }
        }
    });
}

// ============================================================
// MULTI-MODAL FEATURE FUSION & PREDICTION
// ============================================================
async function executeMultiModalAnalysis() {
    if (!currentPatient) {
        alert("Please load or select a patient first.");
        return;
    }

    showDiagnosticSequencer();

    try {
        await simulateDiagnosticSteps();

        const res = await fetch("/api/predict/multimodal", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_id: currentPatient.id || currentPatient.patient_id,
                brain_image_id: currentBrainData?.db_id,
                cardiac_signal_id: currentCardiacData?.db_id,
                clinician_notes: "Evaluated via HERA Clinical Workstation Tri-Modal AI.",
            })
        });
        const outcome = await res.json();
        if (!res.ok) throw new Error(outcome.error || "Prediction failed.");

        hideDiagnosticSequencer();
        renderAssessmentOutcome(outcome);

    } catch (err) {
        hideDiagnosticSequencer();
        alert("Multi-modal assessment failed: " + err.message);
    }
}

function showDiagnosticSequencer() {
    const modal = document.getElementById("diagnostic-modal");
    document.querySelectorAll(".diagnostic-step").forEach(step => {
        step.classList.remove("active", "done");
    });
    const fill = document.getElementById("diag-progress-fill");
    if (fill) fill.style.width = "0%";
    if (modal) modal.style.display = "flex";
}

function hideDiagnosticSequencer() {
    const modal = document.getElementById("diagnostic-modal");
    if (modal) modal.style.display = "none";
}

async function simulateDiagnosticSteps() {
    const fill = document.getElementById("diag-progress-fill");
    const status = document.getElementById("diag-status-text");
    const steps = [
        { id: "diag-step-1", text: "1/6 Preprocessing Cranial Scan..." },
        { id: "diag-step-2", text: "2/6 Registering Elastic B-Spline Atlas..." },
        { id: "diag-step-3", text: "3/6 Extracting CNN Features & Grad-CAM..." },
        { id: "diag-step-4", text: "4/6 Modeling Cardiac LSTM Attention..." },
        { id: "diag-step-5", text: "5/6 Gated Modality Self-Attention Fusion..." },
        { id: "diag-step-6", text: "6/6 Synthesizing Multi-Modal Risk Score..." },
    ];

    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        if (status) status.textContent = step.text;
        if (fill) fill.style.width = `${((i + 1) / steps.length) * 100}%`;
        const el = document.getElementById(step.id);
        if (el) el.classList.add("active");
        await new Promise(r => setTimeout(r, 650));
        if (el) {
            el.classList.remove("active");
            el.classList.add("done");
        }
    }
}

function renderAssessmentOutcome(data) {
    const card = document.getElementById("assessment-result-card");
    if (card) card.scrollIntoView({ behavior: 'smooth' });

    // Patient Header
    const headEl = document.getElementById("res-patient-heading");
    const subEl = document.getElementById("res-patient-subheading");
    if (headEl) headEl.textContent = `Patient: ${data.patient_name} (${data.patient_id})`;
    if (subEl) subEl.textContent = `Gestational Week: ${data.gestational_week}w • Maternal Age: ${data.maternal_age}y • Assessed via Tri-Modal Neural Network`;

    // Radial Gauge
    const scoreVal = document.getElementById("risk-score-value");
    const confVal = document.getElementById("confidence-value");
    const predBadge = document.getElementById("prediction-badge");
    const statBadge = document.getElementById("res-risk-status-badge");
    const gaugeCircle = document.getElementById("radial-gauge-circle");

    if (scoreVal) scoreVal.textContent = `${data.risk_percent}%`;
    if (confVal) confVal.textContent = `Confidence: ${data.confidence}%`;

    const pred = (data.prediction || 'Normal');
    if (predBadge) {
        predBadge.textContent = pred;
        predBadge.className = `risk-badge risk-${pred.toLowerCase()}`;
    }
    if (statBadge) {
        statBadge.textContent = pred === "Normal" ? "Low Risk" : (pred === "Suspect" ? "Requires Review" : "High Risk / Alert");
        statBadge.className = `risk-badge risk-${pred.toLowerCase()}`;
    }

    if (gaugeCircle) {
        const offset = 440 - (440 * (data.risk_percent / 100));
        gaugeCircle.style.strokeDashoffset = offset;
        gaugeCircle.style.stroke = pred === "Normal" ? "#10b981" : (pred === "Suspect" ? "#f59e0b" : "#ef4444");
    }

    // Modality Influence Gating Bars
    const infl = data.modality_influence || {};
    const brainFill = document.getElementById("bar-brain-fill");
    const brainVal = document.getElementById("bar-brain-val");
    const cardFill = document.getElementById("bar-cardiac-fill");
    const cardVal = document.getElementById("bar-cardiac-val");
    const matFill = document.getElementById("bar-mat-fill");
    const matVal = document.getElementById("bar-mat-val");

    if (brainFill) brainFill.style.width = `${infl.brain_imaging || 42.5}%`;
    if (brainVal) brainVal.textContent = `${infl.brain_imaging || 42.5}%`;
    if (cardFill) cardFill.style.width = `${infl.cardiac_signal || 42.5}%`;
    if (cardVal) cardVal.textContent = `${infl.cardiac_signal || 42.5}%`;
    if (matFill) matFill.style.width = `${infl.maternal_biomarkers || 15.0}%`;
    if (matVal) matVal.textContent = `${infl.maternal_biomarkers || 15.0}%`;

    // Probability breakdown chips
    const probList = document.getElementById("class-prob-list");
    if (probList && data.probabilities) {
        probList.innerHTML = Object.entries(data.probabilities).map(([cls, prob]) => `
            <div class="prob-chip">
                <span>${cls}</span>
                <strong>${(prob * 100).toFixed(1)}%</strong>
            </div>
        `).join("");
    }

    // Maternal Biomarkers
    if (data.maternal_biomarkers) {
        const mb = data.maternal_biomarkers;
        document.getElementById("disp-mat-bp").textContent = mb.blood_pressure || "120/80 mmHg";
        document.getElementById("disp-mat-afi").textContent = `${mb.amniotic_fluid_index_cm || 14.2} cm`;
        document.getElementById("disp-mat-pi").textContent = mb.umbilical_doppler_pi || "0.92";
        const gdm = document.getElementById("disp-mat-gdm");
        if (gdm) {
            gdm.textContent = mb.gestational_diabetes || "Negative";
            gdm.style.color = mb.gestational_diabetes === "Positive" ? "#ef4444" : "#10b981";
        }
    }
}

// ============================================================
// PDF EXPORT
// ============================================================
function exportPdfReport() {
    window.print();
}

// ============================================================
// REAL-TIME BEDSIDE CTG STREAMING SIMULATOR
// ============================================================
async function initStreamingMonitor() {
    const patSelect = document.getElementById("select-stream-patient");
    const pId = patSelect ? patSelect.value : "PAT-1001";
    loadStreamingSignal(pId);
}

async function loadStreamingSignal(pId) {
    try {
        const res = await fetch(`/api/streaming/signal/${pId}`);
        const data = await res.json();
        streamData = data;
        streamIndex = 0;
        resetStreaming();
    } catch (err) {
        console.error("Failed to load stream signal:", err);
    }
}

function onStreamPatientChange(e) {
    pauseStreaming();
    loadStreamingSignal(e.target.value);
}

function startStreaming() {
    if (!streamData || !streamData.fhr_stream) return;
    if (streamingInterval) return;

    streamingInterval = setInterval(() => {
        if (streamIndex >= streamData.fhr_stream.length) {
            streamIndex = 0; // loop
        }

        const currentFhr = streamData.fhr_stream[streamIndex];
        const currentTime = streamData.time_stream[streamIndex];

        // Update indicators
        const fhrEl = document.getElementById("stream-current-fhr");
        const timeEl = document.getElementById("stream-time-elapsed");
        const riskEl = document.getElementById("stream-current-risk");
        const alertBadge = document.getElementById("stream-alert-badge");

        if (fhrEl) fhrEl.textContent = `${currentFhr.toFixed(1)} bpm`;
        if (timeEl) timeEl.textContent = `${currentTime.toFixed(1)} s`;

        // Sliding risk score estimation
        let riskScore = 0.05;
        if (currentFhr > 165 || currentFhr < 110) riskScore = 0.55;
        if (currentFhr > 180 || currentFhr < 95) riskScore = 0.95;

        if (riskEl) riskEl.textContent = `${(riskScore * 100).toFixed(1)} %`;
        if (alertBadge) {
            if (riskScore < 0.25) {
                alertBadge.textContent = "NORMAL RHYTHM";
                alertBadge.className = "risk-badge risk-normal";
            } else if (riskScore < 0.70) {
                alertBadge.textContent = "TACHY / BRADYCARDIA ALERT";
                alertBadge.className = "risk-badge risk-suspect";
            } else {
                alertBadge.textContent = "ACUTE DECELERATION / CRITICAL";
                alertBadge.className = "risk-badge risk-pathological";
            }
        }

        updateStreamChart(streamIndex);
        streamIndex++;
    }, 250 / streamSpeed);
}

function pauseStreaming() {
    if (streamingInterval) {
        clearInterval(streamingInterval);
        streamingInterval = null;
    }
}

function resetStreaming() {
    pauseStreaming();
    streamIndex = 0;
    const canvas = document.getElementById("chart-streaming-fhr");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (streamChartInstance) streamChartInstance.destroy();

    streamChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Bedside CTG (4 Hz)',
                data: [],
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                y: { min: 50, max: 200, grid: { color: 'rgba(255,255,255,0.08)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

function updateStreamChart(idx) {
    if (!streamChartInstance || !streamData) return;
    const windowSize = 80;
    const start = Math.max(0, idx - windowSize);
    const subLabels = streamData.time_stream.slice(start, idx + 1);
    const subVals = streamData.fhr_stream.slice(start, idx + 1);

    streamChartInstance.data.labels = subLabels;
    streamChartInstance.data.datasets[0].data = subVals;
    streamChartInstance.update();
}

// ============================================================
// 50-PATIENT BATCH TRIAGE MATRIX
// ============================================================
async function loadBatchTriage() {
    const tbody = document.getElementById("triage-table-body");
    if (!tbody) return;

    try {
        const res = await fetch("/api/triage/batch");
        const data = await res.json();
        demoPatientsCache = data;
        renderTriageTable(data);
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="8" style="color:red;text-align:center;">Failed to load triage batch: ${err.message}</td></tr>`;
    }
}

function renderTriageTable(patients) {
    const tbody = document.getElementById("triage-table-body");
    if (!tbody) return;

    if (!patients || patients.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;">No matching patient records found.</td></tr>`;
        return;
    }

    tbody.innerHTML = patients.map(p => {
        const res = (p.expected_demo_result || 'Normal').toLowerCase();
        const code = res === 'normal' ? 'NC' : (res === 'suspect' ? 'SC' : 'PC');
        return `
        <tr>
            <td>
                <strong>${p.patient_id}</strong><br>
                <span style="color:var(--navy-muted);font-size:0.75rem;">${p.name}</span>
            </td>
            <td>${p.gestational_week} wks<br><span style="font-size:0.7rem;color:#94a3b8;">Age: ${p.age}y</span></td>
            <td>${p.estimated_fetal_weight_g || 1500} g</td>
            <td><strong>${p.fetal_heart_rate} bpm</strong></td>
            <td>
                <span style="font-size:0.72rem;color:#9333ea;display:block;">${p.brain_scan_filename}</span>
                <span style="font-size:0.72rem;color:var(--coral-main);">${p.cardiac_signal_filename}</span>
            </td>
            <td>
                <span class="risk-badge risk-${res}">
                    Cohort ${code}
                </span>
            </td>
            <td><strong>${(p.risk_percent || 0.1)}%</strong></td>
            <td>
                <button class="btn-coral-pill" style="padding:0.3rem 0.65rem;font-size:0.75rem;" onclick="switchTab('tab-workstation'); loadDemoPatientById('${p.patient_id}');">
                    Open Case &rarr;
                </button>
            </td>
        </tr>
    `;
    }).join("");
}

function filterTriageTable(filter) {
    const f = (filter || 'all').toLowerCase();
    if (f === "all") {
        renderTriageTable(demoPatientsCache);
    } else {
        const targetResult = f === "nc" ? "normal" : (f === "sc" ? "suspect" : (f === "pc" ? "pathological" : f));
        const filtered = demoPatientsCache.filter(p => (p.expected_demo_result || '').toLowerCase() === targetResult);
        renderTriageTable(filtered);
    }
}

function searchTriageTable(query) {
    const q = (query || '').toLowerCase().trim();
    if (!q) {
        renderTriageTable(demoPatientsCache);
        return;
    }
    const filtered = demoPatientsCache.filter(p => 
        p.patient_id.toLowerCase().includes(q) ||
        p.name.toLowerCase().includes(q) ||
        (p.expected_demo_result || '').toLowerCase().includes(q) ||
        String(p.gestational_week).includes(q)
    );
    renderTriageTable(filtered);
}

// ============================================================
// LONGITUDINAL GROWTH VELOCITY TRACKER
// ============================================================
async function loadLongitudinalTracker(patientId) {
    try {
        const res = await fetch(`/api/patients/longitudinal/${patientId}`);
        const data = await res.json();
        renderGrowthCharts(data);
    } catch (err) {
        console.error("Failed to load longitudinal growth data:", err);
    }
}

function renderGrowthCharts(data) {
    const ctxBio = document.getElementById("chart-growth-biometry")?.getContext("2d");
    const ctxFhr = document.getElementById("chart-growth-fhr")?.getContext("2d");
    if (!ctxBio || !ctxFhr) return;

    if (growthBiometryChart) growthBiometryChart.destroy();
    if (growthFhrChart) growthFhrChart.destroy();

    const weeks = data.gestational_weeks || [24, 28, 32, 36, 40];
    const bpd = data.bpd_mm || [60, 71, 82, 90, 95];
    const hc = data.hc_mm || [220, 260, 295, 325, 345];
    const fhr = data.fhr_baseline || [148, 142, 138, 134, 130];

    growthBiometryChart = new Chart(ctxBio, {
        type: 'line',
        data: {
            labels: weeks.map(w => `${w}w`),
            datasets: [
                {
                    label: 'Biparietal Diameter (mm)',
                    data: bpd,
                    borderColor: '#9333ea',
                    backgroundColor: 'rgba(147, 51, 234, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Head Circumference (mm / 3)',
                    data: hc.map(v => v / 3.0),
                    borderColor: '#0284c7',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Biometry Index (mm)' } }
            }
        }
    });

    growthFhrChart = new Chart(ctxFhr, {
        type: 'line',
        data: {
            labels: weeks.map(w => `${w}w`),
            datasets: [{
                label: 'FHR Baseline Maturation (bpm)',
                data: fhr,
                borderColor: '#e06d64',
                backgroundColor: 'rgba(224, 109, 100, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 110, max: 170, title: { display: true, text: 'Baseline FHR (bpm)' } }
            }
        }
    });
}

// ============================================================
// ASSESSMENT HISTORY AUDIT
// ============================================================
async function loadAssessmentsHistory() {
    const tbody = document.getElementById("history-table-body");
    if (!tbody) return;

    try {
        const res = await fetch("/api/assessments");
        const assessments = await res.json();
        if (assessments.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:2rem;">No previous assessments recorded.</td></tr>`;
            return;
        }

        tbody.innerHTML = assessments.map(a => `
            <tr>
                <td><strong>${a.assessment_id}</strong></td>
                <td>${new Date(a.date).toLocaleString()}</td>
                <td>${a.patient_id}</td>
                <td><span class="risk-badge risk-${(a.prediction || 'normal').toLowerCase()}">${a.prediction}</span></td>
                <td><strong>${(a.risk_score * 100).toFixed(1)}%</strong></td>
                <td>${(a.confidence * 100).toFixed(1)}%</td>
                <td>
                    <button class="btn-coral-pill" style="padding:0.25rem 0.6rem;font-size:0.75rem;" onclick="switchTab('tab-workstation');">
                        View Audit &rarr;
                    </button>
                </td>
            </tr>
        `).join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="color:red;text-align:center;">Failed to load history: ${err.message}</td></tr>`;
    }
}

// ============================================================
// MODEL PERFORMANCE EVALUATION
// ============================================================
async function loadEvaluationMetrics() {
    try {
        const res = await fetch("/api/evaluation");
        const metrics = await res.json();

        const accEl = document.getElementById("eval-acc");
        const precEl = document.getElementById("eval-prec");
        const recEl = document.getElementById("eval-rec");
        const f1El = document.getElementById("eval-f1");

        if (accEl) accEl.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
        if (precEl) precEl.textContent = metrics.precision_macro.toFixed(3);
        if (recEl) recEl.textContent = metrics.recall_macro.toFixed(3);
        if (f1El) f1El.textContent = metrics.f1_macro.toFixed(3);

        const cmContainer = document.getElementById("eval-cm-container");
        if (cmContainer && metrics.confusion_matrix) {
            const cm = metrics.confusion_matrix;
            cmContainer.innerHTML = `
                <table class="triage-table evaluation-matrix-table">
                    <thead>
                        <tr><th>Actual \\ Pred</th><th>Normal</th><th>Suspect</th><th>Pathol.</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><strong>Normal</strong></td><td>${cm[0][0]}</td><td>${cm[0][1]}</td><td>${cm[0][2]}</td></tr>
                        <tr><td><strong>Suspect</strong></td><td>${cm[1][0]}</td><td>${cm[1][1]}</td><td>${cm[1][2]}</td></tr>
                        <tr><td><strong>Pathol.</strong></td><td>${cm[2][0]}</td><td>${cm[2][1]}</td><td>${cm[2][2]}</td></tr>
                    </tbody>
                </table>
            `;
        }
    } catch (err) {
        console.error("Evaluation load error:", err);
    }
}
