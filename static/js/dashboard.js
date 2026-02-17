// ============================================
// AgriGraph AI Dashboard - Main JavaScript
// ============================================

// Global State
let allAlerts = [];
let filteredAlerts = [];
let currentPage = 1;
const alertsPerPage = 20;
let eventSource = null;
let currentJobId = null;

// ============================================
// Document Ready
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');

    // Attach event listeners
    document.getElementById('trainBtn').addEventListener('click', handleTrainModel);
    document.getElementById('loadBtn').addEventListener('click', handleLoadModel);
    document.getElementById('riskFilter').addEventListener('change', filterAlerts);
    document.getElementById('alertSearch').addEventListener('input', filterAlerts);
    document.getElementById('exportCSVBtn').addEventListener('click', exportToCSV);
    document.getElementById('uploadBtn').addEventListener('click', handleFileUpload);
    document.getElementById('generateLLMBtn').addEventListener('click', generateLLMRecommendations);

    // Seed control event listeners
    document.getElementById('seedMode').addEventListener('change', handleSeedModeChange);
    document.getElementById('generateSeedBtn').addEventListener('click', generateNewSeed);

    // Add cancel button listener if it exists
    const cancelBtn = document.getElementById('cancelBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', cancelTraining);
    }

    // Check for pre-trained model
    checkPretrainedModel();

    // Load current seed on page load
    loadCurrentSeed();
});

// Close EventSource connection on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
});

// ============================================
// Model Status Check
// ============================================

async function checkPretrainedModel() {
    try {
        const response = await fetch('/api/check_model');
        const data = await response.json();

        if (data.model_exists) {
            document.getElementById('loadBtn').style.display = 'block';
            document.getElementById('modelStatus').innerHTML =
                '<i class="bi bi-check-circle"></i> Pre-trained Model Available';
            document.getElementById('modelStatus').className = 'badge bg-success';
        }
    } catch (error) {
        console.error('Error checking model:', error);
    }
}

// ============================================
// Seed Control Handlers
// ============================================

function handleSeedModeChange() {
    const mode = document.getElementById('seedMode').value;
    const customInput = document.getElementById('customSeed');

    if (mode === 'custom') {
        customInput.disabled = false;
        customInput.focus();
    } else {
        customInput.disabled = true;
        customInput.value = '';
    }
}

async function generateNewSeed() {
    const mode = document.getElementById('seedMode').value;
    const customInput = document.getElementById('customSeed');
    let customValue = null;

    if (mode === 'custom') {
        customValue = parseInt(customInput.value);
        if (isNaN(customValue) || customValue < 0 || customValue > 2147483647) {
            showError('Invalid Seed', 'Please enter a valid seed between 0 and 2147483647');
            return;
        }
    }

    try {
        const response = await fetch('/api/seed/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode, custom_seed: customValue })
        });

        const data = await response.json();
        if (data.success) {
            updateCurrentSeedDisplay(data.seed, data.mode);
            return data.seed;
        } else {
            showError('Seed Generation Failed', data.error || 'Unknown error');
        }
    } catch (error) {
        console.error('Error generating seed:', error);
        showError('Seed Generation Error', error.message);
    }
}

function updateCurrentSeedDisplay(seed, mode) {
    const display = document.getElementById('currentSeedDisplay');
    let modeText;

    if (mode === 'auto') {
        modeText = 'Auto (timestamp-based)';
    } else if (mode === 'fixed') {
        modeText = 'Fixed (42)';
    } else if (mode === 'custom') {
        modeText = `Custom (${seed})`;
    } else {
        modeText = mode;
    }

    display.innerHTML = `
        <div class="alert alert-info alert-sm mb-0">
            <i class="bi bi-check-circle"></i>
            Current Seed: <strong>${seed}</strong> (${modeText})
        </div>
    `;
}

async function loadCurrentSeed() {
    try {
        const response = await fetch('/api/seed/current');
        const data = await response.json();
        if (data.success) {
            updateCurrentSeedDisplay(data.seed, data.mode);
        }
    } catch (error) {
        console.error('Error loading current seed:', error);
        // Fail silently - this is not critical
    }
}

// ============================================
// Button Handlers
// ============================================

async function handleTrainModel() {
    console.log('=== TRAIN BUTTON CLICKED ===');
    console.log('trainBtn element:', document.getElementById('trainBtn'));
    console.log('progressSection element:', document.getElementById('progressSection'));
    await runAnalysis(false);
}

async function handleLoadModel() {
    await runAnalysis(true);
}

async function runAnalysis(usePretrained) {
    console.log('=== RUNANALYSIS FUNCTION STARTED ===');
    console.log('usePretrained:', usePretrained);

    const trainBtn = document.getElementById('trainBtn');
    const loadBtn = document.getElementById('loadBtn');
    const progressSection = document.getElementById('progressSection');

    console.log('trainBtn:', trainBtn);
    console.log('loadBtn:', loadBtn);
    console.log('progressSection:', progressSection);

    // Disable buttons and show progress
    if (trainBtn) trainBtn.disabled = true;
    if (loadBtn) loadBtn.disabled = true;
    if (progressSection) {
        console.log('Removing d-none from progressSection');
        progressSection.classList.remove('d-none');
        console.log('progressSection classList:', progressSection.classList);

        // SCROLL TO SHOW PROGRESS BAR
        console.log('Scrolling to progress section...');
        progressSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
        console.error('ERROR: progressSection not found!');
    }
    document.getElementById('trainingHistorySection').classList.add('d-none');
    document.getElementById('llmSection').classList.add('d-none');
    closeErrorAlert();

    // Clear old LLM recommendations
    document.getElementById('gptRecommendation').innerHTML = '<em class="text-muted">Running new analysis...</em>';
    document.getElementById('claudeRecommendation').innerHTML = '<em class="text-muted">Running new analysis...</em>';
    document.getElementById('gptStatus').textContent = '';
    document.getElementById('claudeStatus').textContent = '';

    try {
        // Get seed parameters from UI
        const seedMode = document.getElementById('seedMode').value;
        let customSeed = null;
        if (seedMode === 'custom') {
            const seedValue = document.getElementById('customSeed').value;
            customSeed = seedValue ? parseInt(seedValue) : null;
        }

        console.log('Calling /api/run_analysis with:', {
            use_pretrained: usePretrained,
            seed_mode: seedMode,
            custom_seed: customSeed
        });

        // Call API to start training and get job_id
        const response = await fetch('/api/run_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                use_pretrained: usePretrained,
                seed_mode: seedMode,
                custom_seed: customSeed
            })
        });

        console.log('API response status:', response.status);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Analysis failed');
        }

        const data = await response.json();
        console.log('API response data:', data);

        currentJobId = data.job_id;
        console.log('Job ID:', currentJobId);
        console.log('Connecting to SSE stream...');

        // Connect to progress stream
        connectToProgressStream(currentJobId);

    } catch (error) {
        console.error('Error:', error);
        showError('Analysis Failed', error.message);
        trainBtn.disabled = false;
        loadBtn.disabled = false;
        document.getElementById('progressSection').classList.add('d-none');
    }
}

// ============================================
// Progress Tracking with EventSource
// ============================================

function connectToProgressStream(jobId) {
    // Close existing connection
    if (eventSource) eventSource.close();

    console.log('Creating EventSource for job:', jobId);
    eventSource = new EventSource(`/api/training/progress/${jobId}`);

    eventSource.addEventListener('progress', (e) => {
        try {
            console.log('SSE progress event received:', e.data.substring(0, 100));
            const progress = JSON.parse(e.data);
            console.log('Parsed progress data:', progress);
            updateProgressUI(progress);

            // Check for completion using 'event' field from backend
            if (progress.event === 'completed') {
                eventSource.close();
                eventSource = null;
                loadResults();
            } else if (progress.event === 'error') {
                eventSource.close();
                eventSource = null;
                showError('Training Error', progress.error || 'Unknown error occurred');
                resetButtons();
            }
        } catch (error) {
            console.error('Error parsing progress data:', error);
        }
    });

    eventSource.onerror = (e) => {
        console.error('SSE connection error:', e);
        console.log('EventSource readyState:', eventSource?.readyState);
        eventSource.close();
        eventSource = null;
        showError('Connection Lost', 'SSE connection failed - training may continue in background');
        resetButtons();
    };
}

function updateProgressUI(progress) {
    console.log('updateProgressUI called with:', progress);

    // Skip UI update for completion/error events - they're handled by main listener
    if (progress.event === 'completed' || progress.event === 'error') {
        console.log('Skipping UI update for completion/error event');
        return;
    }

    // FIX: Use correct field names from backend and add fallbacks for missing fields
    const totalEpochs = progress.total_epochs || progress.num_epochs || progress.epoch || 1;
    const epoch = progress.epoch || 0;

    const pct = totalEpochs > 0 ? Math.round((epoch / totalEpochs) * 100) : 0;
    const bar = document.getElementById('progressBar');
    const percent = document.getElementById('progressPercent');
    const status = document.getElementById('progressStatus');

    console.log('Progress bar elements:', { bar, percent, status });

    if (!bar || !percent || !status) {
        console.error('Missing progress bar elements!');
        return;
    }

    bar.style.width = pct + '%';
    percent.textContent = pct + '%';

    // ETA calculation - handle missing field
    const eta = progress.eta_seconds || 0;
    const etaMin = Math.floor(eta / 60);
    const etaSec = Math.round(eta % 60);

    status.innerHTML = `
        <i class="bi bi-hourglass-split spinner"></i>
        Epoch ${epoch}/${totalEpochs} |
        Train Loss: ${(progress.train_loss || 0).toFixed(4)} |
        Val Loss: ${(progress.val_loss || 0).toFixed(4)} |
        Val R²: ${(progress.val_r2 || 0).toFixed(4)} |
        ETA: ${etaMin}m ${etaSec}s
    `;

    bar.classList.remove('bg-danger');
}

function cancelTraining() {
    if (!currentJobId) return;
    if (!confirm('Cancel training?')) return;
    fetch(`/api/training/cancel/${currentJobId}`, { method: 'POST' });
}

// ============================================
// Load Results
// ============================================

async function loadResults() {
    try {
        // Close EventSource if still connected
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        // Fetch all data in parallel
        const [dataResponse, fieldMapResponse, heatmapResponse, alertsResponse] =
            await Promise.all([
                fetch('/api/data'),
                fetch('/api/field_map'),
                fetch('/api/heatmap'),
                fetch('/api/alerts')
            ]);

        const data = await dataResponse.json();
        const fieldMapData = await fieldMapResponse.json();
        const heatmapData = await heatmapResponse.json();
        const alertsData = await alertsResponse.json();

        // Update stats
        updateStats(data);

        // Update visualizations
        updateFieldMap(fieldMapData);
        updateHeatmap(heatmapData);

        // Check for training history
        if (data.history) {
            try {
                const historyResponse = await fetch('/api/training_history');
                const historyData = await historyResponse.json();
                updateTrainingHistory(historyData);
                document.getElementById('trainingHistorySection').classList.remove('d-none');
            } catch (error) {
                console.error('Error loading training history:', error);
            }
        }

        // Update alerts (extract alerts array from response)
        updateAlerts(alertsData.alerts || []);

        // Load GNN graph visualization
        loadGNNGraph();

        // Hide progress and show export button and LLM section
        document.getElementById('progressSection').classList.add('d-none');
        document.getElementById('exportCSVBtn').style.display = 'block';
        document.getElementById('llmSection').classList.remove('d-none');

        resetButtons();

    } catch (error) {
        console.error('Error loading results:', error);
        showError('Load Results Failed', error.message);
        resetButtons();
    }
}

function resetButtons() {
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('loadBtn').disabled = false;
}

// ============================================
// Update Stats Cards
// ============================================

function updateStats(data) {
    document.getElementById('numLocations').textContent = data.num_nodes || '-';
    document.getElementById('criticalAlerts').textContent = data.critical_count || 0;
    document.getElementById('highRiskCount').textContent = data.high_count || 0;

    const r2Score = data.r2_score !== undefined ? parseFloat(data.r2_score).toFixed(3) : '-';
    document.getElementById('r2Score').textContent = r2Score;
}

// ============================================
// Update Visualizations
// ============================================

function updateFieldMap(data) {
    try {
        Plotly.newPlot('fieldMapPlot', data.data, data.layout, { responsive: true, displayModeBar: true });
    } catch (error) {
        console.error('Error updating field map:', error);
    }
}

function updateHeatmap(data) {
    try {
        Plotly.newPlot('heatmapPlot', data.data, data.layout, { responsive: true, displayModeBar: true });
    } catch (error) {
        console.error('Error updating heatmap:', error);
    }
}

function updateTrainingHistory(data) {
    try {
        Plotly.newPlot('trainingHistoryPlot', data.data, data.layout, { responsive: true, displayModeBar: true });
    } catch (error) {
        console.error('Error updating training history:', error);
    }
}

// ============================================
// Alert Management
// ============================================

function updateAlerts(alerts) {
    allAlerts = alerts;
    filteredAlerts = [...allAlerts];
    currentPage = 1;
    renderAlerts();
}

function filterAlerts() {
    const riskLevel = document.getElementById('riskFilter').value;
    const searchTerm = document.getElementById('alertSearch').value.toLowerCase();

    filteredAlerts = allAlerts.filter(alert => {
        const matchesRisk = riskLevel === 'all' || alert.risk_level === riskLevel;
        const matchesSearch = searchTerm === '' ||
            alert.location_id.toString().includes(searchTerm) ||
            alert.primary_gas.toLowerCase().includes(searchTerm) ||
            alert.recommendation.toLowerCase().includes(searchTerm);
        return matchesRisk && matchesSearch;
    });

    currentPage = 1;
    renderAlerts();
}

function renderAlerts() {
    const start = (currentPage - 1) * alertsPerPage;
    const end = start + alertsPerPage;
    const pageAlerts = filteredAlerts.slice(start, end);

    const tbody = document.getElementById('alertsTableBody');

    if (pageAlerts.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted py-4">
                    <i class="bi bi-inbox"></i> No alerts match the selected filters.
                </td>
            </tr>
        `;
    } else {
        tbody.innerHTML = pageAlerts.map(alert => `
            <tr class="fade-in">
                <td>
                    <strong>ID: ${alert.location_id}</strong><br>
                    <small class="text-muted">(${alert.x.toFixed(1)}, ${alert.y.toFixed(1)})</small>
                </td>
                <td>
                    <span class="badge badge-${alert.risk_level}">
                        ${alert.risk_level.toUpperCase()}
                    </span>
                </td>
                <td>
                    <strong>${alert.risk_score.toFixed(3)}</strong>
                </td>
                <td>
                    ${alert.primary_gas}<br>
                    <small class="text-muted">${alert.primary_gas_concentration.toFixed(2)} ppm</small>
                </td>
                <td>
                    <small>${alert.recommendation}</small>
                </td>
            </tr>
        `).join('');
    }

    renderPagination();
}

function renderPagination() {
    const totalPages = Math.ceil(filteredAlerts.length / alertsPerPage);
    const pagination = document.getElementById('alertsPagination');

    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = '';

    // Previous button
    if (currentPage > 1) {
        html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(${currentPage - 1}); return false;">Previous</a></li>`;
    }

    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    if (startPage > 1) {
        html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(1); return false;">1</a></li>`;
        if (startPage > 2) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        if (i === currentPage) {
            html += `<li class="page-item active"><span class="page-link">${i}</span></li>`;
        } else {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(${i}); return false;">${i}</a></li>`;
        }
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
        html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(${totalPages}); return false;">${totalPages}</a></li>`;
    }

    // Next button
    if (currentPage < totalPages) {
        html += `<li class="page-item"><a class="page-link" href="#" onclick="changePage(${currentPage + 1}); return false;">Next</a></li>`;
    }

    pagination.innerHTML = html;
}

function changePage(pageNum) {
    currentPage = pageNum;
    renderAlerts();
    // Scroll to table
    document.querySelector('#alertsTable').scrollIntoView({ behavior: 'smooth' });
}

// ============================================
// Export Functionality
// ============================================

function exportToCSV() {
    window.location.href = '/api/export/alerts/csv';
}

// ============================================
// File Upload Functionality
// ============================================

async function handleFileUpload() {
    const fileInput = document.getElementById('csvFileInput');
    const file = fileInput.files[0];
    const uploadStatus = document.getElementById('uploadStatus');

    if (!file) {
        uploadStatus.innerHTML = '<div class="alert alert-warning alert-sm mb-0"><i class="bi bi-exclamation-circle"></i> Please select a file first</div>';
        return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        uploadStatus.innerHTML = '<div class="alert alert-danger alert-sm mb-0"><i class="bi bi-x-circle"></i> File is too large (max 10MB)</div>';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const uploadBtn = document.getElementById('uploadBtn');
    const originalBtnText = uploadBtn.innerHTML;
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="bi bi-hourglass-split spinner"></i> Uploading...';

    uploadStatus.innerHTML = '<div class="alert alert-info alert-sm mb-0"><i class="bi bi-info-circle"></i> Processing file...</div>';

    try {
        const response = await fetch('/api/upload_csv', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            uploadStatus.innerHTML = `
                <div class="alert alert-success alert-sm mb-0">
                    <i class="bi bi-check-circle"></i>
                    <strong>Success!</strong> Loaded ${data.num_nodes} sensor locations from ${data.filename}
                    <br><small class="text-muted">
                        ${data.has_labels ? '✓ Risk labels found' : '○ No risk labels'} |
                        ${data.columns.length} columns
                    </small>
                </div>
            `;
            // Clear file input
            fileInput.value = '';
        } else {
            uploadStatus.innerHTML = `
                <div class="alert alert-danger alert-sm mb-0">
                    <i class="bi bi-x-circle"></i>
                    <strong>Upload Failed:</strong> ${data.error}
                </div>
            `;
        }
    } catch (error) {
        uploadStatus.innerHTML = `
            <div class="alert alert-danger alert-sm mb-0">
                <i class="bi bi-x-circle"></i>
                <strong>Upload Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = originalBtnText;
    }
}

// ============================================
// Error Handling
// ============================================

function showError(title, message) {
    const errorAlert = document.getElementById('errorAlert');
    document.getElementById('errorTitle').textContent = title;
    document.getElementById('errorMessage').textContent = message;
    errorAlert.classList.remove('hide');
    errorAlert.classList.add('show', 'fade-in');
}

function closeErrorAlert() {
    const errorAlert = document.getElementById('errorAlert');
    errorAlert.classList.add('hide');
    errorAlert.classList.remove('show');
}

// ============================================
// Utility Functions
// ============================================

function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

// Auto-close error alert after 10 seconds
document.addEventListener('DOMContentLoaded', function() {
    const errorAlert = document.getElementById('errorAlert');
    if (errorAlert) {
        errorAlert.addEventListener('show.bs.alert', function() {
            setTimeout(() => {
                errorAlert.classList.add('hide');
            }, 10000);
        });
    }
});

// ============================================
// LLM Recommendations
// ============================================

let lastLLMRequestTime = 0;
const LLM_REQUEST_COOLDOWN = 5000; // 5 second minimum between requests

async function generateLLMRecommendations() {
    // Prevent rapid-fire requests (rate limit protection)
    const now = Date.now();
    if (now - lastLLMRequestTime < LLM_REQUEST_COOLDOWN) {
        const waitSeconds = Math.ceil((LLM_REQUEST_COOLDOWN - (now - lastLLMRequestTime)) / 1000);
        showError('Please Wait', `Please wait ${waitSeconds} second${waitSeconds > 1 ? 's' : ''} before generating new recommendations`);
        return;
    }
    lastLLMRequestTime = now;
    const gptDiv = document.getElementById('gptRecommendation');
    const claudeDiv = document.getElementById('claudeRecommendation');
    const gptStatus = document.getElementById('gptStatus');
    const claudeStatus = document.getElementById('claudeStatus');
    const button = document.getElementById('generateLLMBtn');
    const comparisonNotes = document.getElementById('comparisonNotes');

    button.disabled = true;
    button.innerHTML = '<i class="bi bi-hourglass-split spinner"></i> Generating...';

    // Show loading states
    gptDiv.innerHTML = '<div class="d-flex align-items-center"><div class="spinner-border text-primary me-2" role="status" style="width: 1.5rem; height: 1.5rem;"></div> Analyzing with GPT-4...</div>';
    claudeDiv.innerHTML = '<div class="d-flex align-items-center"><div class="spinner-border text-info me-2" role="status" style="width: 1.5rem; height: 1.5rem;"></div> Analyzing with Claude...</div>';
    gptStatus.textContent = '';
    claudeStatus.textContent = '';
    comparisonNotes.style.display = 'none';

    try {
        const response = await fetch('/api/llm_recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.success) {
            const comparison = data.comparison;

            // Display GPT-4 recommendation
            if (comparison.gpt4 && comparison.gpt4.success) {
                gptDiv.innerHTML = `
                    <div class="recommendation-text">
                        ${formatRecommendation(comparison.gpt4.recommendation)}
                    </div>
                `;
                gptStatus.innerHTML = `<i class="bi bi-check-circle text-success"></i> ${comparison.gpt4.model}`;
            } else if (comparison.openai_gpt && comparison.openai_gpt.success) {
                gptDiv.innerHTML = `
                    <div class="recommendation-text">
                        ${formatRecommendation(comparison.openai_gpt.recommendation)}
                    </div>
                `;
                gptStatus.innerHTML = `<i class="bi bi-check-circle text-success"></i> ${comparison.openai_gpt.model}`;
            } else {
                const error = comparison.gpt4?.error || comparison.openai_gpt?.error || 'Unknown error';
                gptDiv.innerHTML = `<div class="alert alert-danger mb-0"><i class="bi bi-exclamation-triangle"></i> Error: ${error}</div>`;
                gptStatus.innerHTML = '<i class="bi bi-x-circle text-danger"></i> Failed';
            }

            // Display Claude recommendation
            if (comparison.claude && comparison.claude.success) {
                claudeDiv.innerHTML = `
                    <div class="recommendation-text">
                        ${formatRecommendation(comparison.claude.recommendation)}
                    </div>
                `;
                claudeStatus.innerHTML = `<i class="bi bi-check-circle text-success"></i> ${comparison.claude.model}`;
            } else {
                const error = comparison.claude?.error || 'Unknown error';
                claudeDiv.innerHTML = `<div class="alert alert-danger mb-0"><i class="bi bi-exclamation-triangle"></i> Error: ${error}</div>`;
                claudeStatus.innerHTML = '<i class="bi bi-x-circle text-danger"></i> Failed';
            }

            // Show comparison notes if both succeeded
            const gptSuccess = comparison.gpt4?.success || comparison.openai_gpt?.success;
            const claudeSuccess = comparison.claude?.success;
            if (gptSuccess && claudeSuccess) {
                comparisonNotes.style.display = 'block';
            }

        } else {
            showError('LLM Generation Failed', data.error);
            gptDiv.innerHTML = `<div class="alert alert-danger mb-0">Failed: ${data.error}</div>`;
            claudeDiv.innerHTML = `<div class="alert alert-danger mb-0">Failed: ${data.error}</div>`;
        }

    } catch (error) {
        showError('LLM Request Failed', error.message);
        gptDiv.innerHTML = `<div class="alert alert-danger mb-0">Error: ${error.message}</div>`;
        claudeDiv.innerHTML = `<div class="alert alert-danger mb-0">Error: ${error.message}</div>`;
    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="bi bi-magic"></i> Generate Recommendations';
    }
}

function formatRecommendation(text) {
    // Escape HTML
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Convert markdown-style formatting to HTML
    formatted = formatted
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Convert line breaks
    formatted = formatted.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');

    return `<p>${formatted}</p>`;
}

// ============================================
// GNN Graph Visualization
// ============================================

async function loadGNNGraph() {
    try {
        const response = await fetch('/api/graph/structure');
        if (!response.ok) return;

        const data = await response.json();
        if (!data.success) return;

        renderGNNGraph(data);
        document.getElementById('gnnSection').classList.remove('d-none');
    } catch (error) {
        console.error('Error loading GNN graph:', error);
    }
}

function renderGNNGraph(data) {
    const nodes = data.nodes;
    const edges = data.edges;
    const showEdges = document.getElementById('showEdgesToggle')?.checked ?? true;

    const traces = [];

    // Draw edges first (behind nodes)
    if (showEdges && edges.length > 0) {
        const edgeX = [], edgeY = [];
        edges.forEach(([i, j]) => {
            edgeX.push(nodes[i].x, nodes[j].x, null);
            edgeY.push(nodes[i].y, nodes[j].y, null);
        });

        traces.push({
            x: edgeX,
            y: edgeY,
            mode: 'lines',
            line: { color: 'rgba(100,100,100,0.15)', width: 0.5 },
            hoverinfo: 'none',
            showlegend: false
        });
    }

    // Draw nodes
    const nodeX = nodes.map(n => n.x);
    const nodeY = nodes.map(n => n.y);
    const nodeColor = nodes.map(n => n.risk_score);

    traces.push({
        x: nodeX,
        y: nodeY,
        mode: 'markers',
        marker: {
            size: 10,
            color: nodeColor,
            colorscale: 'RdYlGn_r',
            showscale: true,
            colorbar: {
                title: 'Risk<br>Score',
                thickness: 15,
                len: 0.7
            },
            line: { color: 'white', width: 1 }
        },
        text: nodes.map(n => `Node ${n.id}<br>Risk: ${n.risk_level}<br>Score: ${n.risk_score.toFixed(3)}<br>Neighbors: ${n.degree}`),
        hovertemplate: '<b>%{text}</b><extra></extra>',
        showlegend: false
    });

    const layout = {
        title: `GNN Graph Structure (${data.num_nodes} nodes, ${data.num_edges} edges)`,
        showlegend: false,
        hovermode: 'closest',
        margin: { b: 20, l: 20, r: 20, t: 40 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        plot_bgcolor: 'rgba(240,240,240,0.5)',
        paper_bgcolor: 'white'
    };

    Plotly.newPlot('gnnGraph', traces, layout, { responsive: true });
}

// Add toggle listener for show edges checkbox
document.addEventListener('DOMContentLoaded', function() {
    const showEdgesToggle = document.getElementById('showEdgesToggle');
    if (showEdgesToggle) {
        showEdgesToggle.addEventListener('change', function() {
            if (document.getElementById('gnnGraph')?.data) {
                loadGNNGraph();  // Redraw with toggle state
            }
        });
    }
});
