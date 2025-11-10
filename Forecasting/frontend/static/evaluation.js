// Model Evaluation JavaScript

const API_BASE = window.location.origin;
let allModels = [];
let selectedModel = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Model Evaluation page initialized');
    
    // Setup event listeners
    document.getElementById('refreshModels').addEventListener('click', loadTrainedModels);
    document.getElementById('filterSymbol').addEventListener('change', filterModels);
    document.getElementById('filterModel').addEventListener('change', filterModels);
    
    // Load trained models
    loadTrainedModels();
});

// Load all trained models from database
async function loadTrainedModels() {
    try {
        addLoadingState();
        
        const response = await fetch(`${API_BASE}/api/adaptive/trained-models`);
        const data = await response.json();
        
        if (data.success && data.models.length > 0) {
            allModels = data.models;
            
            // Populate symbol filter
            const symbols = [...new Set(data.models.map(m => m.symbol))];
            populateSymbolFilter(symbols);
            
            // Display models
            displayModels(allModels);
        } else {
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading trained models:', error);
        showErrorState(error.message);
    }
}

// Populate symbol filter dropdown
function populateSymbolFilter(symbols) {
    const filterSelect = document.getElementById('filterSymbol');
    
    // Keep "All Symbols" option
    filterSelect.innerHTML = '<option value="all">All Symbols</option>';
    
    // Add symbol options
    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol;
        option.textContent = symbol;
        filterSelect.appendChild(option);
    });
}

// Filter models based on selected filters
function filterModels() {
    const symbolFilter = document.getElementById('filterSymbol').value;
    const modelFilter = document.getElementById('filterModel').value;
    
    let filtered = allModels;
    
    if (symbolFilter !== 'all') {
        filtered = filtered.filter(m => m.symbol === symbolFilter);
    }
    
    if (modelFilter !== 'all') {
        filtered = filtered.filter(m => m.model_name === modelFilter);
    }
    
    displayModels(filtered);
}

// Display models in kanban board columns by performance
function displayModels(models) {
    if (models.length === 0) {
        showEmptyState();
        return;
    }
    
    // Categorize models by MAPE
    const excellent = models.filter(m => m.recent_mape < 3);
    const good = models.filter(m => m.recent_mape >= 3 && m.recent_mape < 5);
    const fair = models.filter(m => m.recent_mape >= 5 && m.recent_mape < 10);
    const poor = models.filter(m => m.recent_mape >= 10 || !m.recent_mape);
    
    // Update column counts
    document.querySelector('#excellentModels').closest('.performance-column').querySelector('.count').textContent = excellent.length;
    document.querySelector('#goodModels').closest('.performance-column').querySelector('.count').textContent = good.length;
    document.querySelector('#fairModels').closest('.performance-column').querySelector('.count').textContent = fair.length;
    document.querySelector('#poorModels').closest('.performance-column').querySelector('.count').textContent = poor.length;
    
    // Render model chips in each column
    renderColumnModels('excellentModels', excellent, '🌟');
    renderColumnModels('goodModels', good, '✓');
    renderColumnModels('fairModels', fair, '⚠️');
    renderColumnModels('poorModels', poor, '⚠️');
}

function renderColumnModels(containerId, models, icon) {
    const container = document.getElementById(containerId);
    
    if (models.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 30px 15px; font-size: 0.9em;">No models</p>';
        return;
    }
    
    const html = models.map(model => {
        const mape = model.recent_mape?.toFixed(2) || 'N/A';
        const predictions = model.total_predictions || 0;
        
        return `
            <div class="model-chip" onclick="selectModel('${model.symbol}', '${model.model_name}')">
                <div class="chip-header">
                    <div class="chip-name">${model.model_name.toUpperCase()}</div>
                    <div class="chip-badge">${icon}</div>
                </div>
                <div style="font-size: 0.85em; color: var(--text-muted); font-weight: 700; margin-bottom: 10px;">
                    ${model.symbol}
                </div>
                <div class="chip-metric">
                    <span class="key">MAPE</span>
                    <span class="val">${mape}%</span>
                </div>
                <div class="chip-metric">
                    <span class="key">Predictions</span>
                    <span class="val">${predictions}</span>
                </div>
                <div class="chip-metric" style="border-top: 1px solid var(--border); padding-top: 8px; margin-top: 8px;">
                    <span class="key">Versions</span>
                    <span class="val">${model.version_count || 1}</span>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

// Select a model to view details
async function selectModel(symbol, modelName) {
    selectedModel = { symbol, modelName };
    
    // Highlight selected card
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    
    // Show details section
    document.getElementById('modelDetails').style.display = 'block';
    
    // Scroll to details
    document.getElementById('modelDetails').scrollIntoView({ behavior: 'smooth' });
    
    // Load model details
    await Promise.all([
        loadPerformanceOverview(symbol, modelName),
        loadPerformanceTrend(symbol, modelName),
        loadVersionHistory(symbol, modelName),
        loadVersionComparison(symbol, modelName)
    ]);
}

// Load performance overview
async function loadPerformanceOverview(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/${symbol}/${modelName}`);
        const data = await response.json();
        
        if (data.success && data.statistics.status !== 'no_data') {
            const stats = data.statistics;
            
            const recent = stats.recent_performance || {};
            const baseline = stats.baseline_performance || {};
            const allTime = stats.all_time_performance || {};
            
            const html = `
                <div class="showcase-card">
                    <div class="title">Recent MAPE</div>
                    <div class="number">${recent.mape?.toFixed(2) || 'N/A'}%</div>
                </div>
                <div class="showcase-card">
                    <div class="title">Baseline MAPE</div>
                    <div class="number" style="color: var(--accent);">${baseline.mape?.toFixed(2) || 'N/A'}%</div>
                </div>
                <div class="showcase-card">
                    <div class="title">All-Time MAPE</div>
                    <div class="number" style="color: var(--secondary);">${allTime.mape?.toFixed(2) || 'N/A'}%</div>
                </div>
                <div class="showcase-card">
                    <div class="title">Predictions</div>
                    <div class="number" style="color: var(--success);">${stats.total_predictions || 0}</div>
                </div>
                <div class="showcase-card">
                    <div class="title">Trainings</div>
                    <div class="number" style="color: var(--info);">${stats.training_count || 0}</div>
                </div>
                <div class="showcase-card">
                    <div class="title">Days Ago</div>
                    <div class="number" style="color: var(--warning);">${stats.days_since_training !== null ? stats.days_since_training : 'N/A'}</div>
                </div>
            `;
            
            document.getElementById('performanceOverview').innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading performance overview:', error);
    }
}

// Load performance trend
async function loadPerformanceTrend(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/trend/${symbol}/${modelName}?days=30`);
        const data = await response.json();
        
        if (data.success && data.trend.length > 0) {
            const trend = data.trend;
            
            const trace = {
                x: trend.map(t => t.date),
                y: trend.map(t => t.mape),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MAPE Score',
                line: { color: '#14b8a6', width: 4 },
                marker: { 
                    size: 9, 
                    color: '#14b8a6',
                    line: { color: 'white', width: 2 }
                },
                fill: 'tozeroy',
                fillcolor: 'rgba(20, 184, 166, 0.1)'
            };
            
            const layout = {
                title: {
                    text: `${modelName.toUpperCase()} - ${symbol}`,
                    font: { color: '#1e293b', size: 17, family: 'system-ui', weight: 800 }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(241, 245, 249, 0.5)',
                xaxis: {
                    title: 'Date',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                yaxis: {
                    title: 'MAPE (%)',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                font: { family: 'system-ui, sans-serif', color: '#1e293b' },
                margin: { t: 50, b: 50, l: 60, r: 30 }
            };
            
            Plotly.newPlot('performanceTrendChart', [trace], layout, {responsive: true});
        } else {
            document.getElementById('performanceTrendChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No performance trend data available</p>';
        }
    } catch (error) {
        console.error('Error loading performance trend:', error);
    }
}

// Load version history
async function loadVersionHistory(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/versions/${symbol}/${modelName}?limit=10`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const html = data.history.map((v, index) => {
                const isActive = v.status === 'active';
                const date = new Date(v.trained_at).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
                const mape = v.performance?.mape?.toFixed(2) || 'N/A';
                
                return `
                    <div class="timeline-entry ${isActive ? 'active' : ''}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 1.3em; font-weight: 900; color: var(--primary);">${v.version}</span>
                            ${isActive ? '<span style="background: var(--success); color: white; padding: 5px 14px; border-radius: 15px; font-size: 0.75em; font-weight: 800; text-transform: uppercase;">Active</span>' : ''}
                        </div>
                        <div style="font-size: 0.85em; color: var(--text-muted); font-weight: 600; margin-bottom: 12px;">
                            📅 ${date}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px;">
                            <div style="background: white; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 0.7em; color: var(--text-muted); font-weight: 700;">MAPE</div>
                                <div style="font-size: 1.5em; font-weight: 900; color: var(--primary);">${mape}%</div>
                            </div>
                            <div style="background: white; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 0.7em; color: var(--text-muted); font-weight: 700;">Type</div>
                                <div style="font-size: 1em; font-weight: 800; color: var(--secondary); text-transform: uppercase;">${v.update_type || 'N/A'}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            document.getElementById('versionHistory').innerHTML = html;
        } else {
            document.getElementById('versionHistory').innerHTML = 
                '<p style="text-align: center; color: var(--text-muted); padding: 40px;">No versions yet</p>';
        }
    } catch (error) {
        console.error('Error loading version history:', error);
    }
}

// Load version comparison chart
async function loadVersionComparison(symbol, modelName) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/versions/${symbol}/${modelName}?limit=10`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const versions = data.history.reverse(); // Oldest to newest
            
            // Create separate traces for each metric
            const mapeTrace = {
                x: versions.map(v => v.version),
                y: versions.map(v => v.performance?.mape || 0),
                type: 'bar',
                name: 'MAPE (%)',
                marker: { 
                    color: '#14b8a6',
                    line: {
                        color: '#0f766e',
                        width: 2
                    }
                },
                text: versions.map(v => (v.performance?.mape || 0).toFixed(2) + '%'),
                textposition: 'outside',
                hovertemplate: '<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
            };
            
            const rmseTrace = {
                x: versions.map(v => v.version),
                y: versions.map(v => v.performance?.rmse || 0),
                type: 'bar',
                name: 'RMSE',
                marker: { 
                    color: '#fb923c',
                    line: {
                        color: '#f97316',
                        width: 2
                    }
                },
                text: versions.map(v => (v.performance?.rmse || 0).toFixed(2)),
                textposition: 'outside',
                yaxis: 'y2',
                hovertemplate: '<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>'
            };
            
            const layout = {
                title: {
                    text: 'Version Performance',
                    font: { color: '#1e293b', size: 17, family: 'system-ui', weight: 800 }
                },
                paper_bgcolor: 'rgba(0, 0, 0, 0)',
                plot_bgcolor: 'rgba(241, 245, 249, 0.5)',
                xaxis: {
                    title: 'Version',
                    gridcolor: '#e2e8f0',
                    color: '#475569',
                    tickangle: -45
                },
                yaxis: {
                    title: 'MAPE (%)',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                yaxis2: {
                    title: 'RMSE',
                    overlaying: 'y',
                    side: 'right',
                    color: '#475569',
                    gridcolor: '#e2e8f0'
                },
                font: { family: 'system-ui, sans-serif', color: '#1e293b' },
                margin: { t: 50, b: 90, l: 60, r: 60 },
                barmode: 'group',
                bargap: 0.25,
                bargroupgap: 0.12,
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1.15,
                    orientation: 'h',
                    bgcolor: 'rgba(255, 255, 255, 0.9)',
                    bordercolor: '#cbd5e1',
                    borderwidth: 2
                }
            };
            
            Plotly.newPlot('versionComparisonChart', [mapeTrace, rmseTrace], layout, {responsive: true});
        } else {
            document.getElementById('versionComparisonChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No version comparison data available</p>';
        }
    } catch (error) {
        console.error('Error loading version comparison:', error);
    }
}

// Show empty state
function showEmptyState() {
    // Clear all columns
    document.getElementById('excellentModels').innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 20px;">No models</p>';
    document.getElementById('goodModels').innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 20px;">No models</p>';
    document.getElementById('fairModels').innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px 20px;">No models</p>';
    document.getElementById('poorModels').innerHTML = `
        <div style="text-align: center; padding: 30px 20px;">
            <div style="font-size: 3em; margin-bottom: 15px;">📭</div>
            <p style="color: var(--text-muted); font-weight: 600;">No models found</p>
            <a href="/" class="btn-primary" style="display: inline-block; margin-top: 15px; text-decoration: none; padding: 10px 20px; font-size: 0.9em;">
                Generate Forecasts
            </a>
        </div>
    `;
}

// Show error state
function showErrorState(message) {
    document.getElementById('excellentModels').innerHTML = '';
    document.getElementById('goodModels').innerHTML = '';
    document.getElementById('fairModels').innerHTML = '';
    document.getElementById('poorModels').innerHTML = `
        <div style="text-align: center; padding: 30px 20px;">
            <div style="font-size: 3em; margin-bottom: 15px; color: var(--error);">⚠️</div>
            <p style="color: var(--text-secondary); font-weight: 600; margin-bottom: 15px;">Error Loading</p>
            <p style="color: var(--text-muted); font-size: 0.85em; margin-bottom: 15px;">${message}</p>
            <button onclick="loadTrainedModels()" class="btn-primary" style="padding: 10px 20px; font-size: 0.9em;">
                Retry
            </button>
        </div>
    `;
}

// Show loading state
function addLoadingState() {
    const loadingHTML = '<div style="text-align: center; padding: 40px 20px;"><div style="font-size: 2.5em;">⏳</div><p style="color: var(--text-muted); font-weight: 600; margin-top: 10px;">Loading...</p></div>';
    document.getElementById('excellentModels').innerHTML = loadingHTML;
    document.getElementById('goodModels').innerHTML = loadingHTML;
    document.getElementById('fairModels').innerHTML = loadingHTML;
    document.getElementById('poorModels').innerHTML = loadingHTML;
}
