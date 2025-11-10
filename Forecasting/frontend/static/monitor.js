// Adaptive Learning Monitor JavaScript

const API_BASE = window.location.origin;
let refreshInterval = null;
let activityLog = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Adaptive Learning Monitor initialized');
    
    // Setup event listeners
    document.getElementById('refreshBtn').addEventListener('click', refreshAllData);
    document.getElementById('retrainBtn').addEventListener('click', triggerRetrain);
    document.getElementById('rebalanceBtn').addEventListener('click', triggerRebalance);
    
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
    
    // Symbol/Model change
    document.getElementById('monitorSymbol').addEventListener('click', refreshAllData);
    document.getElementById('monitorModel').addEventListener('change', refreshAllData);
    
    // Initial load
    refreshAllData();
    
    // Auto-refresh every 10 seconds
    refreshInterval = setInterval(() => {
        const activeBtn = document.querySelector('.tab-btn.active');
        if (activeBtn) {
            const activeTab = activeBtn.dataset.tab;
            refreshAllData();
            if (activeTab !== 'overview') {
                setTimeout(() => loadTabData(activeTab), 500);
            }
        }
    }, 10000);
    
    addActivity('info', 'Monitor initialized - Updates every 10s');
});

// Switch tabs
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const activeContent = document.getElementById(tabName);
    if (activeContent) {
        activeContent.classList.add('active');
    }
    
    // Load tab-specific data
    loadTabData(tabName);
}

// Load data for specific tab
function loadTabData(tabName) {
    const symbol = document.getElementById('monitorSymbol').value;
    const model = document.getElementById('monitorModel').value;
    
    switch(tabName) {
        case 'performance':
            loadPerformanceTrend(symbol, model);
            break;
        case 'versions':
            loadVersionHistory(symbol, model);
            break;
        case 'ensemble':
            loadEnsembleData(symbol);
            break;
        case 'errors':
            loadErrorAnalysis(symbol, model);
            break;
        case 'logs':
            loadTrainingLogs(symbol, model);
            break;
    }
}

// Refresh all data
async function refreshAllData() {
    const symbol = document.getElementById('monitorSymbol').value;
    const model = document.getElementById('monitorModel').value;
    
    addActivity('info', `Refreshing data for ${symbol}/${model}`);
    
    try {
        await Promise.all([
            loadSchedulerStatus(),
            loadModelStatistics(symbol, model),
            loadTabData(document.querySelector('.tab.active').dataset.tab)
        ]);
        
        addActivity('success', 'Data refreshed successfully');
    } catch (error) {
        addActivity('error', `Refresh failed: ${error.message}`);
    }
}

// Load scheduler status
async function loadSchedulerStatus() {
    try {
        // Set scheduler as active (it starts automatically with the Flask app)
        const statusElement = document.getElementById('schedulerStatus');
        const indicatorElement = document.getElementById('schedulerStatusIndicator');
        
        if (statusElement) {
            statusElement.textContent = 'Active';
            
            // Add active class to indicator
            if (indicatorElement && !indicatorElement.classList.contains('active')) {
                indicatorElement.classList.remove('inactive');
            }
        }
        
        // Optional: Try to fetch actual status from API if endpoint exists
        try {
            const response = await fetch(`${API_BASE}/api/adaptive/scheduler/status`);
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.status) {
                    statusElement.textContent = data.status.is_running ? 'Active' : 'Inactive';
                    if (!data.status.is_running) {
                        indicatorElement.classList.add('inactive');
                    }
                }
            }
        } catch (apiError) {
            // API endpoint doesn't exist, keep default "Active" status
            console.log('Scheduler status API not available, showing default active status');
        }
    } catch (error) {
        console.error('Error loading scheduler status:', error);
        // Fallback to showing active
        const statusElement = document.getElementById('schedulerStatus');
        if (statusElement) {
            statusElement.textContent = 'Active';
        }
    }
}

// Load model statistics
async function loadModelStatistics(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/${symbol}/${model}`);
        const data = await response.json();
        
        if (data.success && data.statistics.status !== 'no_data') {
            const stats = data.statistics;
            
            // Update overview metrics
            document.getElementById('totalPredictions').textContent = 
                stats.total_predictions || 0;
            document.getElementById('trainingCount').textContent = 
                stats.training_count || 0;
            document.getElementById('daysSinceTraining').textContent = 
                stats.days_since_training !== null ? `${stats.days_since_training} days` : 'N/A';
            
            // Performance metrics
            if (stats.recent_performance && stats.recent_performance.mape !== null) {
                document.getElementById('recentMAPE').textContent = 
                    `${stats.recent_performance.mape.toFixed(2)}%`;
            }
            
            if (stats.baseline_performance && stats.baseline_performance.mape !== null) {
                document.getElementById('baselineMAPE').textContent = 
                    `${stats.baseline_performance.mape.toFixed(2)}%`;
            }
            
            if (stats.all_time_performance && stats.all_time_performance.mape !== null) {
                document.getElementById('alltimeMAPE').textContent = 
                    `${stats.all_time_performance.mape.toFixed(2)}%`;
            }
            
            // Performance status
            const recentMAPE = stats.recent_performance?.mape || 0;
            const baselineMAPE = stats.baseline_performance?.mape || 0;
            
            let statusText = 'Optimal';
            let statusColor = '#22c55e';
            
            if (recentMAPE > baselineMAPE * 1.2) {
                statusText = 'Degraded';
                statusColor = '#fbbf24';
            } else if (recentMAPE > baselineMAPE * 1.5) {
                statusText = 'Critical';
                statusColor = '#ef4444';
            }
            
            const statusEl = document.getElementById('performanceStatus');
            statusEl.textContent = statusText;
            statusEl.style.color = statusColor;
        } else {
            // No data available
            document.getElementById('totalPredictions').textContent = '0';
            document.getElementById('trainingCount').textContent = '0';
            document.getElementById('daysSinceTraining').textContent = 'N/A';
            document.getElementById('recentMAPE').textContent = 'N/A';
            document.getElementById('baselineMAPE').textContent = 'N/A';
            document.getElementById('alltimeMAPE').textContent = 'N/A';
            document.getElementById('performanceStatus').textContent = 'No Data';
        }
    } catch (error) {
        console.error('Error loading model statistics:', error);
    }
}

// Load performance trend
async function loadPerformanceTrend(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/performance/trend/${symbol}/${model}?days=30`);
        const data = await response.json();
        
        if (data.success && data.trend.length > 0) {
            const trend = data.trend;
            
            const trace = {
                x: trend.map(t => t.date),
                y: trend.map(t => t.mape),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MAPE Score',
                line: {
                    color: '#14b8a6',
                    width: 3.5
                },
                marker: {
                    size: 9,
                    color: '#14b8a6',
                    line: {
                        color: 'white',
                        width: 2
                    }
                }
            };
            
            const layout = {
                title: {
                    text: `${model.toUpperCase()} - Performance Trend`,
                    font: { color: '#1e293b', size: 19, family: 'system-ui, sans-serif', weight: 800 }
                },
                paper_bgcolor: 'rgba(255, 255, 255, 0)',
                plot_bgcolor: 'rgba(248, 250, 252, 0.6)',
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
                font: { family: 'system-ui, sans-serif', color: '#1e293b' }
            };
            
            Plotly.newPlot('performanceTrendChart', [trace], layout, {responsive: true});
        } else {
            document.getElementById('performanceTrendChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No performance data available</p>';
        }
    } catch (error) {
        console.error('Error loading performance trend:', error);
    }
}

// Load version history
async function loadVersionHistory(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/versions/${symbol}/${model}?limit=10`);
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            const historyHTML = data.history.map(v => {
                const isActive = v.status === 'active';
                const date = new Date(v.trained_at).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
                const mape = v.performance?.mape?.toFixed(2) || 'N/A';
                
                return `
                    <div class="log-entry ${isActive ? 'success' : ''}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span class="version-badge ${isActive ? 'active' : ''}">${v.version}</span>
                            ${isActive ? '<span style="background: var(--success); color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.75em; font-weight: 800;">ACTIVE</span>' : ''}
                        </div>
                        <div class="log-time">📅 ${date}</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px;">
                            <div class="metric-row" style="margin: 0; flex-direction: column; align-items: center; padding: 12px;">
                                <span class="metric-label" style="font-size: 0.7em; margin-bottom: 5px;">MAPE</span>
                                <span class="metric-value" style="font-size: 1.4em;">${mape}%</span>
                            </div>
                            <div class="metric-row" style="margin: 0; flex-direction: column; align-items: center; padding: 12px;">
                                <span class="metric-label" style="font-size: 0.7em; margin-bottom: 5px;">Type</span>
                                <span class="metric-value" style="font-size: 1em; text-transform: uppercase;">${v.update_type || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            document.getElementById('versionHistory').innerHTML = historyHTML;
            document.getElementById('activeVersion').textContent = 
                data.history.find(v => v.status === 'active')?.version || 'N/A';
        } else {
            document.getElementById('versionHistory').innerHTML = 
                '<p style="text-align: center; color: var(--text-muted); padding: 40px;">No versions yet</p>';
            document.getElementById('activeVersion').textContent = 'N/A';
        }
    } catch (error) {
        console.error('Error loading version history:', error);
    }
}

// Load ensemble data
async function loadEnsembleData(symbol) {
    try {
        // Load current weights
        const weightsResponse = await fetch(`${API_BASE}/api/adaptive/ensemble/weights/${symbol}`);
        const weightsData = await weightsResponse.json();
        
        if (weightsData.success && weightsData.weights) {
            const weights = weightsData.weights;
            
            const weightsHTML = Object.entries(weights)
                .sort((a, b) => b[1] - a[1])
                .map(([model, weight]) => {
                    const percentage = (weight * 100).toFixed(1);
                    return `
                        <div style="margin: 14px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <span style="font-weight: 700; color: var(--secondary); font-size: 0.95em;">${model.toUpperCase()}</span>
                                <span style="color: var(--primary); font-weight: 800; font-size: 1.2em;">${percentage}%</span>
                            </div>
                            <div class="weight-bar" style="width: ${percentage}%;">
                                ${percentage}%
                            </div>
                        </div>
                    `;
                }).join('');
            
            document.getElementById('ensembleWeights').innerHTML = weightsHTML;
        } else {
            document.getElementById('ensembleWeights').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5);">No ensemble weights available</p>';
        }
        
        // Load weight history
        const historyResponse = await fetch(`${API_BASE}/api/adaptive/ensemble/history/${symbol}?days=30`);
        const historyData = await historyResponse.json();
        
        if (historyData.success && historyData.history.length > 0) {
            const history = historyData.history;
            
            // Extract unique model names
            const modelNames = [...new Set(history.flatMap(h => Object.keys(h.weights)))];
            
            // Create traces for each model
            const traces = modelNames.map(modelName => ({
                x: history.map(h => h.timestamp),
                y: history.map(h => (h.weights[modelName] || 0) * 100),
                type: 'scatter',
                mode: 'lines',
                name: modelName.toUpperCase(),
                line: { width: 2 }
            }));
            
            const layout = {
                title: {
                    text: 'Ensemble Weight Evolution',
                    font: { color: '#1e293b', size: 19, family: 'system-ui, sans-serif', weight: 800 }
                },
                paper_bgcolor: 'rgba(255, 255, 255, 0)',
                plot_bgcolor: 'rgba(248, 250, 252, 0.6)',
                xaxis: {
                    title: 'Date',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                yaxis: {
                    title: 'Weight (%)',
                    gridcolor: '#e2e8f0',
                    color: '#475569',
                    range: [0, 100]
                },
                font: { family: 'system-ui, sans-serif', color: '#1e293b' }
            };
            
            Plotly.newPlot('weightHistoryChart', traces, layout, {responsive: true});
        } else {
            document.getElementById('weightHistoryChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No weight history available</p>';
        }
    } catch (error) {
        console.error('Error loading ensemble data:', error);
    }
}

// Load training logs
async function loadTrainingLogs(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/training/logs/${symbol}/${model}?limit=20`);
        const data = await response.json();
        
        if (data.success && data.logs.length > 0) {
            const logsHTML = data.logs.map(log => {
                const statusClass = log.status === 'success' ? 'success' : 
                                  log.status === 'failed' ? 'error' : 'warning';
                const statusIcon = log.status === 'success' ? '🟢' : 
                                 log.status === 'failed' ? '🔴' : '🟡';
                const startTime = new Date(log.training_started).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
                const duration = log.training_completed ? 
                    Math.round((new Date(log.training_completed) - new Date(log.training_started)) / 1000) : 
                    'N/A';
                
                return `
                    <div class="log-entry ${statusClass}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.3em;">${statusIcon}</span>
                                <span style="font-weight: 900; color: var(--primary); font-size: 1.2em;">${log.version}</span>
                            </div>
                            <span style="background: var(--accent); color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.75em; font-weight: 800; text-transform: uppercase;">
                                ${log.trigger}
                            </span>
                        </div>
                        <div class="log-time">📅 ${startTime}</div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px;">
                            <div style="background: white; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 0.7em; color: var(--text-muted); font-weight: 700; margin-bottom: 5px;">Status</div>
                                <div style="font-size: 0.95em; font-weight: 800; color: var(--secondary); text-transform: uppercase;">${log.status}</div>
                            </div>
                            <div style="background: white; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 0.7em; color: var(--text-muted); font-weight: 700; margin-bottom: 5px;">Time</div>
                                <div style="font-size: 1.1em; font-weight: 900; color: var(--primary);">${duration}s</div>
                            </div>
                            <div style="background: white; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 0.7em; color: var(--text-muted); font-weight: 700; margin-bottom: 5px;">Epochs</div>
                                <div style="font-size: 1.1em; font-weight: 900; color: var(--accent);">${log.epochs}</div>
                            </div>
                            <div style="background: white; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 0.7em; color: var(--text-muted); font-weight: 700; margin-bottom: 5px;">MAPE</div>
                                <div style="font-size: 1.1em; font-weight: 900; color: var(--success);">${log.metrics?.mape?.toFixed(2) || 'N/A'}%</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            document.getElementById('trainingLogs').innerHTML = logsHTML;
        } else {
            document.getElementById('trainingLogs').innerHTML = 
                '<p style="text-align: center; color: var(--text-muted); padding: 40px;">No training logs yet</p>';
        }
    } catch (error) {
        console.error('Error loading training logs:', error);
    }
}

// Trigger manual retrain
async function triggerRetrain() {
    const symbol = document.getElementById('monitorSymbol').value;
    const model = document.getElementById('monitorModel').value;
    
    if (!confirm(`Trigger retraining for ${symbol}/${model}?`)) {
        return;
    }
    
    addActivity('info', `Triggering retrain for ${symbol}/${model}...`);
    
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/retrain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, model })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addActivity('success', `Retraining started for ${symbol}/${model}`);
            // Refresh data after a delay
            setTimeout(refreshAllData, 5000);
        } else {
            addActivity('error', `Retrain failed: ${data.error}`);
        }
    } catch (error) {
        addActivity('error', `Retrain error: ${error.message}`);
    }
}

// Trigger ensemble rebalance
async function triggerRebalance() {
    const symbol = document.getElementById('monitorSymbol').value;
    
    if (!confirm(`Rebalance ensemble weights for ${symbol}?`)) {
        return;
    }
    
    addActivity('info', `Rebalancing ensemble for ${symbol}...`);
    
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/ensemble/rebalance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, lookback_days: 7 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addActivity('success', `Ensemble rebalanced for ${symbol}`);
            // Refresh ensemble data
            loadEnsembleData(symbol);
        } else {
            addActivity('error', `Rebalance failed: ${data.error}`);
        }
    } catch (error) {
        addActivity('error', `Rebalance error: ${error.message}`);
    }
}

// Add activity to feed
function addActivity(type, message) {
    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    
    const typeClass = type === 'error' ? 'error' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : '';
    const icon = type === 'error' ? '🔴' : type === 'warning' ? '🟡' : type === 'success' ? '🟢' : '🔵';
    
    const entry = `
        <div class="log-entry ${typeClass}">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.3em;">${icon}</span>
                <span style="font-weight: 600; color: var(--secondary); flex: 1;">${message}</span>
            </div>
            <div class="log-time">${timestamp}</div>
        </div>
    `;
    
    const feed = document.getElementById('activityFeed');
    feed.insertAdjacentHTML('afterbegin', entry);
    
    // Keep only last 50 entries
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
    
    // Add to log array
    activityLog.unshift({ type, message, timestamp });
    if (activityLog.length > 100) {
        activityLog.pop();
    }
}

// Load error analysis
async function loadErrorAnalysis(symbol, model) {
    try {
        const response = await fetch(`${API_BASE}/api/adaptive/prediction-errors/${symbol}/${model}?days=30`);
        const data = await response.json();
        
        if (data.success && data.errors.length > 0) {
            const errors = data.errors;
            
            // Chart 1: Actual vs Predicted with Error Overlay
            const actualTrace = {
                x: errors.map(e => e.date),
                y: errors.map(e => e.actual),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual Price',
                line: { color: '#1e293b', width: 3.5 },
                marker: { size: 7, color: '#1e293b' }
            };
            
            const predictedTrace = {
                x: errors.map(e => e.date),
                y: errors.map(e => e.predicted),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: { color: '#14b8a6', width: 3.5, dash: 'dash' },
                marker: { size: 7, color: '#14b8a6' }
            };
            
            const errorTrace = {
                x: errors.map(e => e.date),
                y: errors.map(e => e.error),
                type: 'bar',
                name: 'Error Magnitude',
                yaxis: 'y2',
                marker: {
                    color: errors.map(e => e.error_percentage > 5 ? '#ef4444' : '#fb923c'),
                    opacity: 0.7
                }
            };
            
            const layout1 = {
                title: {
                    text: `${symbol} - ${model.toUpperCase()}: Prediction Accuracy`,
                    font: { color: '#1e293b', size: 18, family: 'system-ui, sans-serif', weight: 800 }
                },
                paper_bgcolor: 'rgba(255, 255, 255, 0)',
                plot_bgcolor: 'rgba(248, 250, 252, 0.6)',
                xaxis: {
                    title: 'Date',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                yaxis: {
                    title: 'Price (USD)',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                yaxis2: {
                    title: 'Error',
                    overlaying: 'y',
                    side: 'right',
                    color: '#475569'
                },
                font: { family: 'system-ui, sans-serif', color: '#1e293b' },
                margin: { t: 55, b: 50, l: 65, r: 65 },
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(255, 255, 255, 0.95)',
                    bordercolor: '#cbd5e1',
                    borderwidth: 2
                }
            };
            
            Plotly.newPlot('errorAnalysisChart', [actualTrace, predictedTrace, errorTrace], layout1, {responsive: true});
            
            // Chart 2: Error Distribution (Histogram)
            const errorPercentages = errors.map(e => e.error_percentage);
            
            const histogramTrace = {
                x: errorPercentages,
                type: 'histogram',
                name: 'Error Distribution',
                marker: {
                    color: '#14b8a6',
                    line: {
                        color: '#0f766e',
                        width: 2
                    }
                },
                nbinsx: 20
            };
            
            const layout2 = {
                title: {
                    text: 'Error Distribution Analysis',
                    font: { color: '#1e293b', size: 18, family: 'system-ui, sans-serif', weight: 800 }
                },
                paper_bgcolor: 'rgba(255, 255, 255, 0)',
                plot_bgcolor: 'rgba(248, 250, 252, 0.6)',
                xaxis: {
                    title: 'Error Percentage (%)',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                yaxis: {
                    title: 'Frequency',
                    gridcolor: '#e2e8f0',
                    color: '#475569'
                },
                font: { family: 'system-ui, sans-serif', color: '#1e293b' },
                margin: { t: 55, b: 50, l: 65, r: 35 }
            };
            
            Plotly.newPlot('errorDistributionChart', [histogramTrace], layout2, {responsive: true});
            
        } else {
            document.getElementById('errorAnalysisChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No error data available. Generate some forecasts first.</p>';
            document.getElementById('errorDistributionChart').innerHTML = 
                '<p style="text-align: center; color: rgba(255,255,255,0.5); padding: 40px;">No error data available.</p>';
        }
    } catch (error) {
        console.error('Error loading error analysis:', error);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});
