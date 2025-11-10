// API base URL
const API_BASE = window.location.origin;

// Toast Notification System
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = {
        success: '🟢',
        error: '🔴',
        warning: '🟡',
        info: '🔵'
    };
    
    toast.innerHTML = `
        <span style="font-size: 1.3em; margin-right: 12px;">${icons[type] || icons.info}</span>
        <span style="flex: 1; font-weight: 600;">${message}</span>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; color: inherit; cursor: pointer; font-size: 1.2em; padding: 0 5px;">×</button>
    `;
    
    document.body.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Event listeners
document.getElementById('forecastBtn').addEventListener('click', generateForecast);
document.getElementById('compareBtn').addEventListener('click', compareModels);
document.getElementById('symbol').addEventListener('change', handleSymbolChange);

// Handle custom symbol input visibility
function handleSymbolChange() {
    const symbolSelect = document.getElementById('symbol');
    const customSymbolGroup = document.getElementById('customSymbolGroup');
    
    if (symbolSelect.value === 'CUSTOM') {
        customSymbolGroup.style.display = 'flex';
        document.getElementById('customSymbol').focus();
    } else {
        customSymbolGroup.style.display = 'none';
    }
}

// Get the actual symbol to use
function getSelectedSymbol() {
    const symbolSelect = document.getElementById('symbol');
    
    if (symbolSelect.value === 'CUSTOM') {
        const customSymbol = document.getElementById('customSymbol').value.trim().toUpperCase();
        if (!customSymbol) {
            showToast('Please enter a custom symbol (e.g., TSLA, NFLX, DOGE-USD)', 'warning');
            return null;
        }
        return customSymbol;
    }
    
    return symbolSelect.value;
}

async function generateForecast() {
    const symbol = getSelectedSymbol();
    if (!symbol) return;
    
    const model = document.getElementById('model').value;
    const horizon = document.getElementById('horizon').value;
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('metrics').style.display = 'none';
    document.getElementById('comparison').style.display = 'none';
    document.getElementById('chart').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol, model, horizon })
        });
        
        if (!response.ok) {
            const text = await response.text();
            console.error('Server response:', text);
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayMetrics(data.metrics, data.latest_data_time, data.prediction_time);
            displayChart(data.historical_data, data.predictions, symbol);
            showToast('Forecast generated successfully!', 'success');
        } else {
            showToast('Error: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Forecast error:', error);
        showToast('Error generating forecast: ' + error.message, 'error');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

async function compareModels() {
    const symbol = getSelectedSymbol();
    if (!symbol) return;
    
    const horizon = document.getElementById('horizon').value;
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('metrics').style.display = 'none';
    document.getElementById('chart').innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/compare_models`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol, horizon })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayComparison(data.results);
            showToast('Models compared successfully!', 'success');
        } else {
            showToast('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error comparing models: ' + error.message, 'error');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayMetrics(metrics, latestDataTime, predictionTime) {
    document.getElementById('rmse').textContent = metrics.rmse.toFixed(2);
    document.getElementById('mae').textContent = metrics.mae.toFixed(2);
    document.getElementById('mape').textContent = metrics.mape.toFixed(2) + '%';
    document.getElementById('model-name').textContent = metrics.model;
    
    // Add accuracy interpretation with new color scheme
    const mape = metrics.mape;
    let accuracyText = '';
    let accuracyColor = '';
    
    if (mape < 5) {
        accuracyText = 'Excellent';
        accuracyColor = '#14b8a6';
    } else if (mape < 10) {
        accuracyText = 'Good';
        accuracyColor = '#22c55e';
    } else if (mape < 20) {
        accuracyText = 'Fair';
        accuracyColor = '#fbbf24';
    } else {
        accuracyText = 'Poor';
        accuracyColor = '#ef4444';
    }
    
    document.getElementById('accuracy-rating').textContent = accuracyText;
    document.getElementById('accuracy-rating').style.color = accuracyColor;
    
    // Show data freshness info
    if (latestDataTime) {
        const dataDate = new Date(latestDataTime);
        const predDate = new Date(predictionTime);
        document.getElementById('data-time').textContent = dataDate.toLocaleString();
        document.getElementById('prediction-time').textContent = predDate.toLocaleString();
    }
    
    document.getElementById('metrics').style.display = 'block';
}

function displayChart(historicalData, predictions, symbol) {
    // Prepare historical data
    const historicalDates = historicalData.map(d => d.date);
    const historicalClose = historicalData.map(d => d.close);
    const historicalOpen = historicalData.map(d => d.open);
    const historicalHigh = historicalData.map(d => d.high);
    const historicalLow = historicalData.map(d => d.low);
    
    // Prepare prediction data
    const predictionDates = predictions.map(p => p.date);
    const predictionValues = predictions.map(p => p.predicted_close);
    
    // Create candlestick trace for historical data
    const candlestickTrace = {
        x: historicalDates,
        close: historicalClose,
        high: historicalHigh,
        low: historicalLow,
        open: historicalOpen,
        type: 'candlestick',
        name: 'Market Data',
        increasing: { line: { color: '#22c55e' } },
        decreasing: { line: { color: '#ef4444' } }
    };
    
    // Create line trace for predictions
    const predictionTrace = {
        x: predictionDates,
        y: predictionValues,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'AI Forecast',
        line: {
            color: '#14b8a6',
            width: 4,
            dash: 'dash'
        },
        marker: {
            size: 11,
            color: '#14b8a6',
            line: {
                color: 'white',
                width: 2
            },
            symbol: 'diamond'
        }
    };
    
    // Create error overlay trace (shows prediction errors)
    // Calculate error bars based on historical prediction accuracy
    const lastActualPrice = historicalClose[historicalClose.length - 1];
    const errorPercentage = 0.02; // 2% error margin (can be dynamic based on MAPE)
    
    const errorBandUpper = predictionValues.map(v => v * (1 + errorPercentage));
    const errorBandLower = predictionValues.map(v => v * (1 - errorPercentage));
    
    const errorBandTrace = {
        x: predictionDates.concat(predictionDates.slice().reverse()),
        y: errorBandUpper.concat(errorBandLower.slice().reverse()),
        fill: 'toself',
        fillcolor: 'rgba(20, 184, 166, 0.12)',
        line: { color: 'transparent' },
        name: 'Confidence Band',
        type: 'scatter',
        showlegend: true,
        hoverinfo: 'skip'
    };
    
    // Layout with clean professional theme
    const layout = {
        title: {
            text: `${symbol} - Price Forecast`,
            font: { 
                size: 22,
                color: '#1e293b',
                family: 'system-ui, sans-serif',
                weight: 800
            }
        },
        paper_bgcolor: 'rgba(255, 255, 255, 0)',
        plot_bgcolor: 'rgba(248, 250, 252, 0.6)',
        xaxis: {
            title: {
                text: 'Date',
                font: { color: '#475569', weight: 700 }
            },
            gridcolor: '#e2e8f0',
            color: '#475569',
            rangeslider: { visible: false }
        },
        yaxis: {
            title: {
                text: 'Price (USD)',
                font: { color: '#475569', weight: 700 }
            },
            gridcolor: '#e2e8f0',
            color: '#475569'
        },
        hovermode: 'x unified',
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(255, 255, 255, 0.95)',
            bordercolor: '#cbd5e1',
            borderwidth: 2,
            font: { color: '#1e293b', weight: 600 }
        },
        font: {
            family: 'system-ui, sans-serif',
            color: '#1e293b'
        }
    };
    
    // Plot with dark theme config
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };
    
    Plotly.newPlot('chart', [candlestickTrace, errorBandTrace, predictionTrace], layout, config);
}

function displayComparison(results) {
    const tbody = document.querySelector('#comparisonTable tbody');
    tbody.innerHTML = '';
    
    // Sort by MAPE (lower is better)
    const sortedModels = Object.entries(results).sort((a, b) => 
        a[1].metrics.mape - b[1].metrics.mape
    );
    
    sortedModels.forEach(([modelName, data], index) => {
        const row = tbody.insertRow();
        const rank = index + 1;
        const rankClass = rank <= 3 ? `rank-${rank}` : '';
        
        row.innerHTML = `
            <td style="font-weight: 700; color: var(--secondary); font-size: 1.05em;">${modelName.toUpperCase()}</td>
            <td style="text-align: center; font-weight: 600;">${data.metrics.rmse.toFixed(2)}</td>
            <td style="text-align: center; font-weight: 600;">${data.metrics.mae.toFixed(2)}</td>
            <td style="text-align: center; font-weight: 700; color: var(--primary);">${data.metrics.mape.toFixed(2)}%</td>
            <td style="text-align: center;">
                <span class="rank-badge ${rankClass}">${rank}</span>
            </td>
        `;
    });
    
    document.getElementById('comparison').style.display = 'block';
}

// Initialize on page load
window.addEventListener('load', () => {
    console.log('Stock Forecasting App Loaded');
});
