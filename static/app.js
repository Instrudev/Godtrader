let chart;
let candlestickSeries;
let currentAssetId = null;
let liveSocket = null;
let lastHistoricalTime = 0;
let isBotRunning = false;

// Initialize Chart
function initChart() {
    const chartContainer = document.getElementById('chart-container');
    chart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { color: '#0d0f14' },
            textColor: '#94a3b8',
        },
        grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
        },
        timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    candlestickSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });

    // Resize handler
    window.addEventListener('resize', () => {
        chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
    });
}

// Fetch Assets
async function loadAssets() {
    const assetList = document.getElementById('asset-list');
    assetList.innerHTML = '<li class="loading">Cargando activos...</li>';
    try {
        const response = await fetch('/assets');
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            assetList.innerHTML = `<li class="loading" style="color:#ef4444">Error ${response.status}: ${err.detail || 'No se pudieron cargar los activos'}</li>`;
            return;
        }
        const data = await response.json();

        if (!data.assets || data.assets.length === 0) {
            assetList.innerHTML = '<li class="loading">No se encontraron activos</li>';
            return;
        }

        assetList.innerHTML = '';

        // Sort: open assets first, then alphabetically
        const sorted = [...data.assets].sort((a, b) => {
            if (a.open !== b.open) return a.open ? -1 : 1;
            return a.name.localeCompare(b.name);
        });

        sorted.forEach(asset => {
            const li = document.createElement('li');
            if (!asset.open) li.classList.add('closed');
            li.innerHTML = `
                <span class="asset-name">${asset.name}</span>
                <div class="asset-meta">
                    <span class="asset-type">${asset.type}</span>
                    <span class="asset-status ${asset.open ? 'open' : 'closed'}">${asset.open ? '●' : '○'}</span>
                </div>
            `;
            li.onclick = () => selectAsset(asset);
            assetList.appendChild(li);
        });

        // Initialize search
        const searchInput = document.getElementById('asset-search');
        searchInput.oninput = (e) => {
            const term = e.target.value.toLowerCase();
            const items = assetList.getElementsByTagName('li');
            Array.from(items).forEach(item => {
                const name = item.querySelector('.asset-name').textContent.toLowerCase();
                item.style.display = name.includes(term) ? 'flex' : 'none';
            });
        };
    } catch (error) {
        console.error('Error loading assets:', error);
        assetList.innerHTML = `<li class="loading" style="color:#ef4444">Error: ${error.message}</li>`;
    }
}

// Select Asset and Load data
async function selectAsset(asset) {
    currentAssetId = asset.id;
    document.getElementById('current-asset-name').textContent = asset.name;
    document.getElementById('current-asset-type').textContent = asset.type;

    // Update active class in list
    const items = document.querySelectorAll('#asset-list li');
    items.forEach(item => {
        if (item.querySelector('.asset-name').textContent === asset.name) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    const botBtn = document.getElementById('start-bot-btn');
    if (botBtn && !isBotRunning) {
        botBtn.disabled = false;
    }

    loadCandles();
}

// Fetch historical candles then start live stream
async function loadCandles() {
    if (!currentAssetId) return;

    const interval = document.getElementById('interval-select').value;

    // Stop any existing WebSocket
    stopLiveStream();

    try {
        const response = await fetch(`/candles/${currentAssetId}?interval=${interval}`);
        const data = await response.json();

        if (data.candles && data.candles.length > 0) {
            const sortedData = data.candles.sort((a, b) => a.time - b.time);
            candlestickSeries.setData(sortedData);
            lastHistoricalTime = sortedData[sortedData.length - 1].time;
            chart.timeScale().scrollToRealTime();
        }
    } catch (error) {
        console.error('Error loading candles:', error);
    }

    // Start live stream via WebSocket
    startLiveStream(currentAssetId, interval);
}

function startLiveStream(assetId, interval) {
    const wsUrl = `ws://${location.host}/ws/${encodeURIComponent(assetId)}?interval=${interval}`;
    liveSocket = new WebSocket(wsUrl);

    const indicator = document.getElementById('live-indicator');

    liveSocket.onopen = () => {
        if (indicator) indicator.style.display = 'flex';
    };

    liveSocket.onmessage = (event) => {
        try {
            const candle = JSON.parse(event.data);
            if (candle.error) {
                console.error('Stream error:', candle.error);
                return;
            }
            // Only update if the candle time is >= the last historical candle
            // (LightweightCharts throws if time goes backward)
            if (candle.time >= lastHistoricalTime) {
                candlestickSeries.update(candle);
                lastHistoricalTime = candle.time;
            }
        } catch (e) {
            console.error('Error processing candle:', e);
        }
    };

    liveSocket.onclose = () => {
        if (indicator) indicator.style.display = 'none';
    };

    liveSocket.onerror = (e) => {
        console.error('WebSocket error:', e);
        if (indicator) indicator.style.display = 'none';
    };
}

function stopLiveStream() {
    if (liveSocket) {
        liveSocket.close();
        liveSocket = null;
    }
}

// Login Handler
document.getElementById('login-form').onsubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const btn = document.getElementById('login-btn');
    const errorMsg = document.getElementById('login-error');

    btn.textContent = 'Conectando...';
    btn.disabled = true;
    errorMsg.textContent = '';

    try {
        const response = await fetch('/connect', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            document.getElementById('login-overlay').style.display = 'none';
            try { initChart(); } catch(e) { console.error('Chart init error:', e); }
            loadAssets();
            checkBotStatus();
            loadTradeHistory();
        } else {
            const data = await response.json();
            errorMsg.textContent = data.detail || 'Error de conexión';
        }
    } catch (error) {
        errorMsg.textContent = 'Error al contactar con el servidor';
    } finally {
        btn.textContent = 'Conectar';
        btn.disabled = false;
    }
};

// Bot Controls
const botBtn = document.getElementById('start-bot-btn');
if (botBtn) {
    botBtn.onclick = async () => {
        if (!currentAssetId) return;

        if (isBotRunning) {
            // Stop bot
            botBtn.disabled = true;
            try {
                const res = await fetch('/bot/stop', { method: 'POST' });
                if (res.ok) {
                    isBotRunning = false;
                    botBtn.textContent = 'Iniciar Bot';
                    botBtn.classList.remove('running');
                } else {
                    alert('Error al detener bot');
                }
            } catch (e) {
                alert('Error de conexión con el servidor');
            } finally {
                botBtn.disabled = false;
                checkBotStatus();
            }
        } else {
            // Start bot
            botBtn.disabled = true;
            botBtn.textContent = 'Iniciando...';
            try {
                const res = await fetch('/bot/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        asset: currentAssetId,
                        asset_type: 'binary',
                        amount: 1000.0
                    })
                });
                
                if (res.ok) {
                    isBotRunning = true;
                    botBtn.textContent = 'Detener Bot';
                    botBtn.classList.add('running');
                } else {
                    const error = await res.json();
                    alert('Status: ' + error.detail || 'Error al iniciar bot');
                    botBtn.textContent = 'Iniciar Bot';
                }
            } catch (e) {
                alert('Error de conexión con el servidor');
                botBtn.textContent = 'Iniciar Bot';
            } finally {
                botBtn.disabled = false;
                checkBotStatus();
            }
        }
    };
}

// Check Bot Status Periodically
async function checkBotStatus() {
    try {
        const res = await fetch('/bot/status');
        if (res.ok) {
            const data = await res.json();
            const btn = document.getElementById('start-bot-btn');
            if (btn) {
                isBotRunning = data.running;
                if (isBotRunning) {
                    if (data.asset === currentAssetId) {
                        btn.textContent = 'Detener Bot';
                        btn.classList.add('running');
                        btn.disabled = false;
                    } else {
                        btn.textContent = 'Corriendo en ' + data.asset;
                        btn.classList.add('running');
                        btn.disabled = true;
                    }
                } else {
                    btn.textContent = 'Iniciar Bot';
                    btn.classList.remove('running');
                    btn.disabled = !currentAssetId;
                }
            }
        }
    } catch(e) {}
}
setInterval(checkBotStatus, 5000);

// Interval Change
document.getElementById('interval-select').onchange = loadCandles;

// ─── Real-Time Notifications (WebSocket) ──────────────────────────────────
let notifSocket = null;

function startNotificationStream() {
    if (notifSocket) return;
    
    const wsUrl = `ws://${location.host}/ws/notifications`;
    notifSocket = new WebSocket(wsUrl);

    notifSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'HISTORY') {
                renderSignalHistory(data.data.history || []);
            } else {
                showToast(data.type, data.message, data.data);
                // Actualizar scan signals en la pestaña "Último Scan"
                if (data.type === 'SCANNER_SCAN' && data.data && data.data.scan) {
                    renderScanSignals(data.data.scan);
                }
                // Recargar historial tras ejecución o resultado
                if (data.type === 'SCANNER_TRADE' || data.type === 'EXECUTION') {
                    onTradeExecuted();
                }
                if (data.type === 'TRADE_RESULT') {
                    onTradeExecuted();
                }
            }
        } catch (e) {
            console.error('Error parsing notification', e);
        }
    };

    notifSocket.onclose = () => {
        notifSocket = null;
        setTimeout(startNotificationStream, 3000); // Reconnect
    };
}

function showToast(type, message, data) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    // Check if we should play sound (only for executions or important alerts)
    if (type === 'EXECUTION' || type === 'SCANNER_TRADE') {
        const audio = document.getElementById('ding-sound');
        if (audio) {
            audio.volume = 0.5;
            audio.play().catch(e => console.log('Audio autoplay blocked', e));
        }
    }
    
    let title = type;
    if (type === 'ANALYSIS')      title = 'IA Analizando';
    if (type === 'EXECUTION')     title = 'Orden Ejecutada';
    if (type === 'SIGNAL')        title = 'Alerta de IA';
    if (type === 'ERROR')         title = 'Error o Bloqueo';
    if (type === 'SCANNER_ON')    title = 'Scanner Iniciado';
    if (type === 'SCANNER_SCAN')  title = `Scanner — ${data && data.best ? '✓ Señal: ' + data.best : 'Sin señales'}`;
    if (type === 'SCANNER_TRADE') { title = 'Scanner — Trade Ejecutado'; toast.className = 'toast SCANNER_TRADE'; }
    if (type === 'TRADE_RESULT') {
        const isWin = data && data.result === 'WIN';
        title = isWin ? '✓ Resultado: WIN' : '✕ Resultado: LOSS';
        toast.className = `toast ${isWin ? 'EXECUTION' : 'ERROR'}`;
        if (isWin) {
            const audio = document.getElementById('ding-sound');
            if (audio) audio.play().catch(() => {});
        }
    }
    
    // Mapeo específico para Exnova (Alerta Naranja)
    if (type === 'ERROR' && message && (message.toLowerCase().includes('asset not available') || message.toLowerCase().includes('cerrado'))) {
        toast.className = 'toast WARNING';
        title = 'Exnova: Rechazo de Broker';
    }

    let extraData = '';
    if (data && data.reason) extraData = `<div class="toast-message" style="margin-top:3px">${data.reason}</div>`;

    toast.innerHTML = `
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
        ${extraData}
    `;

    container.appendChild(toast);

    // Fade out and remove
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 500);
    }, 4000);
}

function renderSignalHistory(historyArray) {
    const list = document.getElementById('signal-list');
    if (!list) return;
    
    list.innerHTML = '';
    if (!historyArray || historyArray.length === 0) {
        list.innerHTML = '<div style="color:var(--text-secondary); font-size:0.8rem">No hay señales recientes</div>';
        return;
    }

    historyArray.forEach(sig => {
        const card = document.createElement('div');
        card.className = 'signal-card';
        
        let opColor = 'white';
        if (sig.op === 'CALL') opColor = 'var(--success-color)';
        if (sig.op === 'PUT') opColor = 'var(--error-color)';
        if (sig.op === 'WAIT') opColor = '#f59e0b';
        
        card.innerHTML = `
            <div class="signal-card-header">
                <span>${sig.ts || '--:--:--'}</span>
                <span>Confianza: <strong>${sig.pr || '--'}%</strong></span>
            </div>
            <div class="signal-card-body">
                <div><strong style="color:${opColor}">${sig.op || ''}</strong> - ${sig.status || ''}</div>
                <div class="reason" title="${sig.an || ''}">${sig.an || 'Sin razón específica'}</div>
            </div>
        `;
        list.appendChild(card);
    });
}

// ─── Render scan signal cards ─────────────────────────────────────────────
function renderScanSignals(scanArray) {
    const list = document.getElementById('signal-list');
    if (!list) return;
    list.innerHTML = '';
    if (!scanArray || scanArray.length === 0) {
        list.innerHTML = '<div style="color:var(--text-secondary);font-size:0.8rem;padding:12px">Sin señales en este ciclo</div>';
        return;
    }
    const qualified = scanArray.filter(s => s.qualifies);
    const others    = scanArray.filter(s => !s.qualifies);

    [...qualified, ...others].forEach(sig => {
        const card = document.createElement('div');
        card.className = 'signal-card' + (sig.qualifies ? ' qualified' : '');
        const dirColor = sig.direction === 'CALL' ? '#10b981' : sig.direction === 'PUT' ? '#ef4444' : '#94a3b8';
        const scoreBarW = Math.round((sig.score || 0) * 100);
        const scoreBarColor = scoreBarW > 70 ? '#10b981' : scoreBarW > 45 ? '#f59e0b' : '#475569';
        card.innerHTML = `
            <div class="signal-card-header">
                <span class="signal-card-asset">${sig.qualifies ? '✓' : '✗'} ${sig.asset || '—'}</span>
                <span class="signal-card-score">score ${sig.score != null ? sig.score.toFixed(3) : '—'}</span>
            </div>
            <div class="signal-card-dir" style="color:${dirColor}">${sig.direction || 'sin dir'}</div>
            <div class="signal-card-reason" title="${sig.reason || ''}">${sig.reason || '—'}</div>
            <div class="score-bar-wrap">
                <div class="score-bar-fill" style="width:${scoreBarW}%;background:${scoreBarColor}"></div>
            </div>`;
        list.appendChild(card);
    });
}

// ─── Scanner Multi-Activo ─────────────────────────────────────────────────
let isScannerRunning = false;
let scannerMode = 'practice'; // 'practice' | 'paper'

// Mode toggle buttons
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (isScannerRunning) return; // no cambiar mientras corre
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        scannerMode = btn.dataset.mode;
    });
});

const scannerBtn = document.getElementById('start-scanner-btn');
if (scannerBtn) {
    scannerBtn.onclick = async () => {
        if (isScannerRunning) {
            scannerBtn.disabled = true;
            try {
                const res = await fetch('/scanner/stop', { method: 'POST' });
                if (res.ok) {
                    isScannerRunning = false;
                    setScannerBtnState(false);
                    document.getElementById('scanner-indicator').style.display = 'none';
                    document.querySelectorAll('.mode-btn').forEach(b => b.style.opacity = '1');
                }
            } catch (e) {
                alert('Error al detener scanner');
            } finally {
                scannerBtn.disabled = false;
            }
        } else {
            scannerBtn.disabled = true;
            scannerBtn.innerHTML = '<span class="scanner-btn-icon">⬡</span> Iniciando...';
            const amount    = parseFloat(document.getElementById('scanner-amount').value) || 1.0;
            const expiry    = parseInt(document.getElementById('scanner-expiry').value) || 2;
            try {
                const res = await fetch('/scanner/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ amount, expiry_min: expiry, mode: scannerMode })
                });
                if (res.ok) {
                    const data = await res.json();
                    isScannerRunning = true;
                    setScannerBtnState(true);
                    document.getElementById('scanner-indicator').style.display = 'flex';
                    document.querySelectorAll('.mode-btn').forEach(b => b.style.opacity = '0.5');
                    const modeLabel = scannerMode === 'practice' ? 'CUENTA PRÁCTICA' : 'SIMULADO';
                    showToast('SCANNER_ON', `Scanner activo — ${modeLabel}`, {
                        reason: `$${amount} × ${expiry}min | Detectando activos automáticamente`
                    });
                } else {
                    const err = await res.json();
                    alert(err.detail || 'Error al iniciar scanner');
                    setScannerBtnState(false);
                }
            } catch (e) {
                alert('Error de conexión');
                setScannerBtnState(false);
            } finally {
                scannerBtn.disabled = false;
            }
        }
    };
}

function setScannerBtnState(running) {
    const btn = document.getElementById('start-scanner-btn');
    if (!btn) return;
    if (running) {
        btn.innerHTML = '<span class="scanner-btn-icon">■</span> Detener Scanner';
        btn.classList.add('running');
    } else {
        btn.innerHTML = '<span class="scanner-btn-icon">⬡</span> Iniciar Scanner';
        btn.classList.remove('running');
    }
}

async function checkScannerStatus() {
    try {
        const res = await fetch('/scanner/status');
        if (!res.ok) return;
        const data = await res.json();
        const indicator = document.getElementById('scanner-indicator');
        isScannerRunning = data.running;
        setScannerBtnState(data.running);
        if (indicator) indicator.style.display = data.running ? 'flex' : 'none';
    } catch (e) {}
}
setInterval(checkScannerStatus, 5000);

// ─── Trade History Panel ──────────────────────────────────────────────────

// Tab switching
document.querySelectorAll('.history-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        if (!tab) return;
        document.querySelectorAll('.history-tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const content = document.getElementById('tab-' + tab);
        if (content) content.classList.add('active');
        if (tab === 'trades') loadTradeHistory();
    });
});

document.getElementById('refresh-trades-btn')?.addEventListener('click', loadTradeHistory);

async function loadTradeHistory() {
    try {
        const res = await fetch('/trades?limit=150');
        if (!res.ok) return;
        const data = await res.json();
        renderTradeHistory(data.trades || []);
    } catch (e) {
        console.error('Error loading trades:', e);
    }
}

function stratInfo(reasoning) {
    const r = (reasoning || '').toLowerCase();
    if (r.includes('bb_body_reversal')) return { label: 'BB Body Rev', cls: 'strat-bb' };
    if (r.includes('streak'))           return { label: 'Streak',      cls: 'strat-str' };
    if (r.includes('classical') || r.includes('rsi')) return { label: 'RSI+BB', cls: 'strat-rsi' };
    return { label: 'Scanner', cls: 'strat-scan' };
}

function renderTradeHistory(trades) {
    const tbody   = document.getElementById('trades-tbody');
    if (!tbody) return;

    // ── Stats bar ──────────────────────────────────────────────────────────
    const closed  = trades.filter(t => t.result === 'WIN' || t.result === 'LOSS');
    const pending = trades.filter(t => t.result === 'PENDING' || t.result === null);
    const wins    = closed.filter(t => t.result === 'WIN').length;
    const wr      = closed.length ? Math.round(wins / closed.length * 100) : null;
    const profit  = closed.reduce((s, t) => s + (t.profit || 0), 0);

    const setVal = (id, val, color) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = val;
        if (color) el.style.color = color;
    };

    setVal('stat-total',   trades.length || '—');
    setVal('stat-wr',
        wr !== null ? `${wr}%` : '—',
        wr !== null ? (wr >= 55 ? '#10b981' : wr >= 45 ? '#f59e0b' : '#ef4444') : null
    );
    setVal('stat-pnl',
        closed.length ? `${profit >= 0 ? '+' : ''}$${profit.toFixed(2)}` : '—',
        profit >= 0 ? '#10b981' : '#ef4444'
    );
    setVal('stat-pending', pending.length || '0');

    // ── Table rows ─────────────────────────────────────────────────────────
    if (!trades.length) {
        tbody.innerHTML = '<tr><td colspan="9" class="table-empty">Sin operaciones registradas aún. Inicia el scanner para comenzar.</td></tr>';
        return;
    }

    tbody.innerHTML = trades.map(t => {
        const dt   = new Date(t.timestamp);
        const hora = dt.toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' });
        const dia  = dt.toLocaleDateString('es', { day: '2-digit', month: '2-digit' });

        const dirBadge = t.direction === 'CALL'
            ? `<span class="dir-badge dir-call">▲ CALL</span>`
            : `<span class="dir-badge dir-put">▼ PUT</span>`;

        const { label: stratLabel, cls: stratCls } = stratInfo(t.ai_reasoning);
        const stratBadge = `<span class="strat-pill ${stratCls}">${stratLabel}</span>`;

        let resBadge  = `<span class="res-pending">● PEND</span>`;
        let rowClass  = 'row-pending';
        if (t.result === 'WIN')  { resBadge = `<span class="res-win">✓ WIN</span>`;   rowClass = 'row-win'; }
        if (t.result === 'LOSS') { resBadge = `<span class="res-loss">✕ LOSS</span>`; rowClass = 'row-loss'; }

        const proba = t.predicted_proba != null
            ? Math.round(t.predicted_proba * (t.predicted_proba <= 1 ? 100 : 1)) + '%'
            : '—';

        const profitStr = (t.profit != null && t.result !== 'PENDING')
            ? `<span class="${t.profit >= 0 ? 'profit-pos' : 'profit-neg'}">${t.profit >= 0 ? '+' : ''}$${Math.abs(t.profit).toFixed(2)}</span>`
            : '<span style="color:#475569">—</span>';

        const modeStr = (t.mode === 'paper')
            ? `<span class="mode-badge-paper">paper</span>`
            : `<span class="mode-badge-live">demo</span>`;

        return `<tr class="${rowClass}">
            <td style="color:var(--text-secondary);font-size:0.75rem">${dia} ${hora}</td>
            <td><strong style="font-size:0.8rem">${t.asset || '—'}</strong></td>
            <td>${dirBadge}</td>
            <td>${stratBadge}</td>
            <td style="color:#60a5fa;font-weight:600">${proba}</td>
            <td style="color:var(--text-secondary)">${t.expiry_min || '—'}m</td>
            <td>${modeStr}</td>
            <td>${resBadge}</td>
            <td>${profitStr}</td>
        </tr>`;
    }).join('');
}

function onTradeExecuted() {
    setTimeout(loadTradeHistory, 1500);
}

// Auto-refresh trades cada 20s
setInterval(loadTradeHistory, 20000);

// Start notification WS
startNotificationStream();

// Logout handler
const logoutBtn = document.getElementById('logout-btn');
if (logoutBtn) {
    logoutBtn.onclick = async () => {
        if (!confirm('¿Estás seguro de que deseas cerrar la sesión en Exnova y apagar completamente el servidor web local?')) return;
        
        try {
            logoutBtn.textContent = 'Apagando servidor...';
            logoutBtn.disabled = true;
            await fetch('/shutdown', { method: 'POST' });
        } catch (e) {
            // Expected to throw error since the backend disconnects entirely
        }
        
        // Render shutdown screen
        document.body.innerHTML = `
            <div style="display:flex; height:100vh; align-items:center; justify-content:center; flex-direction:column; color:white; background:#0d0f14; text-align:center;">
                <svg style="width:64px;height:64px;color:#10b981;margin-bottom:20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <h2 style="font-size:2rem; margin-bottom:10px;">Servidor Apagado Exitosamente</h2>
                <p style="color:#94a3b8; max-width:400px; display:block">La sesión de Exnova se cerró de forma segura, el bot se ha detenido y el <b>puerto 8000</b> está completamente liberado.<br><br>Ya puedes cerrar esta ventana con total seguridad.</p>
            </div>
        `;
    };
}

