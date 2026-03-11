// MonopolyTrader v6 Dashboard

let DATA = null;
let charts = {};

async function loadData() {
    try {
        const resp = await fetch('data.json?t=' + Date.now());
        DATA = await resp.json();
        render();
    } catch (e) {
        document.getElementById('loading').textContent = 'Error loading data: ' + e.message;
    }
}

function fmt(n, dec = 2) {
    return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function fmtPct(n) {
    const sign = n >= 0 ? '+' : '';
    return sign + Number(n).toFixed(2) + '%';
}

function pnlClass(n) {
    return n > 0 ? 'positive' : n < 0 ? 'negative' : 'neutral';
}

function timeAgo(iso) {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return Math.floor(diff) + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
}

function shortDate(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function shortDateTime(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
        d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// --- Chart defaults ---

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: '#8b90a5', font: { size: 11 } } },
    },
    scales: {
        x: { ticks: { color: '#8b90a5', font: { size: 10 } }, grid: { color: 'rgba(45,49,72,0.5)' } },
        y: { ticks: { color: '#8b90a5', font: { size: 10 } }, grid: { color: 'rgba(45,49,72,0.5)' } },
    }
};

// --- Render ---

function render() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('app').style.display = 'block';

    renderHeader();
    renderStatusBanner();
    renderMarketIntelligence();
    renderAgentMind();
    renderResearchLab();
    renderDailyBriefing();
    renderKPIs();
    renderBenchmarkComparison();
    renderPortfolioChart();
    renderDrawdownChart();
    renderSharpeChart();
    renderPlaybook();
    renderTradeJournal();
    renderTradeLog();
}

// --- Header ---

function renderHeader() {
    const cp = DATA.current_price;
    if (!cp) return;
    const changeClass = cp.change >= 0 ? 'positive' : 'negative';
    const changeSign = cp.change >= 0 ? '+' : '';
    document.getElementById('live-price').innerHTML =
        `TSLA <strong class="${changeClass}">${fmt(cp.price)}</strong> ` +
        `<span class="${changeClass}">${changeSign}${cp.change.toFixed(2)} (${changeSign}${cp.change_pct.toFixed(2)}%)</span>`;

    if (DATA.generated_at) {
        document.getElementById('update-time').textContent = timeAgo(DATA.generated_at);
    }
}

// --- Status Banner ---

function renderStatusBanner() {
    const el = document.getElementById('status-banner');
    const staleHours = DATA.generated_at
        ? (Date.now() - new Date(DATA.generated_at).getTime()) / 3600000
        : 999;

    let cls, text;
    if (staleHours > 6) {
        cls = 'stale';
        text = 'Data may be stale — last updated ' + timeAgo(DATA.generated_at);
    } else if (DATA.market_open) {
        cls = 'market-open';
        text = 'Market is open';
        if (DATA.time_et) text += ' — ' + DATA.time_et + ' ET';
    } else {
        cls = 'market-closed';
        text = 'Market is closed';
        if (DATA.time_et) text += ' — ' + DATA.time_et + ' ET';
    }

    el.className = 'status-banner ' + cls;
    el.innerHTML = `<div class="status-dot"></div><span class="status-text">${text}</span>`;
}

// --- Market Intelligence ---

function renderMarketIntelligence() {
    const mid = DATA.market_intelligence;
    if (!mid) return;

    const section = document.getElementById('market-intel-section');
    section.style.display = 'block';

    // Updated time
    if (mid.last_updated) {
        document.getElementById('intel-updated').textContent = 'Updated ' + timeAgo(mid.last_updated);
    }

    // Thesis
    const thesis = mid.thesis || {};
    const dir = (thesis.direction || 'neutral').toLowerCase();
    const conf = Math.round((thesis.confidence || 0) * 100);
    const dirClass = dir === 'bull' ? 'thesis-bull' : dir === 'bear' ? 'thesis-bear' : 'thesis-neutral';
    const confColor = dir === 'bull' ? '#22c55e' : dir === 'bear' ? '#ef4444' : '#8b90a5';

    document.getElementById('intel-thesis').innerHTML = `
        <span class="thesis-direction-badge ${dirClass}">${escapeHtml(dir)}</span>
        <div class="thesis-conf-ring" style="background: conic-gradient(${confColor} ${conf}%, var(--surface2) 0)">
            <div class="thesis-conf-inner">${conf}%</div>
        </div>
        <div class="thesis-reasoning">${escapeHtml(thesis.reasoning || thesis.evidence || 'No thesis reasoning available.')}</div>
    `;

    // Key levels
    const levels = mid.key_levels || {};
    let levelsHtml = '<div class="intel-panel-title">Key Levels</div>';
    if (levels.support && levels.support.length) {
        levels.support.forEach((s, i) => {
            const val = typeof s === 'object' ? s.price || s.level || s : s;
            levelsHtml += `<div class="level-row"><span class="level-label">Support ${i + 1}</span><span class="level-value level-support">${fmt(val)}</span></div>`;
        });
    }
    if (levels.resistance && levels.resistance.length) {
        levels.resistance.forEach((r, i) => {
            const val = typeof r === 'object' ? r.price || r.level || r : r;
            levelsHtml += `<div class="level-row"><span class="level-label">Resistance ${i + 1}</span><span class="level-value level-resistance">${fmt(val)}</span></div>`;
        });
    }
    // Handle flat format (support_1, resistance_1, etc.)
    if (!levels.support && !levels.resistance) {
        Object.keys(levels).forEach(k => {
            const isSupport = k.toLowerCase().includes('support');
            const cls = isSupport ? 'level-support' : 'level-resistance';
            const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            levelsHtml += `<div class="level-row"><span class="level-label">${escapeHtml(label)}</span><span class="level-value ${cls}">${fmt(levels[k])}</span></div>`;
        });
    }
    document.getElementById('intel-levels').innerHTML = levelsHtml;

    // Catalysts
    const catalysts = mid.active_catalysts || [];
    let catHtml = '<div class="intel-panel-title">Active Catalysts</div>';
    if (Array.isArray(catalysts)) {
        catalysts.forEach(c => {
            const text = typeof c === 'object' ? (c.description || c.catalyst || c.name || JSON.stringify(c)) : c;
            catHtml += `<div class="catalyst-item"><span class="catalyst-dot"></span><span>${escapeHtml(text)}</span></div>`;
        });
    }
    if (!catalysts.length) {
        catHtml += '<div style="font-size:12px;color:var(--text2)">No active catalysts reported.</div>';
    }
    document.getElementById('intel-catalysts').innerHTML = catHtml;

    // Sector context
    const sector = mid.sector_context;
    if (sector) {
        const sectorText = typeof sector === 'string' ? sector : (sector.summary || sector.description || JSON.stringify(sector));
        document.getElementById('intel-sector').innerHTML = `<strong style="color:var(--text)">Sector:</strong> ${escapeHtml(sectorText)}`;
        document.getElementById('intel-sector').style.display = 'block';
    } else {
        document.getElementById('intel-sector').style.display = 'none';
    }

    // What's working / not working
    const working = mid.what_working || mid.whats_working;
    const notWorking = mid.what_not_working || mid.whats_not_working;
    if (working || notWorking) {
        let lessonsHtml = '';
        if (working) {
            const wText = typeof working === 'string' ? working : (Array.isArray(working) ? working.join('. ') : JSON.stringify(working));
            lessonsHtml += `<div class="intel-lessons-panel working"><div class="intel-lessons-title">What's Working</div><div class="intel-lessons-text">${escapeHtml(wText)}</div></div>`;
        }
        if (notWorking) {
            const nwText = typeof notWorking === 'string' ? notWorking : (Array.isArray(notWorking) ? notWorking.join('. ') : JSON.stringify(notWorking));
            lessonsHtml += `<div class="intel-lessons-panel not-working"><div class="intel-lessons-title">What's Not Working</div><div class="intel-lessons-text">${escapeHtml(nwText)}</div></div>`;
        }
        document.getElementById('intel-lessons').innerHTML = lessonsHtml;
    }

    // Thesis History (v6)
    const thesisHistory = DATA.thesis_history;
    const historyEl = document.getElementById('intel-thesis-history');
    if (thesisHistory && typeof thesisHistory === 'string' && thesisHistory.trim()) {
        historyEl.innerHTML = `
            <button class="thesis-history-toggle" onclick="toggleThesisHistory(this)">
                <span class="toggle-arrow">&#9654;</span> Thesis History
            </button>
            <div class="thesis-history-content">${escapeHtml(thesisHistory)}</div>
        `;
        historyEl.style.display = 'block';
    } else {
        historyEl.style.display = 'none';
    }
}

function toggleThesisHistory(btn) {
    btn.classList.toggle('open');
    const content = btn.nextElementSibling;
    content.classList.toggle('open');
}

// --- Agent's Mind ---

function renderAgentMind() {
    const lc = DATA.latest_cycle;
    if (!lc) return;

    const section = document.getElementById('agent-mind-section');
    section.style.display = 'block';

    // Time
    if (lc.timestamp) {
        document.getElementById('agent-mind-time').textContent = timeAgo(lc.timestamp);
    }

    // Decision row
    const action = (lc.action || 'HOLD').toUpperCase();
    const actionClass = action === 'BUY' ? 'am-buy' : action === 'SELL' ? 'am-sell' : 'am-hold';
    const conf = Math.round((lc.confidence || 0) * 100);
    const confColor = action === 'BUY' ? '#22c55e' : action === 'SELL' ? '#ef4444' : '#8b90a5';

    let metaHtml = '';
    if (lc.shares) {
        metaHtml += `<span class="am-shares">${lc.shares} shares</span>`;
    }
    if (lc.price) {
        metaHtml += `<span class="am-price">@ ${fmt(lc.price)}</span>`;
    }

    // Strategy chip
    let strategyChip = '';
    if (lc.strategy) {
        strategyChip = `<span class="am-strategy-chip">${escapeHtml(lc.strategy)}</span>`;
    }

    document.getElementById('agent-mind-decision').innerHTML = `
        <div class="am-decision-row">
            <div class="am-action-badge ${actionClass}">${action}</div>
            <div class="am-conf-ring" style="background: conic-gradient(${confColor} ${conf}%, var(--surface2) 0)">
                <div class="am-conf-inner">${conf}%</div>
            </div>
            <div class="am-decision-meta">
                ${metaHtml}
                ${strategyChip}
            </div>
        </div>
    `;

    // Hypothesis + expected_learning (v6)
    let hypothesisHtml = '';
    if (lc.hypothesis) {
        hypothesisHtml += `<div class="am-hypothesis-text">${escapeHtml(lc.hypothesis)}</div>`;
    }
    if (lc.expected_learning) {
        hypothesisHtml += `
            <div class="am-expected-learning">
                <div class="am-expected-learning-label">Expected Learning</div>
                ${escapeHtml(lc.expected_learning)}
            </div>
        `;
    }
    document.getElementById('agent-mind-hypothesis').innerHTML = hypothesisHtml;

    // Reasoning + risk note
    let reasonHtml = '';
    if (lc.reasoning) {
        reasonHtml += `<div class="am-reasoning-text">${escapeHtml(lc.reasoning)}</div>`;
    }
    if (lc.risk_note) {
        reasonHtml += `<div class="am-risk-note">${escapeHtml(lc.risk_note)}</div>`;
    }
    document.getElementById('agent-mind-reasoning').innerHTML = reasonHtml;

    // Prediction (v6 format: {direction, magnitude, cycles, basis})
    let predHtml = '';
    const pred = lc.prediction;
    if (pred && typeof pred === 'object' && pred.direction) {
        const dir = pred.direction;
        const dirCls = dir === 'up' ? 'pred-up' : dir === 'down' ? 'pred-down' : 'pred-flat';
        const arrow = dir === 'up' ? '\u25B2' : dir === 'down' ? '\u25BC' : '\u25CF';
        const mag = pred.magnitude || '';
        const cycles = pred.cycles || '';

        predHtml = `
            <div class="am-prediction-row">
                <span class="am-prediction-label">Prediction</span>
                <span class="am-prediction-dir ${dirCls}">${arrow} ${escapeHtml(dir)}</span>
                ${mag ? `<span class="am-prediction-detail">${escapeHtml(mag)} move</span>` : ''}
                ${cycles ? `<span class="am-prediction-detail">in ${cycles} cycle${cycles !== 1 ? 's' : ''}</span>` : ''}
                ${pred.basis ? `<span class="am-prediction-basis">${escapeHtml(pred.basis)}</span>` : ''}
            </div>
        `;
    }
    // Also handle the older predictions format: {30min: {direction, target, confidence}, ...}
    else if (lc.predictions && typeof lc.predictions === 'object' && !Array.isArray(lc.predictions)) {
        const preds = lc.predictions;
        const timeframes = Object.keys(preds);
        if (timeframes.length) {
            let rows = '';
            timeframes.forEach(tf => {
                const p = preds[tf];
                if (!p || !p.direction) return;
                const dir = p.direction;
                const dirCls = dir === 'up' ? 'pred-up' : dir === 'down' ? 'pred-down' : 'pred-flat';
                const arrow = dir === 'up' ? '\u25B2' : dir === 'down' ? '\u25BC' : '\u25CF';
                const target = p.target ? fmt(p.target) : '';
                const confStr = p.confidence ? Math.round(p.confidence * 100) + '%' : '';
                rows += `
                    <span class="am-prediction-dir ${dirCls}">${arrow} ${escapeHtml(dir)}</span>
                    <span class="am-prediction-detail">${escapeHtml(tf)}${target ? ' \u2192 ' + target : ''}${confStr ? ' (' + confStr + ')' : ''}</span>
                `;
            });
            if (rows) {
                predHtml = `
                    <div class="am-prediction-row">
                        <span class="am-prediction-label">Predictions</span>
                        ${rows}
                    </div>
                `;
            }
        }
    }
    document.getElementById('agent-mind-prediction').innerHTML = predHtml;

    // Regime chips
    const regime = lc.regime || {};
    let chipsHtml = '';
    if (typeof regime === 'object') {
        if (regime.trend) {
            const tColor = regime.trend === 'bullish' ? 'var(--green)' : regime.trend === 'bearish' ? 'var(--red)' : 'var(--text2)';
            chipsHtml += `<span class="am-regime-chip" style="color:${tColor};border-color:${tColor}">${escapeHtml(regime.trend)}</span>`;
        }
        if (regime.volatility) {
            const vColor = regime.volatility === 'high' ? 'var(--orange)' : regime.volatility === 'low' ? 'var(--cyan)' : 'var(--text2)';
            chipsHtml += `<span class="am-regime-chip" style="color:${vColor};border-color:${vColor}">${escapeHtml(regime.volatility)} vol</span>`;
        }
        if (regime.vix !== undefined) {
            chipsHtml += `<span class="am-regime-chip" style="color:var(--purple);border-color:var(--purple)">VIX ${Number(regime.vix).toFixed(1)}</span>`;
        }
    } else if (typeof regime === 'string') {
        chipsHtml += `<span class="am-regime-chip" style="color:var(--text2);border-color:var(--text2)">${escapeHtml(regime)}</span>`;
    }
    // Also show VIX from top-level if present and not in regime
    if (lc.vix !== undefined && !(regime && regime.vix !== undefined)) {
        chipsHtml += `<span class="am-regime-chip" style="color:var(--purple);border-color:var(--purple)">VIX ${Number(lc.vix).toFixed(1)}</span>`;
    }
    document.getElementById('agent-mind-regime').innerHTML = chipsHtml;
}

// --- Research Lab (v6) ---

function renderResearchLab() {
    const rm = DATA.research_metrics;
    const ps = DATA.prediction_summary;
    const ss = DATA.shadow_summary;

    // Show if any v6 data exists
    const hasResearch = rm && typeof rm === 'object' && Object.keys(rm).length > 0;
    const hasPrediction = ps && typeof ps === 'object' && Object.keys(ps).length > 0;
    const hasShadow = ss && typeof ss === 'object' && Object.keys(ss).length > 0;

    if (!hasResearch && !hasPrediction && !hasShadow) return;

    const section = document.getElementById('research-lab-section');
    section.style.display = 'block';

    // Research Metrics panel
    renderResearchMetrics(rm || {});

    // Prediction Scorecard panel
    renderPredictionScorecard(ps || {});

    // Shadow Journal panel
    renderShadowJournal(ss || {});
}

function renderResearchMetrics(rm) {
    const el = document.getElementById('research-metrics-panel');
    let html = '<div class="research-panel-title">Research Metrics</div>';

    if (!rm || Object.keys(rm).length === 0) {
        html += '<div class="shadow-empty">Research metrics accumulating...</div>';
        el.innerHTML = html;
        return;
    }

    // Experiment efficiency
    if (rm.experiment_efficiency !== undefined) {
        const eff = Math.round(rm.experiment_efficiency * 100);
        const cls = eff >= 30 ? 'metric-good' : eff >= 15 ? '' : 'metric-warn';
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Experiment Efficiency</span>
            <span class="research-metric-value ${cls}">${eff}%</span>
        </div>`;
    }

    // Redundant loss rate
    if (rm.redundant_loss_rate !== undefined) {
        const rlr = Math.round(rm.redundant_loss_rate * 100);
        const cls = rlr === 0 ? 'metric-good' : rlr <= 5 ? 'metric-warn' : 'metric-bad';
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Redundant Loss Rate</span>
            <span class="research-metric-value ${cls}">${rlr}%</span>
        </div>`;
    }

    // Pattern discovery count
    if (rm.pattern_discovery_count !== undefined) {
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Patterns Discovered</span>
            <span class="research-metric-value">${rm.pattern_discovery_count}</span>
        </div>`;
    }

    // Calibration error
    if (rm.calibration_error !== undefined) {
        const ce = rm.calibration_error;
        const cls = ce <= 0.1 ? 'metric-good' : ce <= 0.25 ? '' : 'metric-warn';
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Calibration Error</span>
            <span class="research-metric-value ${cls}">${ce.toFixed(2)}</span>
        </div>`;
    }

    // Rolling win rate
    if (rm.rolling_win_rate_10 !== undefined && rm.rolling_win_rate_10 !== null) {
        const rwr = Math.round(rm.rolling_win_rate_10 * 100);
        const cls = rwr >= 50 ? 'metric-good' : rwr >= 35 ? '' : 'metric-bad';
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Win Rate (last 10)</span>
            <span class="research-metric-value ${cls}">${rwr}%</span>
        </div>`;
    }

    el.innerHTML = html;
}

function renderPredictionScorecard(ps) {
    const el = document.getElementById('prediction-scorecard-panel');
    let html = '<div class="research-panel-title">Prediction Scorecard</div>';

    if (!ps || ps.resolved === undefined || ps.resolved === 0) {
        const total = ps && ps.total_predictions ? ps.total_predictions : 0;
        html += `<div class="shadow-empty">Predictions accumulating${total > 0 ? ' (' + total + ' pending)' : ''}...</div>`;
        el.innerHTML = html;
        return;
    }

    // Featured: direction accuracy (large)
    const dirAcc = Math.round((ps.direction_accuracy || 0) * 100);
    const dirColor = dirAcc >= 60 ? 'var(--green)' : dirAcc >= 45 ? 'var(--orange)' : 'var(--red)';
    html += `
        <div class="prediction-featured">
            <div class="prediction-featured-value" style="color:${dirColor}">${dirAcc}%</div>
            <div class="prediction-featured-label">Direction Accuracy (${ps.resolved} resolved)</div>
        </div>
    `;

    // Magnitude accuracy and avg score
    if (ps.magnitude_accuracy !== undefined) {
        const magAcc = Math.round(ps.magnitude_accuracy * 100);
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Magnitude Accuracy</span>
            <span class="research-metric-value">${magAcc}%</span>
        </div>`;
    }

    if (ps.avg_score !== undefined) {
        html += `<div class="research-metric-row">
            <span class="research-metric-label">Avg Score</span>
            <span class="research-metric-value">${ps.avg_score.toFixed(2)} / 1.0</span>
        </div>`;
    }

    // By-direction breakdown with mini bars
    const byDir = ps.by_direction || {};
    const dirKeys = ['up', 'down', 'flat'];
    const hasAnyDir = dirKeys.some(d => byDir[d]);
    if (hasAnyDir) {
        html += '<div style="margin-top:10px">';
        dirKeys.forEach(d => {
            if (!byDir[d]) return;
            const info = byDir[d];
            const acc = Math.round(info.accuracy * 100);
            const barColor = acc >= 60 ? 'var(--green)' : acc >= 45 ? 'var(--orange)' : 'var(--red)';
            html += `
                <div class="pred-dir-row">
                    <span class="pred-dir-label">${d}</span>
                    <div class="pred-dir-bar-track">
                        <div class="pred-dir-bar" style="width:${Math.max(acc, 5)}%;background:${barColor}"></div>
                    </div>
                    <span class="pred-dir-stat">${acc}% (${info.count})</span>
                </div>
            `;
        });
        html += '</div>';
    }

    // Bias warning
    if (ps.overpredict_magnitude) {
        html += '<div class="pred-bias-warning">Bias: over-predicts magnitude — actual moves are smaller than expected</div>';
    }

    el.innerHTML = html;
}

function renderShadowJournal(ss) {
    const el = document.getElementById('shadow-journal-panel');
    let html = '<div class="research-panel-title">Shadow Journal</div>';

    if (!ss || ss.total_holds === undefined || ss.total_holds === 0) {
        html += '<div class="shadow-empty">Shadow tracking active — data accumulating</div>';
        el.innerHTML = html;
        return;
    }

    const total = ss.total_holds;
    const period = ss.period_hours || 24;

    html += `<div class="research-metric-row">
        <span class="research-metric-label">Total HOLDs (${period}h)</span>
        <span class="research-metric-value">${total}</span>
    </div>`;

    // Profitable holds (missed gains)
    if (ss.profitable_buys !== undefined) {
        html += `<div class="shadow-stat-row">
            <span class="shadow-stat-label">Missed gains</span>
            <span class="shadow-stat-value" style="color:var(--orange)">${ss.profitable_buys}</span>
        </div>`;
    }

    // Unprofitable holds (correctly avoided)
    if (ss.unprofitable_buys !== undefined) {
        html += `<div class="shadow-stat-row">
            <span class="shadow-stat-label">Correctly avoided</span>
            <span class="shadow-stat-value" style="color:var(--green)">${ss.unprofitable_buys}</span>
        </div>`;
    }

    // Biggest missed gain
    if (ss.biggest_missed_gain && ss.biggest_missed_gain > 0) {
        html += `<div class="shadow-stat-row">
            <span class="shadow-stat-label">Biggest missed</span>
            <span class="shadow-stat-value" style="color:var(--orange)">${fmt(ss.biggest_missed_gain)}</span>
        </div>`;
    }

    // Biggest avoided loss
    if (ss.biggest_avoided_loss && ss.biggest_avoided_loss > 0) {
        html += `<div class="shadow-stat-row">
            <span class="shadow-stat-label">Biggest saved</span>
            <span class="shadow-stat-value" style="color:var(--green)">${fmt(ss.biggest_avoided_loss)}</span>
        </div>`;
    }

    // Hold quality
    if (ss.hold_quality !== undefined && ss.hold_quality > 0) {
        const hq = Math.round(ss.hold_quality * 100);
        const cls = hq >= 50 ? 'metric-good' : hq >= 30 ? '' : 'metric-warn';
        html += `<div class="research-metric-row" style="margin-top:6px;padding-top:6px;border-top:1px solid var(--border)">
            <span class="research-metric-label">Hold Quality</span>
            <span class="research-metric-value ${cls}">${hq}%</span>
        </div>`;
    }

    el.innerHTML = html;
}

// --- Daily Briefing ---

function renderDailyBriefing() {
    const br = DATA.daily_briefing;
    if (!br) return;

    const section = document.getElementById('briefing-section');
    section.style.display = 'block';

    if (br.generated_at) {
        document.getElementById('briefing-time').textContent = shortDateTime(br.generated_at);
    }

    // Thesis status
    if (br.thesis_status) {
        document.getElementById('briefing-thesis-status').innerHTML = `
            <div class="briefing-label">Thesis Status</div>
            <div class="briefing-text">${escapeHtml(br.thesis_status)}</div>
        `;
    }

    // Recommended posture
    if (br.recommended_posture) {
        const posture = br.recommended_posture.toLowerCase();
        let postureClass = 'posture-neutral';
        if (posture.includes('aggressive')) postureClass = 'posture-aggressive';
        else if (posture.includes('cautious')) postureClass = 'posture-cautious';
        else if (posture.includes('defensive')) postureClass = 'posture-defensive';

        document.getElementById('briefing-posture').innerHTML = `
            <div class="briefing-label">Recommended Posture</div>
            <span class="posture-badge ${postureClass}">${escapeHtml(br.recommended_posture)}</span>
        `;
    }

    // Scenarios
    if (br.scenarios) {
        let scenHtml = '<div class="briefing-label">Scenarios</div>';
        if (Array.isArray(br.scenarios)) {
            br.scenarios.forEach(s => {
                const text = typeof s === 'string' ? s : (s.description || s.scenario || JSON.stringify(s));
                const type = typeof s === 'object' ? (s.type || '').toLowerCase() : '';
                const bulletClass = type.includes('bull') ? 'scenario-bull' : type.includes('bear') ? 'scenario-bear' : 'scenario-base';
                const bullet = type.includes('bull') ? '\u25B2' : type.includes('bear') ? '\u25BC' : '\u25CF';
                scenHtml += `<div class="scenario-item"><span class="scenario-bullet ${bulletClass}">${bullet}</span><span>${escapeHtml(text)}</span></div>`;
            });
        } else if (typeof br.scenarios === 'string') {
            scenHtml += `<div class="briefing-text">${escapeHtml(br.scenarios)}</div>`;
        }
        document.getElementById('briefing-scenarios').innerHTML = scenHtml;
    }

    // Key data points
    if (br.key_data_points) {
        let dpHtml = '<div class="briefing-label">Key Data Points</div>';
        if (Array.isArray(br.key_data_points)) {
            br.key_data_points.forEach(dp => {
                const text = typeof dp === 'string' ? dp : (dp.point || dp.description || JSON.stringify(dp));
                dpHtml += `<div class="data-point-item">${escapeHtml(text)}</div>`;
            });
        } else if (typeof br.key_data_points === 'string') {
            dpHtml += `<div class="briefing-text">${escapeHtml(br.key_data_points)}</div>`;
        }
        document.getElementById('briefing-data-points').innerHTML = dpHtml;
    }
}

// --- KPIs ---

function renderKPIs() {
    const p = DATA.portfolio;
    if (!p) return;

    document.getElementById('kpi-value').innerHTML =
        `<div class="kpi-value">${fmt(p.total_value)}</div>`;

    const pnl = p.total_pnl || 0;
    const pnlPct = p.total_pnl_pct || 0;
    document.getElementById('kpi-pnl').innerHTML =
        `<div class="kpi-value ${pnlClass(pnl)}">${fmt(pnl)}</div>` +
        `<div class="kpi-sub">${fmtPct(pnlPct)}</div>`;

    document.getElementById('kpi-cash').innerHTML =
        `<div class="kpi-value">${fmt(p.cash)}</div>`;

    const w = p.winning_trades || 0;
    const l = p.losing_trades || 0;
    document.getElementById('kpi-trades').innerHTML =
        `<div class="kpi-value">${p.total_trades || 0}</div>` +
        `<div class="kpi-sub"><span class="positive">${w}W</span> / <span class="negative">${l}L</span></div>`;

    const wr = p.win_rate || 0;
    document.getElementById('kpi-winrate').innerHTML =
        `<div class="kpi-value ${wr >= 50 ? 'positive' : wr > 0 ? 'negative' : 'neutral'}">${wr.toFixed(1)}%</div>`;
}

// --- Benchmark Comparison ---

function renderBenchmarkComparison() {
    const p = DATA.portfolio;
    const benchmarks = DATA.benchmark;
    if (!p || !benchmarks) return;

    const agentReturn = p.total_pnl_pct || 0;

    // Calculate TSLA buy & hold return
    let bhReturn = 0;
    if (benchmarks.length >= 2) {
        const first = benchmarks[0].value;
        const last = benchmarks[benchmarks.length - 1].value;
        bhReturn = ((last - first) / first) * 100;
    }

    // Verdict
    const diff = agentReturn - bhReturn;
    let verdictClass, verdictText, verdictDesc;
    if (diff > 5) {
        verdictClass = 'verdict-great';
        verdictText = 'CRUSHING IT';
        verdictDesc = 'Significantly outperforming buy & hold';
    } else if (diff > 0) {
        verdictClass = 'verdict-good';
        verdictText = 'BEATING THE MARKET';
        verdictDesc = 'Outperforming buy & hold TSLA';
    } else if (diff > -2) {
        verdictClass = 'verdict-ok';
        verdictText = 'CLOSE';
        verdictDesc = 'Nearly matching buy & hold';
    } else if (diff > -10) {
        verdictClass = 'verdict-bad';
        verdictText = 'UNDERPERFORMING';
        verdictDesc = 'Trailing buy & hold TSLA';
    } else {
        verdictClass = 'verdict-bad';
        verdictText = 'GETTING SCHOOLED';
        verdictDesc = 'Significantly underperforming';
    }

    document.getElementById('verdict-badge').innerHTML =
        `<span class="verdict-label ${verdictClass}">${verdictText}</span>` +
        `<span class="verdict-desc">${verdictDesc}</span>`;

    // Bars
    const items = [
        { name: 'MonopolyTrader', value: agentReturn, highlight: true },
        { name: 'Buy & Hold TSLA', value: bhReturn, highlight: false },
    ];

    const maxAbs = Math.max(...items.map(i => Math.abs(i.value)), 1);
    let barsHtml = '';
    items.forEach(item => {
        const pct = Math.abs(item.value) / maxAbs * 100;
        const color = item.value >= 0 ? 'var(--green)' : 'var(--red)';
        const hlClass = item.highlight ? ' bar-highlight' : '';
        barsHtml += `
            <div class="benchmark-row${hlClass}">
                <span class="bench-name">${item.name}</span>
                <div class="bench-bar-track">
                    <div class="bench-bar" style="width:${Math.max(pct, 5)}%;background:${color}">${fmtPct(item.value)}</div>
                </div>
            </div>
        `;
    });
    document.getElementById('benchmark-bars').innerHTML = barsHtml;

    // Detail
    const startBal = p.starting_balance || 1000;
    document.getElementById('benchmark-detail').innerHTML = `
        <span>Started: ${fmt(startBal)}</span>
        <span>Current: ${fmt(p.total_value)}</span>
        <span>Realized: <span class="${pnlClass(p.realized_pnl || 0)}">${fmt(p.realized_pnl || 0)}</span></span>
        <span>Unrealized: <span class="${pnlClass(p.unrealized_pnl || 0)}">${fmt(p.unrealized_pnl || 0)}</span></span>
    `;
}

// --- Portfolio Chart ---

function renderPortfolioChart() {
    const snapshots = DATA.snapshots;
    const benchmarks = DATA.benchmark;
    if (!snapshots || !snapshots.length) return;

    const labels = snapshots.map(s => shortDate(s.date));
    const portfolioData = snapshots.map(s => s.total_value);

    const datasets = [
        {
            label: 'Portfolio',
            data: portfolioData,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            borderWidth: 2,
        }
    ];

    if (benchmarks && benchmarks.length) {
        datasets.push({
            label: 'Buy & Hold TSLA',
            data: benchmarks.map(b => b.value),
            borderColor: '#8b90a5',
            borderDash: [5, 3],
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false,
        });
    }

    if (charts.portfolio) charts.portfolio.destroy();
    charts.portfolio = new Chart(document.getElementById('portfolio-chart'), {
        type: 'line',
        data: { labels, datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    ticks: {
                        ...CHART_DEFAULTS.scales.y.ticks,
                        callback: v => '$' + v,
                    }
                }
            }
        }
    });
}

// --- Drawdown Chart ---

function renderDrawdownChart() {
    const pa = DATA.performance_analytics;
    if (!pa || !pa.drawdown_series || !pa.drawdown_series.length) return;

    const labels = pa.drawdown_series.map(d => shortDate(d.date));
    const values = pa.drawdown_series.map(d => d.drawdown_pct || (d.drawdown * 100) || 0);

    if (charts.drawdown) charts.drawdown.destroy();
    charts.drawdown = new Chart(document.getElementById('drawdown-chart'), {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Drawdown %',
                data: values,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 1.5,
            }]
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: { legend: { display: false } },
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    ticks: {
                        ...CHART_DEFAULTS.scales.y.ticks,
                        callback: v => v.toFixed(1) + '%',
                    }
                }
            }
        }
    });
}

// --- Sharpe Chart ---

function renderSharpeChart() {
    const pa = DATA.performance_analytics;
    if (!pa || !pa.rolling_sharpe || !pa.rolling_sharpe.length) return;

    const labels = pa.rolling_sharpe.map(d => shortDate(d.date));
    const values = pa.rolling_sharpe.map(d => d.sharpe);

    if (charts.sharpe) charts.sharpe.destroy();
    charts.sharpe = new Chart(document.getElementById('sharpe-chart'), {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Sharpe Ratio',
                data: values,
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 1.5,
            }]
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: { legend: { display: false } },
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    ticks: {
                        ...CHART_DEFAULTS.scales.y.ticks,
                        callback: v => v.toFixed(2),
                    }
                }
            }
        }
    });
}

// --- Playbook ---

function renderPlaybook() {
    const ledger = DATA.thesis_ledger;
    if (!ledger || !ledger.theses) return;

    const section = document.getElementById('playbook-section');
    section.style.display = 'block';

    const theses = ledger.theses;
    let html = '';

    // v6 format: theses is a flat dict keyed by "tag_name:tag_value"
    // Group by tag name (part before ":") to create categories
    const categories = {};
    const entries = Object.entries(theses);

    entries.forEach(([key, stats]) => {
        if (!stats || typeof stats !== 'object' || !stats.trades) return;

        // Parse the key: "tag_name:tag_value" or check if it's already nested
        const colonIdx = key.indexOf(':');
        let catName, tagValue;

        if (colonIdx > 0) {
            // v6 flat format: "rsi_zone:neutral"
            catName = key.substring(0, colonIdx);
            tagValue = key.substring(colonIdx + 1);
        } else {
            // Fallback: use as-is
            catName = 'other';
            tagValue = key;
        }

        if (!categories[catName]) {
            categories[catName] = [];
        }
        categories[catName].push({ tagValue, stats });
    });

    // If categories were built, render them
    const catNames = Object.keys(categories).sort();
    if (catNames.length > 0) {
        catNames.forEach(cat => {
            const items = categories[cat]
                .filter(item => item.stats.trades > 0)
                .sort((a, b) => (b.stats.win_rate || 0) - (a.stats.win_rate || 0));

            if (!items.length) return;

            html += `<div class="playbook-category">`;
            html += `<div class="playbook-cat-title">${escapeHtml(cat.replace(/_/g, ' '))}</div>`;

            items.forEach(({ tagValue, stats }) => {
                html += _renderPlaybookRow(tagValue, stats);
            });

            html += `</div>`;
        });
    } else {
        // Legacy nested format: theses[category][tagValue] = stats
        const legacyCategories = Object.keys(theses).sort();
        legacyCategories.forEach(cat => {
            const tags = theses[cat];
            if (!tags || typeof tags !== 'object') return;

            // Check if this looks like nested (values are objects with their own entries)
            const firstVal = Object.values(tags)[0];
            if (!firstVal || typeof firstVal !== 'object' || !firstVal.trades) return;

            const catEntries = Object.entries(tags)
                .filter(([_, v]) => v && v.trades > 0)
                .sort((a, b) => (b[1].win_rate || b[1].weighted_hit_rate || b[1].hit_rate || 0) - (a[1].win_rate || a[1].weighted_hit_rate || a[1].hit_rate || 0));

            if (!catEntries.length) return;

            html += `<div class="playbook-category">`;
            html += `<div class="playbook-cat-title">${escapeHtml(cat.replace(/_/g, ' '))}</div>`;

            catEntries.forEach(([tagVal, stats]) => {
                html += _renderPlaybookRow(tagVal, stats);
            });

            html += `</div>`;
        });
    }

    // Multi-tag patterns (v6)
    const multiPatterns = ledger.multi_tag_patterns;
    if (multiPatterns && typeof multiPatterns === 'object') {
        const multiEntries = Object.entries(multiPatterns)
            .filter(([_, v]) => v && v.trades >= 3)
            .sort((a, b) => (b[1].win_rate || 0) - (a[1].win_rate || 0));

        if (multiEntries.length) {
            html += `<div class="playbook-section-title">Multi-Tag Patterns</div>`;
            multiEntries.forEach(([pattern, stats]) => {
                html += _renderPlaybookRow(pattern, stats);
            });
        }
    }

    // Strategy stats (v6)
    const stratStats = ledger.strategy_stats;
    if (stratStats && typeof stratStats === 'object') {
        const stratEntries = Object.entries(stratStats)
            .filter(([_, v]) => v && v.trades >= 2)
            .sort((a, b) => (b[1].win_rate || 0) - (a[1].win_rate || 0));

        if (stratEntries.length) {
            html += `<div class="playbook-strategy-title">Strategy Performance</div>`;
            stratEntries.forEach(([stratName, stats]) => {
                html += _renderPlaybookRow(stratName.replace(/_/g, ' '), stats);
            });
        }
    }

    if (!html) {
        html = '<div style="font-size:13px;color:var(--text2)">No playbook data yet. Stats build after trades close.</div>';
    }

    document.getElementById('playbook-content').innerHTML = html;
}

function _renderPlaybookRow(label, stats) {
    const wr = stats.win_rate || stats.weighted_hit_rate || stats.hit_rate || 0;
    const wrPct = Math.round(wr * 100);
    const barColor = wr >= 0.5 ? 'var(--green)' : wr >= 0.3 ? 'var(--orange)' : 'var(--red)';
    const avgPnl = stats.avg_pnl || 0;
    const wins = stats.wins || 0;
    const losses = (stats.trades || 0) - wins;

    return `
        <div class="playbook-row">
            <span class="playbook-tag">${escapeHtml(label)}</span>
            <div class="playbook-bar-track">
                <div class="playbook-bar" style="width:${Math.max(wrPct, 8)}%;background:${barColor}">${wrPct}%</div>
            </div>
            <span class="playbook-stats">${stats.trades}T ${wins}W ${losses}L | ${fmt(avgPnl)}</span>
        </div>
    `;
}

// --- Trade Journal ---

function renderTradeJournal() {
    // v6 uses "trade_journal" (array of objects)
    // v5/older might use "journal" (could be a string or array)
    let journal = DATA.trade_journal;
    if (!journal || !Array.isArray(journal) || !journal.length) {
        // Try older "journal" key
        const legacyJournal = DATA.journal;
        if (legacyJournal && Array.isArray(legacyJournal) && legacyJournal.length && typeof legacyJournal[0] === 'object') {
            journal = legacyJournal;
        } else {
            return;
        }
    }

    const section = document.getElementById('journal-section');
    section.style.display = 'block';

    // Stats bar
    const js = DATA.journal_stats || {};
    let statsHtml = '';
    if (js.total_entries !== undefined) statsHtml += `<span><strong>${js.total_entries}</strong> entries</span>`;
    if (js.open_trades !== undefined) statsHtml += `<span><strong>${js.open_trades}</strong> open</span>`;
    if (js.closed_with_lesson !== undefined) statsHtml += `<span><strong>${js.closed_with_lesson}</strong> with lessons</span>`;
    if (js.avg_confidence !== undefined) statsHtml += `<span>Avg confidence: <strong>${Math.round(js.avg_confidence * 100)}%</strong></span>`;
    document.getElementById('journal-stats').innerHTML = statsHtml;

    // Entries (most recent first, limit to 20)
    const entries = [...journal].reverse().slice(0, 20);
    let html = '';

    entries.forEach(e => {
        const action = (e.action || 'HOLD').toUpperCase();
        const actionClass = action === 'BUY' ? 'badge-buy' : action === 'SELL' ? 'badge-sell' : 'badge-hold';
        const entryClass = action === 'BUY' ? 'entry-buy' : action === 'SELL' ? 'entry-sell' : '';

        let pnlHtml = '';
        if (e.realized_pnl !== undefined && e.realized_pnl !== null) {
            pnlHtml = `<span class="je-pnl ${pnlClass(e.realized_pnl)}">${fmt(e.realized_pnl)}</span>`;
        }

        // Strategy chip (v6)
        let strategyHtml = '';
        if (e.strategy) {
            strategyHtml = `<span class="badge-strategy">${escapeHtml(e.strategy)}</span>`;
        }

        // Outcome type chip (v6)
        let outcomeHtml = '';
        if (e.outcome_type) {
            const ot = e.outcome_type;
            let otClass = 'outcome-neutral';
            if (ot === 'thesis_correct' || ot === 'correct') otClass = 'outcome-thesis-correct';
            else if (ot === 'thesis_wrong' || ot === 'wrong') otClass = 'outcome-thesis-wrong';
            else if (ot === 'timing_wrong' || ot === 'execution_wrong') otClass = 'outcome-timing';
            outcomeHtml = `<span class="badge-outcome ${otClass}">${escapeHtml(ot.replace(/_/g, ' '))}</span>`;
        }

        // Hypothesis (v6)
        let hypothesisHtml = '';
        if (e.hypothesis) {
            hypothesisHtml = `<div class="je-hypothesis"><div class="je-hypothesis-label">Hypothesis</div>${escapeHtml(e.hypothesis)}</div>`;
        }

        // Expected learning (v6)
        let expectedLearningHtml = '';
        if (e.expected_learning) {
            expectedLearningHtml = `<div class="je-expected-learning">${escapeHtml(e.expected_learning)}</div>`;
        }

        let lessonHtml = '';
        if (e.lesson) {
            lessonHtml = `<div class="je-lesson"><div class="je-lesson-label">Lesson</div>${escapeHtml(e.lesson)}</div>`;
        }

        let tagsHtml = '';
        if (e.tags && typeof e.tags === 'object' && Object.keys(e.tags).length) {
            tagsHtml = '<div class="je-tags">';
            Object.entries(e.tags).forEach(([k, v]) => {
                tagsHtml += `<span class="je-tag">${escapeHtml(k)}: ${escapeHtml(v)}</span>`;
            });
            tagsHtml += '</div>';
        }

        html += `
            <div class="journal-entry ${entryClass}">
                <div class="je-header">
                    <span class="badge ${actionClass}">${action}</span>
                    ${strategyHtml}
                    ${outcomeHtml}
                    <span class="je-time">${e.timestamp ? shortDateTime(e.timestamp) : ''}</span>
                    ${e.shares ? `<span style="font-size:12px;color:var(--text2)">${e.shares} shares @ ${fmt(e.price || 0)}</span>` : ''}
                    ${pnlHtml}
                </div>
                ${hypothesisHtml}
                ${e.reasoning ? `<div class="je-reasoning">${escapeHtml(e.reasoning)}</div>` : ''}
                ${expectedLearningHtml}
                ${lessonHtml}
                ${tagsHtml}
            </div>
        `;
    });

    document.getElementById('journal-entries').innerHTML = html;
}

// --- Trade Log ---

function renderTradeLog() {
    const txns = DATA.transactions;
    if (!txns || !txns.length) return;

    const recent = [...txns].reverse().slice(0, 30);
    let html = '';

    recent.forEach(t => {
        const action = (t.action || '').toUpperCase();
        const badgeClass = action === 'BUY' ? 'badge-buy' : action === 'SELL' ? 'badge-sell' : 'badge-hold';

        let pnlHtml = '-';
        if (t.realized_pnl !== undefined && t.realized_pnl !== null) {
            pnlHtml = `<span class="${pnlClass(t.realized_pnl)}">${fmt(t.realized_pnl)}</span>`;
        }

        const reasoning = t.reasoning || '';
        const shortReason = reasoning.length > 120 ? reasoning.substring(0, 120) + '...' : reasoning;

        html += `
            <tr>
                <td>${t.timestamp ? shortDateTime(t.timestamp) : ''}</td>
                <td><span class="badge ${badgeClass}">${action}</span></td>
                <td>${t.shares || ''} shares @ ${fmt(t.price || 0)}<br><span style="font-size:11px;color:var(--text2)">Total: ${fmt(t.total_cost || 0)}</span></td>
                <td>${pnlHtml}</td>
                <td style="font-size:12px;color:var(--text2);max-width:300px">${escapeHtml(shortReason)}</td>
            </tr>
        `;
    });

    document.getElementById('trades-body').innerHTML = html;
}

// --- Init ---

loadData();

// Auto-refresh every 2 minutes
setInterval(loadData, 120000);
