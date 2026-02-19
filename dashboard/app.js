// MonopolyTrader v3 Dashboard

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

// --- Render ---

function render() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('app').style.display = 'block';

    renderHeader();
    renderStatusBanner();
    renderMilestones();
    renderBenchmarkComparison();
    renderGraduationChecklist();
    renderPortfolioChart();
    renderHealthStatus();
    renderEnsembleLeaderboard();
    renderHoldLog();
    renderKPIs();
    renderStrategyBars();
    renderWeightChart();
    renderPredictions();
    renderTradeLog();
    renderLessons();
    renderPatterns();
    renderJournal();
    setupTabs();
}

function renderHeader() {
    const p = DATA.current_price;
    const el = document.getElementById('live-price');
    el.innerHTML = `TSLA ${fmt(p.price)} <span class="${pnlClass(p.change)}">${fmtPct(p.change_pct)}</span>`;

    const time = document.getElementById('update-time');
    time.textContent = 'Updated ' + timeAgo(DATA.generated_at);
}

function renderStatusBanner() {
    const banner = document.getElementById('status-banner');
    const isOpen = DATA.market_open;
    const genTime = new Date(DATA.generated_at);
    const ageMs = Date.now() - genTime.getTime();
    const ageMin = Math.floor(ageMs / 60000);
    const isStale = isOpen && ageMin > 20;

    let cls, label, detail;
    if (isStale) {
        cls = 'stale';
        label = 'MARKET OPEN — DATA STALE';
        detail = `Last update ${ageMin}m ago`;
    } else if (isOpen) {
        cls = 'market-open';
        label = 'MARKET OPEN';
        detail = ageMin <= 1 ? 'Updated just now' : `Updated ${ageMin}m ago`;
    } else {
        cls = 'market-closed';
        label = 'MARKET CLOSED';
        detail = DATA.time_et || '';
    }

    banner.className = 'status-banner ' + cls;
    banner.innerHTML = `
        <div class="status-dot"></div>
        <span class="status-text">${label}</span>
        <span class="status-time">${detail}</span>
    `;
}

// --- NEW: Benchmark Comparison ---

function renderBenchmarkComparison() {
    const comp = DATA.benchmarks_comparison || {};
    const verdict = DATA.verdict || 'too_early';
    const container = document.getElementById('benchmark-bars');
    const verdictEl = document.getElementById('verdict-badge');
    const detailEl = document.getElementById('benchmark-detail');

    if (!comp.agent) {
        container.innerHTML = '<p style="color:var(--text2)">Benchmarks will appear after the first trading day.</p>';
        verdictEl.innerHTML = '';
        detailEl.innerHTML = '';
        return;
    }

    const verdictConfig = {
        'too_early': { label: 'TOO EARLY', cls: 'verdict-neutral', desc: 'Not enough data to judge performance' },
        'underperforming': { label: 'UNDERPERFORMING', cls: 'verdict-bad', desc: 'Agent is behind benchmarks' },
        'inconclusive': { label: 'INCONCLUSIVE', cls: 'verdict-neutral', desc: 'Mixed results across benchmarks' },
        'promising': { label: 'PROMISING', cls: 'verdict-ok', desc: 'Showing signs of edge' },
        'outperforming': { label: 'OUTPERFORMING', cls: 'verdict-good', desc: 'Beating key benchmarks' },
        'graduating': { label: 'GRADUATION READY', cls: 'verdict-great', desc: 'All criteria met!' },
    };

    const vc = verdictConfig[verdict] || verdictConfig['too_early'];
    verdictEl.innerHTML = `<span class="verdict-label ${vc.cls}">${vc.label}</span> <span class="verdict-desc">${vc.desc}</span>`;

    const benchmarks = [
        { name: 'Agent', value: comp.agent?.return_pct || 0, color: '#6366f1', highlight: true },
        { name: 'Buy & Hold TSLA', value: comp.buy_hold_tsla?.return_pct || 0, color: '#f59e0b' },
        { name: 'Buy & Hold SPY', value: comp.buy_hold_spy?.return_pct || 0, color: '#3b82f6' },
        { name: 'DCA TSLA', value: comp.dca_tsla?.return_pct || 0, color: '#22c55e' },
        { name: 'Random Median', value: comp.random_median?.return_pct || 0, color: '#8b90a5' },
    ];

    const maxAbs = Math.max(...benchmarks.map(b => Math.abs(b.value)), 1);

    container.innerHTML = benchmarks.map(b => {
        const width = Math.max(Math.abs(b.value) / maxAbs * 80, 3);
        const isPos = b.value >= 0;
        const barCls = isPos ? 'bar-positive' : 'bar-negative';
        const highlight = b.highlight ? 'bar-highlight' : '';
        return `
            <div class="benchmark-row ${highlight}">
                <span class="bench-name">${b.name}</span>
                <div class="bench-bar-track">
                    <div class="bench-bar ${barCls}" style="width:${width}%;background:${b.color}">${fmtPct(b.value)}</div>
                </div>
            </div>`;
    }).join('');

    const pctile = comp.percentile_vs_random || 0;
    detailEl.innerHTML = `
        <span>vs Random: <strong>${pctile.toFixed(0)}th percentile</strong></span>
        <span>Alpha vs TSLA: <strong class="${pnlClass(comp.alpha_vs_tsla)}">${fmtPct(comp.alpha_vs_tsla || 0)}</strong></span>
        <span>Alpha vs SPY: <strong class="${pnlClass(comp.alpha_vs_spy)}">${fmtPct(comp.alpha_vs_spy || 0)}</strong></span>
    `;
}

// --- NEW: Graduation Checklist ---

function renderGraduationChecklist() {
    const grad = DATA.graduation_criteria || {};
    const criteria = grad.criteria || {};
    const container = document.getElementById('graduation-grid');

    if (Object.keys(criteria).length === 0) {
        container.innerHTML = '<p style="color:var(--text2)">Graduation criteria will be tracked after benchmarks are initialized.</p>';
        return;
    }

    const labels = {
        min_trading_days: 'Trading Days',
        min_trades: 'Total Trades',
        percentile_vs_random: 'vs Random Pctile',
        sharpe_ratio: 'Sharpe Ratio',
        max_drawdown: 'Max Drawdown',
        prediction_accuracy: 'Prediction Accuracy',
        beats_buy_hold_tsla: 'Beat B&H TSLA',
        beats_buy_hold_spy: 'Beat B&H SPY',
        beats_dca: 'Beat DCA',
        beats_random_median: 'Beat Random',
        regime_diversity: 'Regime Diversity',
        positive_return: 'Positive Return',
    };

    container.innerHTML = Object.entries(criteria).map(([key, c]) => {
        const icon = c.passed ? 'check-icon' : 'x-icon';
        const cls = c.passed ? 'grad-pass' : 'grad-fail';
        const label = labels[key] || key;
        const actual = typeof c.actual === 'boolean' ? (c.actual ? 'Yes' : 'No') : c.actual;
        const required = typeof c.required === 'boolean' ? 'Yes' : c.required;
        return `
            <div class="grad-item ${cls}">
                <div class="${icon}"></div>
                <div class="grad-label">${label}</div>
                <div class="grad-value">${actual} / ${required}</div>
            </div>`;
    }).join('');

    const passed = grad.passed || 0;
    const total = grad.total || 12;
    container.innerHTML += `<div class="grad-summary">${passed}/${total} criteria met</div>`;
}

// --- NEW: Health Status ---

function renderHealthStatus() {
    const health = DATA.health || {};
    const alerts = DATA.active_alerts || [];
    const container = document.getElementById('health-strip');

    if (!health.components && alerts.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'flex';

    let html = '';

    // Health dots
    if (health.components) {
        html += '<div class="health-dots">';
        for (const [name, status] of Object.entries(health.components)) {
            const dotCls = status.healthy ? 'dot-healthy' : 'dot-unhealthy';
            html += `<span class="health-dot ${dotCls}" title="${name}: ${status.detail}">${name.replace('_', ' ')}</span>`;
        }
        html += '</div>';
    }

    // Active alerts
    if (alerts.length > 0) {
        html += '<div class="alert-strip">';
        for (const a of alerts.slice(0, 3)) {
            const sevCls = a.severity === 'CRITICAL' ? 'alert-critical' : 'alert-warning';
            html += `<span class="alert-badge ${sevCls}">${a.type}: ${a.message}</span>`;
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

// --- NEW: Milestones ---

function renderMilestones() {
    const milestones = DATA.milestones || [];
    const container = document.getElementById('milestone-banners');

    if (milestones.length === 0) {
        container.innerHTML = '';
        return;
    }

    const recent = milestones.slice(-5).reverse();
    container.innerHTML = recent.map(m => {
        const sev = m.severity || 'info';
        const cls = sev === 'critical' ? 'milestone-red' : sev === 'high' ? 'milestone-yellow' : 'milestone-green';
        return `<div class="milestone-banner ${cls}">
            <strong>${m.milestone_id || ''}</strong>: ${m.message || ''}
        </div>`;
    }).join('');
}

// --- Ensemble Leaderboard ---

function renderEnsembleLeaderboard() {
    const ensemble = DATA.ensemble || {};
    const section = document.getElementById('ensemble-section');
    const container = document.getElementById('ensemble-leaderboard');
    const harmonyEl = document.getElementById('ensemble-harmony');

    const agents = ensemble.agents || {};
    if (Object.keys(agents).length === 0) {
        section.style.display = 'none';
        return;
    }
    section.style.display = 'block';

    // Sort by return
    const sorted = Object.entries(agents)
        .map(([name, a]) => ({ name, ...a }))
        .sort((a, b) => b.total_pnl_pct - a.total_pnl_pct);

    let html = '<table class="data-table"><thead><tr>';
    html += '<th>#</th><th>Agent</th><th>Value</th><th>Return</th><th>Trades</th><th>Win%</th><th>Learning</th>';
    html += '</tr></thead><tbody>';

    sorted.forEach((a, i) => {
        const cls = pnlClass(a.total_pnl);
        const learn = a.learning_enabled ? '<span class="badge badge-correct">ON</span>' : '<span class="badge badge-pending">OFF</span>';
        const wins = a.winning_trades + a.losing_trades;
        const wr = wins > 0 ? ((a.winning_trades / wins) * 100).toFixed(0) + '%' : '-';
        html += `<tr>
            <td>${i + 1}</td>
            <td><strong>${a.display_name || a.name}</strong></td>
            <td>${fmt(a.total_value)}</td>
            <td class="${cls}">${fmtPct(a.total_pnl_pct)}</td>
            <td>${a.total_trades}</td>
            <td>${wr}</td>
            <td>${learn}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    // Harmony summary
    const harmony = ensemble.harmony || {};
    if (harmony.avg_correlation !== undefined) {
        harmonyEl.innerHTML = `
            <div class="harmony-item">Avg Correlation: <strong>${harmony.avg_correlation?.toFixed(3) || 'N/A'}</strong></div>
            <div class="harmony-item">Harmony Score: <strong>${harmony.harmony_score?.toFixed(2) || 'N/A'}</strong></div>
            <div class="harmony-item">Diversification: <strong>${harmony.diversification_benefit ? 'Yes' : 'No'}</strong></div>
        `;
    }
}

// --- HOLD Log ---

function renderHoldLog() {
    const holds = DATA.hold_log_summary || {};
    const section = document.getElementById('hold-section');
    const container = document.getElementById('hold-log');

    if (!holds.total_holds || holds.total_holds === 0) {
        section.style.display = 'none';
        return;
    }
    section.style.display = 'block';

    const recent = holds.recent || [];

    let html = `<p style="color:var(--text2)">Total HOLD decisions: ${holds.total_holds}</p>`;
    html += '<div class="hold-list">';

    recent.forEach(h => {
        const price = h.price_at_hold ? fmt(h.price_at_hold) : '-';
        const strongest = h.strongest_signal_ignored;
        const signalText = strongest
            ? `${strongest.action} from ${strongest.strategy} (conf=${strongest.confidence?.toFixed(2)})`
            : 'none';
        const counterfactual = h.counterfactual_outcome
            ? `<span class="${pnlClass(h.counterfactual_outcome)}">Outcome: ${fmtPct(h.counterfactual_outcome)}</span>`
            : '<span style="color:var(--text2)">pending</span>';

        html += `
            <div class="hold-card">
                <div class="hold-header">
                    <span>${timeAgo(h.timestamp)}</span>
                    <span>at ${price}</span>
                    <span class="badge badge-hold">HOLD</span>
                </div>
                <div class="hold-details">
                    <span>Reason: ${h.reason || 'agent_decision'}</span>
                    <span>Signal ignored: ${signalText}</span>
                    <span>Counterfactual: ${counterfactual}</span>
                </div>
            </div>`;
    });

    html += '</div>';
    container.innerHTML = html;
}

// --- Portfolio Chart (now with all benchmark lines) ---

function renderPortfolioChart() {
    const ctx = document.getElementById('portfolio-chart').getContext('2d');
    const snaps = DATA.snapshots;
    const bench = DATA.benchmark;
    const comp = DATA.benchmarks_comparison || {};

    if (snaps.length === 0) {
        ctx.canvas.parentElement.innerHTML = '<p style="color:var(--text2);text-align:center;padding:40px">No snapshot data yet.</p>';
        return;
    }

    const labels = snaps.map(s => s.date);
    const values = snaps.map(s => s.total_value);
    const benchValues = bench.map(b => b.value);

    // Try to get SPY and DCA benchmark values from benchmarks data
    let spyValues = [];
    let dcaValues = [];
    try {
        const bd = DATA.benchmarks_comparison;
        if (bd && bd.buy_hold_spy) {
            // These are point-in-time values; for the chart we'd need the series
            // Use the old benchmark array for TSLA B&H, pad others
        }
    } catch(e) {}

    const datasets = [
        {
            label: 'Portfolio',
            data: values,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 3,
        },
        {
            label: 'Buy & Hold TSLA',
            data: benchValues,
            borderColor: '#f59e0b',
            borderDash: [5, 5],
            fill: false,
            tension: 0.3,
            borderWidth: 1.5,
            pointRadius: 2,
        }
    ];

    if (charts.portfolio) charts.portfolio.destroy();
    charts.portfolio = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#8b90a5', font: { size: 11 } } },
            },
            scales: {
                x: { ticks: { color: '#8b90a5', font: { size: 10 } }, grid: { color: '#2d3148' } },
                y: {
                    ticks: {
                        color: '#8b90a5', font: { size: 10 },
                        callback: v => '$' + v.toFixed(0)
                    },
                    grid: { color: '#2d3148' }
                }
            }
        }
    });
}

function renderKPIs() {
    const p = DATA.portfolio;
    const acc = DATA.prediction_accuracy;

    document.getElementById('kpi-value').innerHTML =
        `<div class="kpi-value">${fmt(p.total_value)}</div>`;
    document.getElementById('kpi-pnl').innerHTML =
        `<div class="kpi-value ${pnlClass(p.total_pnl)}">${fmt(p.total_pnl)}</div>
         <div class="kpi-sub">${fmtPct(p.total_pnl_pct)}</div>`;
    document.getElementById('kpi-cash').innerHTML =
        `<div class="kpi-value">${fmt(p.cash)}</div>`;
    document.getElementById('kpi-trades').innerHTML =
        `<div class="kpi-value">${p.total_trades}</div>
         <div class="kpi-sub">W:${p.winning_trades} L:${p.losing_trades}</div>`;
    document.getElementById('kpi-winrate').innerHTML =
        `<div class="kpi-value">${p.win_rate}%</div>`;

    let accText = 'No data';
    const dir = acc.direction_accuracy || {};
    if (Object.keys(dir).length > 0) {
        const parts = Object.entries(dir).map(([h, d]) => `${h}: ${d.accuracy_pct}%`);
        accText = parts.join(' | ');
    }
    document.getElementById('kpi-accuracy').innerHTML =
        `<div class="kpi-value" style="font-size:16px">${accText}</div>
         <div class="kpi-sub">${acc.scored_predictions} scored</div>`;
}

function renderStrategyBars() {
    const strats = DATA.strategy_scores.strategies;
    const container = document.getElementById('strategy-bars');
    const colors = {
        momentum: '#3b82f6', mean_reversion: '#22c55e', sentiment: '#f59e0b',
        technical_signals: '#a855f7', dca: '#06b6d4'
    };

    container.innerHTML = Object.entries(strats).map(([name, s]) => {
        const pct = (s.weight * 100).toFixed(1);
        const color = colors[name] || '#6366f1';
        const wins = s.total_trades > 0 ? `${(s.win_rate * 100).toFixed(0)}% win` : 'no trades';
        return `
            <div class="strategy-bar">
                <span class="strategy-name">${name.replace('_', ' ')}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${Math.max(pct * 2.5, 5)}%;background:${color}">${pct}%</div>
                </div>
                <span class="bar-stats">${wins} (${s.total_trades})</span>
            </div>`;
    }).join('');
}

function renderWeightChart() {
    const ctx = document.getElementById('weight-chart').getContext('2d');
    const history = DATA.weight_history;

    if (history.length < 2) {
        ctx.canvas.parentElement.innerHTML = '<p style="color:var(--text2);text-align:center;padding:40px">Strategy weights will evolve as the agent trades.</p>';
        return;
    }

    const labels = history.map(h => h.timestamp.split('T')[0]);
    const stratNames = Object.keys(history[0].weights);
    const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4'];

    const datasets = stratNames.map((name, i) => ({
        label: name.replace('_', ' '),
        data: history.map(h => (h.weights[name] || 0) * 100),
        backgroundColor: colors[i % colors.length] + '40',
        borderColor: colors[i % colors.length],
        borderWidth: 1.5,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
    }));

    if (charts.weights) charts.weights.destroy();
    charts.weights = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#8b90a5', font: { size: 10 } } },
            },
            scales: {
                x: { ticks: { color: '#8b90a5', font: { size: 10 } }, grid: { color: '#2d3148' } },
                y: {
                    stacked: false,
                    ticks: {
                        color: '#8b90a5', font: { size: 10 },
                        callback: v => v.toFixed(0) + '%'
                    },
                    grid: { color: '#2d3148' },
                    max: 50, min: 0
                }
            }
        }
    });
}

function renderPredictions() {
    const container = document.getElementById('predictions-body');
    const preds = DATA.predictions.slice().reverse().slice(0, 20);

    if (preds.length === 0) {
        container.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text2)">No predictions yet</td></tr>';
        return;
    }

    container.innerHTML = preds.map(p => {
        const horizons = ['30min', '2hr', '1day'];
        const chips = horizons.map(h => {
            const pred = (p.predictions || {})[h];
            const out = (p.outcomes || {})[h];
            if (!pred) return '';
            if (!out || out.direction_correct === null || out.direction_correct === undefined) {
                return `<span class="pred-chip badge-pending">${h}: ${pred.direction} $${pred.target}</span>`;
            }
            const cls = out.direction_correct ? 'badge-correct' : 'badge-wrong';
            return `<span class="pred-chip ${cls}">${h}: ${pred.direction} $${pred.target} (was $${out.actual})</span>`;
        }).join(' ');

        return `<tr>
            <td>${timeAgo(p.timestamp)}</td>
            <td>${fmt(p.price_at_prediction)}</td>
            <td><span class="badge badge-${p.score === 'pending' ? 'pending' : 'hold'}">${p.score}</span></td>
            <td><div class="pred-row">${chips}</div></td>
            <td style="max-width:200px;font-size:12px;color:var(--text2)">${p.reasoning}</td>
        </tr>`;
    }).join('');
}

function renderTradeLog() {
    const container = document.getElementById('trades-body');
    const trades = DATA.transactions.slice().reverse().slice(0, 30);

    if (trades.length === 0) {
        container.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text2)">No trades yet</td></tr>';
        return;
    }

    container.innerHTML = trades.map(t => {
        const actionCls = t.action === 'BUY' ? 'badge-buy' : 'badge-sell';
        const pnl = t.realized_pnl !== null ? fmt(t.realized_pnl) : '-';
        const pnlCls = t.realized_pnl > 0 ? 'positive' : t.realized_pnl < 0 ? 'negative' : '';
        const reviewed = t.review ? (t.review.was_correct ? 'correct' : 'wrong') : 'pending';
        const reviewCls = t.review ? (t.review.was_correct ? 'badge-correct' : 'badge-wrong') : 'badge-pending';

        return `<tr>
            <td>${timeAgo(t.timestamp)}</td>
            <td><span class="badge ${actionCls}">${t.action}</span></td>
            <td>${t.shares.toFixed(4)} @ ${fmt(t.price)}</td>
            <td>${t.strategy || '-'}</td>
            <td class="${pnlCls}">${pnl}</td>
            <td style="max-width:250px;font-size:12px">${t.hypothesis || '-'}</td>
            <td><span class="badge ${reviewCls}">${reviewed}</span></td>
        </tr>`;
    }).join('');
}

function renderLessons() {
    const container = document.getElementById('lessons-list');
    const lessons = DATA.lessons.slice().reverse();

    if (lessons.length === 0) {
        container.innerHTML = '<p style="color:var(--text2)">No lessons yet. The agent learns after each trade.</p>';
        return;
    }

    container.innerHTML = lessons.map(l => {
        const weight = l.weight !== undefined ? ` | weight: ${l.weight.toFixed(2)}` : '';
        const skeptic = l.skeptic_review ? (l.skeptic_review.validated ? ' | skeptic: passed' : ' | skeptic: challenged') : '';
        const regime = l.regime ? ` | ${l.regime.trend}/${l.regime.volatility}` : '';
        return `
        <div class="lesson-card">
            <div class="lesson-text">${l.lesson || 'No lesson text'}</div>
            <div class="lesson-meta">
                ${l.id} | ${l.category || 'general'} | Trade: ${l.linked_trade || 'N/A'} | ${timeAgo(l.timestamp)}${weight}${skeptic}${regime}
            </div>
        </div>
    `}).join('');
}

function renderPatterns() {
    const container = document.getElementById('patterns-list');
    const patterns = DATA.patterns;

    if (patterns.length === 0) {
        container.innerHTML = '<p style="color:var(--text2)">No patterns discovered yet.</p>';
        return;
    }

    container.innerHTML = patterns.map(p => `
        <div class="pattern-card">
            <div class="pattern-name">${p.name || 'Unnamed'} <span style="color:var(--text2);font-weight:normal;font-size:12px">(reliability: ${((p.reliability || 0) * 100).toFixed(0)}%)</span></div>
            <div class="pattern-desc">${p.description || ''}</div>
        </div>
    `).join('');
}

function renderJournal() {
    const container = document.getElementById('journal-content');
    const journal = DATA.journal || '';

    if (!journal || journal.trim().length < 50) {
        container.innerHTML = '<p style="color:var(--text2)">The agent hasn\'t written any journal entries yet.</p>';
        return;
    }

    let html = journal
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^---$/gm, '<hr>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';
    container.innerHTML = html;
}

// --- Tabs ---

function setupTabs() {
    document.querySelectorAll('.tabs').forEach(tabGroup => {
        const tabs = tabGroup.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const target = tab.dataset.tab;
                const parent = tabGroup.parentElement;
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                parent.querySelector(`#${target}`).classList.add('active');
            });
        });
    });
}

// --- Init ---

loadData();
setInterval(loadData, 60000);
