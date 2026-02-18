// MonopolyTrader Dashboard

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
    renderKPIs();
    renderPortfolioChart();
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
        detail = `Last update ${ageMin}m ago — data may be outdated`;
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

    // Prediction accuracy
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

function renderPortfolioChart() {
    const ctx = document.getElementById('portfolio-chart').getContext('2d');
    const snaps = DATA.snapshots;
    const bench = DATA.benchmark;

    if (snaps.length === 0) {
        ctx.canvas.parentElement.innerHTML = '<p style="color:var(--text2);text-align:center;padding:40px">No snapshot data yet. Run the agent for a day to see the chart.</p>';
        return;
    }

    const labels = snaps.map(s => s.date);
    const values = snaps.map(s => s.total_value);
    const benchValues = bench.map(b => b.value);

    if (charts.portfolio) charts.portfolio.destroy();
    charts.portfolio = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
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
                    borderColor: '#8b90a5',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    borderWidth: 1.5,
                    pointRadius: 2,
                }
            ]
        },
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

    container.innerHTML = lessons.map(l => `
        <div class="lesson-card">
            <div class="lesson-text">${l.lesson || 'No lesson text'}</div>
            <div class="lesson-meta">
                ${l.id} | ${l.category || 'general'} | Trade: ${l.linked_trade || 'N/A'} | ${timeAgo(l.timestamp)}
            </div>
        </div>
    `).join('');
}

function renderPatterns() {
    const container = document.getElementById('patterns-list');
    const patterns = DATA.patterns;

    if (patterns.length === 0) {
        container.innerHTML = '<p style="color:var(--text2)">No patterns discovered yet. Patterns emerge after several trades.</p>';
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

    // Simple markdown rendering
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
// Auto-refresh every 60 seconds
setInterval(loadData, 60000);
