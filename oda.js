// oda.js
// All ODA data logic and visualizations will go here.

let odaDataCache = {};

const ODA_PAGE_SIZE = 200;
let odaCurrentPage = 1;
let odaCurrentYear = 2025;
let odaCurrentData = [];

function renderODAFilters() {
    // Remove filter bar if present
    const filterBar = document.getElementById('oda-filter-bar');
    if (filterBar) filterBar.remove();
}

function getODAFilters() {
    const bar = document.getElementById('oda-filter-bar');
    if (!bar) return {};
    return {
        country: bar.querySelector('#oda-filter-country').value,
        org: bar.querySelector('#oda-filter-org').value,
        type: bar.querySelector('#oda-filter-type').value,
        purpose: bar.querySelector('#oda-filter-purpose').value,
        sdg: bar.querySelector('#oda-filter-sdg').value,
        sortCol: bar.querySelector('#oda-sort-col').value,
        sortDir: bar.querySelector('#oda-sort-dir').value,
    };
}

function filterODAData(data) {
    const f = getODAFilters();
    let filtered = data.filter(row =>
        (!f.country || row.recipient_name === f.country) &&
        (!f.org || row.agency_name === f.org) &&
        (!f.type || row.aid_t === f.type) &&
        (!f.purpose || row.purpose_code === f.purpose) &&
        (!f.sdg || row.sdg_focus === f.sdg)
    );
    if (f.sortCol) {
        filtered = filtered.slice().sort((a, b) => {
            let va = a[f.sortCol] || '';
            let vb = b[f.sortCol] || '';
            if (f.sortCol === 'usd_commitment' || f.sortCol === 'interest1') {
                va = parseFloat(va) || 0;
                vb = parseFloat(vb) || 0;
            }
            if (va < vb) return f.sortDir === 'asc' ? -1 : 1;
            if (va > vb) return f.sortDir === 'asc' ? 1 : -1;
            return 0;
        });
    }
    return filtered;
}

function initODASection() {
    const odaSection = document.getElementById('oda-content');
    odaSection.innerHTML = `
        <div style="display: flex; gap: 2rem; align-items: flex-start; min-height: 500px;">
            <nav id="oda-year-nav" style="min-width: 120px;">
                <ul style="list-style:none; padding:0; margin:0;"></ul>
            </nav>
            <div style="flex:1;">
                <table id="oda-table" style="width:100%; border-collapse:collapse; background:#fff; box-shadow:0 2px 8px #0001; border-radius:8px; overflow:hidden;">
                    <thead style="background:#F2F2F0;">
                        <tr>
                            <th style="padding:0.7em 0.5em; text-align:left;">Country</th>
                            <th style="padding:0.7em 0.5em; text-align:left;">Organization</th>
                            <th style="padding:0.7em 0.5em; text-align:left;">ODA Type</th>
                            <th style="padding:0.7em 0.5em; text-align:right;">Amount</th>
                            <th style="padding:0.7em 0.5em; text-align:right;">Interest Rate</th>
                            <th style="padding:0.7em 0.5em; text-align:left;">Purpose Code</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td colspan="6" style="text-align:center; color:#aaa; padding:2em;">Select a year to view ODA data.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
    renderYearNav();
    // Show most recent year by default
    renderODAFilters(() => renderODATable(2025));
    if (odaDataCache[2025]) {
        renderODATable(2025);
        highlightYearLink(2025);
    } else {
        renderODATable(2025);
        highlightYearLink(2025);
    }
}

function highlightYearLink(year) {
    const nav = document.querySelector('#oda-year-nav ul');
    if (!nav) return;
    nav.querySelectorAll('.oda-year-link').forEach(l => {
        l.style.background = (l.dataset.year == year) ? '#e6f6fb' : 'none';
    });
}

function renderYearNav() {
    // Years from 2025 down to 2010
    const years = [];
    for (let y = 2025; y >= 2010; y--) years.push(y);
    const nav = document.querySelector('#oda-year-nav ul');
    nav.innerHTML = years.map(year => `
        <li><a href="#" class="oda-year-link" data-year="${year}" style="display:block;padding:0.6em 0.5em;color:#009EDB;font-weight:600;text-decoration:none;border-radius:4px;margin-bottom:0.2em;transition:background 0.2s;">${year}</a></li>
    `).join('');
    // Highlight and click handler
    nav.querySelectorAll('.oda-year-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            highlightYearLink(this.dataset.year);
            renderODATable(this.dataset.year);
        });
    });
}

function loadODADataForYear(year, callback) {
    if (odaDataCache[year]) {
        callback(odaDataCache[year]);
        return;
    }
    const url = `src/oda/processed/oda_${year}.json`;
    fetch(url)
        .then(res => {
            if (!res.ok) throw new Error('Failed to load ODA data for year ' + year);
            return res.json();
        })
        .then(data => {
            odaDataCache[year] = data;
            callback(data);
        })
        .catch(err => {
            console.error(err);
            callback([]);
        });
}

function renderODAPagination(total, page, onPage) {
    const odaSection = document.getElementById('oda-content');
    let nav = document.getElementById('oda-pagination');
    if (!nav) {
        nav = document.createElement('div');
        nav.id = 'oda-pagination';
        nav.style = 'margin:1.5em 0 0 0; text-align:center;';
        odaSection.appendChild(nav);
    }
    const pageCount = Math.ceil(total / ODA_PAGE_SIZE);
    if (pageCount <= 1) { nav.innerHTML = ''; return; }
    let html = '';
    for (let i = 1; i <= pageCount; i++) {
        html += `<a href="#" data-page="${i}" style="display:inline-block;padding:0.4em 0.9em;margin:0 0.2em;border-radius:4px;${i===page?'background:#009EDB;color:#fff;':'background:#f2f2f0;color:#009EDB;'}text-decoration:none;font-weight:600;">
        <li><a href="#" class="oda-year-link" data-year="${year}" style="display:block;padding:0.6em 0.5em;color:#009EDB;font-weight:600;text-decoration:none;border-radius:4px;margin-bottom:0.2em;transition:background 0.2s;">${year}</a></li>
    `).join('');
    // Highlight and click handler
    nav.querySelectorAll('.oda-year-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            highlightYearLink(this.dataset.year);
            renderODATable(this.dataset.year);
        });
    });
}

function loadODADataForYear(year, callback) {
    if (odaDataCache[year]) {
        callback(odaDataCache[year]);
        return;
    }
    const url = `src/oda/processed/oda_${year}.json`;
    fetch(url)
        .then(res => {
            if (!res.ok) throw new Error('Failed to load ODA data for year ' + year);
            return res.json();
        })
        .then(data => {
            odaDataCache[year] = data;
            callback(data);
        })
        .catch(err => {
            console.error(err);
            callback([]);
        });
}

function renderODAPagination(total, page, onPage) {
    const odaSection = document.getElementById('oda-content');
    let nav = document.getElementById('oda-pagination');
    if (!nav) {
        nav = document.createElement('div');
        nav.id = 'oda-pagination';
        nav.style = 'margin:1.5em 0 0 0; text-align:center;';
        odaSection.appendChild(nav);
    }
    const pageCount = Math.ceil(total / ODA_PAGE_SIZE);
    if (pageCount <= 1) { nav.innerHTML = ''; return; }
    let html = '';
    for (let i = 1; i <= pageCount; i++) {
        html += `<a href="#" data-page="${i}" style="display:inline-block;padding:0.4em 0.9em;margin:0 0.2em;border-radius:4px;${i===page?'background:#009EDB;color:#fff;':'background:#f2f2f0;color:#009EDB;'}text-decoration:none;font-weight:600;">${i}</a>`;
    }
    nav.innerHTML = html;
    nav.querySelectorAll('a').forEach(a => {
        a.onclick = (e) => {
            e.preventDefault();
            onPage(Number(a.dataset.page));
        };
    });
}

function renderSDGSummaryTable(filteredData) {
    // Remove SDG summary table entirely
    let sdgDiv = document.getElementById('oda-sdg-summary');
    if (sdgDiv) sdgDiv.innerHTML = '';
}

function renderODATable(year, page = 1) {
    odaCurrentYear = year;
    odaCurrentPage = page;
    const tbody = document.querySelector('#oda-table tbody');
    const thead = document.querySelector('#oda-table thead tr');
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#aaa; padding:2em;">Loading ODA data...</td></tr>`;
    loadODADataForYear(year, (data) => {
        odaCurrentData = filterODAData(data);
        const filtered = odaCurrentData;
        if (!filtered.length) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#aaa; padding:2em;">No ODA data for <b>${year}</b>.</td></tr>`;
            renderODAPagination(0, 1, () => {});
            return;
        }
        // Add filter triangles to headers
        const headers = [
            { key: 'recipient_name', label: 'Country' },
            { key: 'agency_name', label: 'Organization' },
            { key: 'aid_t', label: 'ODA Type' },
            { key: 'usd_commitment', label: 'Amount' },
            { key: 'interest1', label: 'Interest Rate' },
            { key: 'purpose_code', label: 'Purpose Code' }
        ];
        if (thead) {
            thead.innerHTML = headers.map(h =>
                `<th style="padding:0.7em 0.5em; text-align:left; position:relative;">
                    ${h.label}
                    <span class="oda-header-filter" data-col="${h.key}" style="cursor:pointer; font-size:1em; margin-left:0.3em;">&#9660;</span>
                </th>`
            ).join('');
        }
        // Attach filter dropdown logic
        document.querySelectorAll('.oda-header-filter').forEach(el => {
            el.onclick = function(e) {
                e.stopPropagation();
                showHeaderFilterDropdown(this.dataset.col, data, this);
            };
        });
        // Remove any open dropdown on click elsewhere
        document.body.onclick = function() {
            const dd = document.getElementById('oda-header-filter-dropdown');
            if (dd) dd.remove();
        };
        const start = (page-1)*ODA_PAGE_SIZE;
        const end = start + ODA_PAGE_SIZE;
        const pageData = filtered.slice(start, end);
        tbody.innerHTML = pageData.map(row => {
            return `
                <tr>
                    <td style="padding:0.5em 0.3em;">${row.recipient_name || ''}</td>
                    <td style="padding:0.5em 0.3em;">${row.agency_name || ''}</td>
                    <td style="padding:0.5em 0.3em;">${row.aid_t || ''}</td>
                    <td style="padding:0.5em 0.3em; text-align:right;">${row.usd_commitment ? Number(row.usd_commitment).toLocaleString() : ''}</td>
                    <td style="padding:0.5em 0.3em; text-align:right;">${row.interest1 || ''}</td>
                    <td style="padding:0.5em 0.3em;">${row.purpose_code || ''}</td>
                </tr>
            `;
        }).join('');
        renderODAPagination(filtered.length, page, (newPage) => renderODATable(year, newPage));
    });
}

function showHeaderFilterDropdown(col, data, anchor) {
    let dd = document.getElementById('oda-header-filter-dropdown');
    if (dd) dd.remove();
    dd = document.createElement('div');
    dd.id = 'oda-header-filter-dropdown';
    dd.style = 'position:absolute;z-index:1000;background:#fff;border:1px solid #ccc;box-shadow:0 2px 8px #0002;padding:0.5em 1em;min-width:180px;max-height:300px;overflow:auto;top:1.8em;left:0;';
    // Unique values
    const values = Array.from(new Set(data.map(r => (r[col]||'').trim()))).filter(v=>v).sort();
    dd.innerHTML = `<div style='font-weight:bold;margin-bottom:0.5em;'>Filter by:</div>` +
        `<input type='text' id='oda-header-filter-search' placeholder='Search...' style='width:100%;margin-bottom:0.5em;padding:0.2em;'>` +
        `<div style='max-height:200px;overflow:auto;'>` +
        values.map(v => `<div><label><input type='checkbox' class='oda-header-filter-chk' value="${v.replace(/"/g,'&quot;')}"> ${v}</label></div>`).join('') +
        `</div><div style='margin-top:0.7em;text-align:right;'><button id='oda-header-filter-apply'>Apply</button> <button id='oda-header-filter-clear'>Clear</button></div>`;
    anchor.parentElement.appendChild(dd);
    // Search filter
    dd.querySelector('#oda-header-filter-search').oninput = function() {
        const q = this.value.toLowerCase();
        dd.querySelectorAll('.oda-header-filter-chk').forEach(chk => {
            const label = chk.parentElement;
            label.style.display = chk.value.toLowerCase().includes(q) ? '' : 'none';
        });
    };
    // Apply filter
    dd.querySelector('#oda-header-filter-apply').onclick = function(e) {
        e.stopPropagation();
        const checked = Array.from(dd.querySelectorAll('.oda-header-filter-chk:checked')).map(c=>c.value);
        setHeaderFilter(col, checked);
        dd.remove();
    };
    // Clear filter
    dd.querySelector('#oda-header-filter-clear').onclick = function(e) {
        e.stopPropagation();
        setHeaderFilter(col, []);
        dd.remove();
    };
}

let odaHeaderFilters = {};
function setHeaderFilter(col, values) {
    odaHeaderFilters[col] = values;
    renderODATable(odaCurrentYear, 1);
}
function filterODAData(data) {
    // Apply header filters
    let filtered = data;
    Object.entries(odaHeaderFilters).forEach(([col, vals]) => {
        if (vals && vals.length) {
            filtered = filtered.filter(row => vals.includes((row[col]||'').trim()));
        }
    });
    return filtered;
}

// ODA Board: Annual ODA Budget Table for Korea

const ODA_YEARS = Array.from({length: 2025-2010+1}, (_,i)=>2010+i);
const ODA_DATA_PATH = 'src/oda/processed/oda_';

async function loadAllODAData() {
    let all = [];
    for (const year of ODA_YEARS) {
        try {
            const res = await fetch(`${ODA_DATA_PATH}${year}.json`);
            if (!res.ok) continue;
            const data = await res.json();
            // Add year to each row if missing
            data.forEach(row => { if (!row.year) row.year = String(year); });
            all = all.concat(data);
        } catch (e) {
            // skip missing years
        }
    }
    return all;
}

function renderODABoardTable(data) {
    // Group by year
    const years = [...new Set(data.map(row => row.year))].sort((a,b)=>b-a);
    let html = `
        <table style="width:100%;border-collapse:collapse;margin-top:1em;">
            <thead style="background:#F2F2F0;">
                <tr>
                    <th style="padding:0.7em 0.5em;">Year</th>
                    <th style="padding:0.7em 0.5em;">Nation</th>
                    <th style="padding:0.7em 0.5em;">Type</th>
                    <th style="padding:0.7em 0.5em;text-align:right;">Amount (USD)</th>
                    <th style="padding:0.7em 0.5em;">Purpose Code</th>
                </tr>
            </thead>
            <tbody>
    `;
    for (const year of years) {
        const rows = data.filter(row => row.year == year);
        for (const row of rows) {
            html += `
                <tr>
                    <td style="padding:0.5em 0.3em;">${row.year}</td>
                    <td style="padding:0.5em 0.3em;">${row.recipient_name || ''}</td>
                    <td style="padding:0.5em 0.3em;">${row.aid_t || ''}</td>
                    <td style="padding:0.5em 0.3em;text-align:right;">${row.usd_commitment ? Number(row.usd_commitment).toLocaleString() : ''}</td>
                    <td style="padding:0.5em 0.3em;">${row.purpose_code || ''}</td>
                </tr>
            `;
        }
    }
    html += '</tbody></table>';
    return html;
}

async function drawODABoard() {
    const container = document.getElementById('oda-board');
    container.innerHTML = '<div style="padding:2em;text-align:center;">Loading ODA data...</div>';
    const data = await loadAllODAData();
    if (!data.length) {
        container.innerHTML = '<div style="padding:2em;text-align:center;color:#aaa;">No ODA data found.</div>';
        return;
    }
    container.innerHTML = renderODABoardTable(data);
}

// Only run if #oda-board exists (ODA subpage)
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('oda-board')) drawODABoard();
});

// Only run if ODA page is active on load
if (window.location.hash.replace('#','') === 'oda') {
    document.addEventListener('DOMContentLoaded', initODASection);
}
// Also run when navigating to ODA page
window.addEventListener('hashchange', function() {
    if (window.location.hash.replace('#','') === 'oda') {
        initODASection();
    }
});
