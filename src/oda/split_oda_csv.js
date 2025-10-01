// Node.js script to split oda_korea_dataset.csv into year-based JSON files (2010-2025)
// Usage: node split_oda_csv.js

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

const INPUT = path.join(__dirname, 'oda_korea_dataset.csv');
const OUTDIR = path.join(__dirname, 'processed');
const YEARS = Array.from({length: 2025-2010+1}, (_,i) => (2010+i).toString());
const FIELDS = [
  'year', 'recipient_name', 'agency_name', 'aid_t', 'usd_commitment', 'interest1', 'purpose_code', 'sdg_focus'
];

if (!fs.existsSync(OUTDIR)) fs.mkdirSync(OUTDIR);

const yearData = {};
YEARS.forEach(y => yearData[y] = []);

fs.createReadStream(INPUT)
  .pipe(csv())
  .on('data', (row) => {
    const y = row.year;
    if (YEARS.includes(y)) {
      const filtered = {};
      FIELDS.forEach(f => filtered[f] = row[f] || '');
      yearData[y].push(filtered);
    }
  })
  .on('end', () => {
    YEARS.forEach(y => {
      const outPath = path.join(OUTDIR, `oda_${y}.json`);
      fs.writeFileSync(outPath, JSON.stringify(yearData[y], null, 2));
      console.log(`Wrote ${yearData[y].length} records to ${outPath}`);
    });
    console.log('Done.');
  });
