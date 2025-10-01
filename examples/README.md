# Example Data

This directory contains sample data files that demonstrate the expected format and structure for the SDG Media Dashboard.

## Files

### `sample_oda_data.csv`
Sample ODA (Official Development Assistance) dataset showing the required CSV format with:
- Project information
- Funding amounts
- SDG classifications
- Recipient countries
- Monthly temporal granularity

### Usage
These files can be used to:
1. Test the dashboard functionality
2. Understand the required data format
3. Validate your own data structure
4. Demonstrate the analysis capabilities

## Getting Started

1. Copy these files to the appropriate directories:
   ```bash
   cp examples/sample_oda_data.csv src/oda/oda_korea_dataset.csv
   ```

2. Open `index.html` in your browser

3. Click "Load Local Data Files" to see the dashboard in action

## Data Characteristics

The sample data includes:
- **Time Period**: 2013 (immediate response period)
- **Countries**: 10 different recipient countries
- **SDGs**: Coverage of multiple SDG categories
- **Funding Range**: $1.6M - $5.2M per project
- **Geographic Diversity**: Asia, Africa, Latin America

This sample demonstrates the correlation analysis capabilities and visualization features of the dashboard.
