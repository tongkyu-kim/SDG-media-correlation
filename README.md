# SDG Media Dashboard

## Overview

The SDG Media Dashboard is an interactive web application that analyzes the correlation between Korean media coverage and Sustainable Development Goals (SDG) funding allocation patterns. This tool provides monthly analysis of how media attention influences development funding across different timeframes.

## Features

### 📊 Interactive Heatmaps
- **Immediate Response (1-6 months)**: Shows short-term funding allocation patterns following media coverage
- **Medium Term (6 months - 2 years)**: Tracks medium-term funding responses
- **Long Term (2-5 years)**: Analyzes long-term funding allocation trends

### 📈 Advanced Analytics
- **17x17 SDG Correlation Matrix**: Visual representation of cross-SDG funding relationships
- **Monthly Data Granularity**: Precise tracking of funding allocation patterns
- **Seasonal Pattern Recognition**: Identifies seasonal trends in media coverage and funding
- **Impact Scoring**: Measures the intensity and type of media coverage

### 🎯 Key Visualizations
- SDG funding distribution pie charts with official UN colors
- Timeline analysis of ODA projects
- Regional distribution mapping
- Media sentiment analysis
- Correlation scatter plots

## Data Sources

### Required Input Files
1. **ODA Dataset (CSV)**: Korean Official Development Assistance data
   - Location: \`src/oda/oda_korea_dataset.csv\`
   - Format: CSV with project details, funding amounts, SDG classifications

2. **News Dataset (XLSX)**: Korean media coverage data
   - Location: \`src/raw/2012.xlsx\`
   - Format: Excel with article dates, titles, categories, keywords

## Project Structure

\`\`\`
SDG-media-dashboard/
├── index.html                 # Main dashboard application
├── README.md                  # This file
├── src/                       # Data sources
│   ├── oda/                   # ODA dataset files
│   ├── processed/             # Processed data files
│   └── raw/                   # Raw data files
├── assets/                    # Static assets
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript modules
│   └── images/                # Images and icons
├── docs/                      # Documentation
└── examples/                  # Example data and configurations
\`\`\`

## Quick Start

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/[username]/SDG-media-dashboard.git
cd SDG-media-dashboard
\`\`\`

### 2. Prepare Data Files
- Place your ODA dataset in \`src/oda/oda_korea_dataset.csv\`
- Place your news dataset in \`src/raw/2012.xlsx\`

### 3. Open the Dashboard
- Open \`index.html\` in a modern web browser
- Click "Load Local Data Files" to use sample data
- Or upload your own data files using the upload interface

## Technical Specifications

### Dependencies
- **Plotly.js v2.26.0**: Interactive charting library
- **Chart.js**: Additional chart components
- **XLSX.js v0.18.5**: Excel file parsing
- **PapaParse v5.4.1**: CSV file parsing
- **Font Awesome 6.4.0**: Icons
- **Inter Font**: Typography

### Browser Compatibility
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Contributing

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/amazing-feature\`)
3. Commit your changes (\`git commit -m 'Add amazing feature'\`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Compatibility**: Modern web browsers with ES6 support
