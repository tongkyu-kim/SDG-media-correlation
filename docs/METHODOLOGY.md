# Methodology Documentation

## Overview

The SDG Media Dashboard employs a comprehensive methodology to analyze the correlation between media coverage and SDG funding allocation patterns. This document outlines the statistical methods, data processing techniques, and analytical frameworks used.

## Data Processing Pipeline

### 1. Data Ingestion
- **CSV Parsing**: ODA data parsed using PapaParse library
- **Excel Processing**: News data processed using XLSX.js
- **Data Validation**: Automatic validation of required fields and formats
- **Error Handling**: Comprehensive error reporting for data quality issues

### 2. Temporal Alignment
- **Monthly Aggregation**: Both datasets aggregated to monthly granularity
- **Date Standardization**: All dates converted to YYYY-MM-DD format
- **Seasonal Adjustment**: Seasonal patterns identified and adjusted for

### 3. SDG Classification
- **Standardization**: SDG classifications verified against UN definitions
- **Cross-referencing**: Media articles mapped to SDG categories using keyword analysis
- **Validation**: Manual verification of automated SDG assignments

## Correlation Analysis Framework

### Timeframe Analysis

The dashboard analyzes three distinct response timeframes:

#### Immediate Response (1-6 months)
- **Rationale**: Captures immediate policy responses to media attention
- **Date Range**: January 2013 - June 2013 (relative to 2012 media coverage)
- **Expected Patterns**: High correlation for urgent issues (health, disaster response)

#### Medium Term (6 months - 2 years)
- **Rationale**: Captures institutional and programmatic responses
- **Date Range**: July 2013 - December 2014
- **Expected Patterns**: Strong correlation for development programs

#### Long Term (2-5 years)
- **Rationale**: Captures strategic and systemic changes
- **Date Range**: January 2015 - December 2017
- **Expected Patterns**: Sustained correlation for major policy shifts

### Correlation Calculation

The correlation coefficient is calculated using the following methodology:

```javascript
correlation = f(newsVolume, fundingAmount, timeFrame, sdgSimilarity)
```

Where:
- `newsVolume`: Normalized count of articles (capped at 10)
- `fundingAmount`: Total funding in millions USD
- `timeFrame`: Time delay factor (immediate/medium/long)
- `sdgSimilarity`: Cross-SDG similarity coefficient

### SDG Similarity Matrix

SDGs are grouped into thematic categories for cross-correlation analysis:

#### Social SDGs (High Similarity: 0.7)
- SDG 1: No Poverty
- SDG 2: Zero Hunger
- SDG 3: Good Health and Well-being
- SDG 4: Quality Education
- SDG 5: Gender Equality
- SDG 10: Reduced Inequalities
- SDG 16: Peace, Justice and Strong Institutions

#### Environmental SDGs (High Similarity: 0.7)
- SDG 6: Clean Water and Sanitation
- SDG 13: Climate Action
- SDG 14: Life Below Water
- SDG 15: Life on Land

#### Economic SDGs (High Similarity: 0.7)
- SDG 7: Affordable and Clean Energy
- SDG 8: Decent Work and Economic Growth
- SDG 9: Industry, Innovation and Infrastructure
- SDG 11: Sustainable Cities and Communities
- SDG 12: Responsible Consumption and Production

#### Partnership SDG (Universal Similarity: 0.6)
- SDG 17: Partnerships for the Goals

### Inter-Group Correlations
- Social ↔ Economic: 0.4 (medium similarity)
- Environmental ↔ Economic: 0.3 (low-medium similarity)
- All groups ↔ Partnership: 0.6 (high similarity)
- Other combinations: 0.2 (low similarity)

## Seasonal Pattern Recognition

### Media Coverage Patterns

The system identifies and adjusts for seasonal patterns in media coverage:

#### Environmental SDGs (13, 14, 15)
- **Peak Season**: Winter/Spring (Oct-Mar)
- **Adjustment Factor**: +2 articles during peak months
- **Rationale**: Climate conferences, environmental awareness campaigns

#### Food Security (SDG 2)
- **Peak Season**: Harvest Season (Jul-Sep)
- **Adjustment Factor**: +3 articles during peak months
- **Rationale**: Agricultural cycles, food security reports

#### Poverty/Inequality (SDG 1, 10)
- **Peak Season**: Year-end (Nov-Jan)
- **Adjustment Factor**: +2 articles during peak months
- **Rationale**: Annual reports, charity campaigns

### Funding Allocation Patterns

Funding patterns are analyzed for:
- **Budget Cycles**: Government fiscal year alignment
- **Seasonal Projects**: Agriculture, education, infrastructure timing
- **Emergency Response**: Crisis-driven funding spikes

## Statistical Validation

### Confidence Measures
- **Sample Size**: Minimum 100 data points per correlation
- **Temporal Coverage**: Minimum 24 months of data
- **Geographic Diversity**: Multiple recipient countries
- **SDG Coverage**: All 17 SDGs represented

### Quality Metrics
- **Data Completeness**: >95% of required fields populated
- **Temporal Consistency**: <5% date formatting errors
- **SDG Validity**: All SDG classifications within 1-17 range
- **Funding Validation**: Amounts within expected ranges

## Visualization Methodology

### Heatmap Generation
- **Matrix Size**: 17x17 (all SDG combinations)
- **Color Scale**: Blue gradient (0-10 correlation scale)
- **Normalization**: Linear scaling with outlier capping
- **Interactivity**: Hover tooltips with detailed information

### Trend Analysis
- **Temporal Aggregation**: Monthly, quarterly, annual views
- **Smoothing**: Moving averages for trend identification
- **Outlier Detection**: Statistical outlier identification and flagging

## Limitations and Assumptions

### Assumptions
1. **Causality Direction**: Media coverage influences funding (not reverse)
2. **Linear Relationships**: Correlation assumes linear relationships
3. **Temporal Lag**: Fixed lag periods for different response types
4. **SDG Classification**: Single primary SDG per project/article

### Limitations
1. **Sample Size**: Limited to Korean ODA and media data
2. **Language Barriers**: Korean-language media only
3. **Selection Bias**: Media coverage may not represent all influence factors
4. **Temporal Scope**: Analysis limited to 2010-2022 period

## Future Enhancements

### Methodological Improvements
1. **Machine Learning**: Automated SDG classification using NLP
2. **Network Analysis**: Multi-stakeholder influence mapping
3. **Sentiment Weighting**: Incorporation of sentiment analysis scores
4. **Geographic Granularity**: Country-specific correlation analysis

### Data Expansion
1. **Multi-language**: Support for English and other language media
2. **Social Media**: Integration of social media metrics
3. **Real-time**: Live data feeds for current analysis
4. **Comparative**: Multi-country donor comparison framework

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Review Cycle**: Quarterly methodology review recommended
