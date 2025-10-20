# Climate Risk Analysis

A Python tool for collecting climate risk data from corporate documents using Perplexity's sonar-deep-research API.

## Features

- Automated data collection from corporate websites and regulatory databases
- Integration with Perplexity's sonar-deep-research model
- Extraction of real URLs from search results
- Support for multiple document types (annual reports, sustainability reports, SEC filings, etc.)
- CSV output for further analysis

## Setup

1. Install dependencies:
```bash
pip install pandas requests python-dotenv openpyxl
```

2. Set up your Perplexity API key:
```bash
# Create a .env file
echo "PERPLEXITY_API_KEY=your_api_key_here" > .env
```

3. Run the collector:
```bash
python climate_collector.py
```

## Usage

The script will:
1. Load company data from `20Companies.xlsx`
2. Generate prompts optimized for sonar-deep-research
3. Extract real URLs from search results
4. Save results to `RESULTS_DIR/`

## Output

- `{ISIN}_sonar_reasoning_response.json` - Full API response
- `{ISIN}_sonar_reasoning_data_sources.csv` - Extracted URLs and metadata
- `{ISIN}_raw_response.txt` - Raw response for debugging

## Model

Uses Perplexity's `sonar-deep-research` model for advanced reasoning and URL extraction from search results.
