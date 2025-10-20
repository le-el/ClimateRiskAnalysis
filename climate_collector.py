#!/usr/bin/env python3
"""
Sonar-Reasoning API Climate Risk Data Collector
==============================================

This script generates prompts and makes API calls to Perplexity's sonar-reasoning model
to find REAL, VALID URLs from actual company websites and regulatory databases.

The script will:
1. Generate prompts optimized for sonar-reasoning model
2. Make API calls to Perplexity with model="sonar-reasoning"
3. Process responses and save real URLs
4. Focus on actual company websites and SEC filings
"""

import pandas as pd
import json
import os
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from dotenv import load_dotenv

class SonarReasoningAPIClimateRiskDataCollector:
    def __init__(self, companies_file: str = "20Companies.xlsx", 
                 prompts_dir: str = "PROMPTS_DIR", 
                 results_dir: str = "RESULTS_DIR",
                 api_key: Optional[str] = None):
        # Load environment variables from .env file
        load_dotenv()
        
        self.companies_file = companies_file
        self.prompts_dir = Path(prompts_dir)
        self.results_dir = Path(results_dir)
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        
        # Create directories
        self.prompts_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load company data
        self.companies_df = self._load_companies_data()
        
        # Historical years to focus on
        self.historical_years = ['2022', '2023', '2024']
        
        # API configuration
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-deep-research"
        
    def _load_companies_data(self) -> pd.DataFrame:
        """Load and clean company data from the Excel file."""
        # Read Excel file directly using pandas
        df = pd.read_excel(self.companies_file)
        
        # Clean the data
        df = df.dropna()
        df['Type'] = df['Type'].astype(str)
        
        return df
    
    def _extract_company_info(self, row: pd.Series) -> Dict[str, Any]:
        """Extract and structure company information from a dataframe row."""
        isin = row['Type']
        company_name = row['NAME']
        
        # Extract country from ISIN
        country_code = isin[:2]
        country_map = {
            'US': 'United States',
            'CA': 'Canada', 
            'HK': 'Hong Kong',
            'FR': 'France',
            'CH': 'Switzerland',
            'JP': 'Japan'
        }
        country = country_map.get(country_code, country_code)
        
        # Extract ticker (simplified - using company name for now)
        ticker = company_name.replace(' ', '').replace('.', '').replace(',', '')[:10]
        
        # Determine HQ city based on country
        hq_city_map = {
            'United States': 'New York',
            'Canada': 'Toronto',
            'Hong Kong': 'Hong Kong',
            'France': 'Paris',
            'Switzerland': 'Zurich',
            'Japan': 'Tokyo'
        }
        hq_city = hq_city_map.get(country, 'Unknown')
        
        # Determine primary domain (simplified)
        domain_map = {
            'United States': 'com',
            'Canada': 'ca',
            'Hong Kong': 'hk', 
            'France': 'fr',
            'Switzerland': 'ch',
            'Japan': 'jp'
        }
        domain_ext = domain_map.get(country, 'com')
        primary_domain = f"{company_name.lower().replace(' ', '').replace('.', '').replace(',', '')}.{domain_ext}"
        
        # Determine reporting languages
        language_map = {
            'United States': ['English'],
            'Canada': ['English', 'French'],
            'Hong Kong': ['English', 'Chinese'],
            'France': ['French', 'English'],
            'Switzerland': ['German', 'French', 'English'],
            'Japan': ['Japanese', 'English']
        }
        reporting_languages = language_map.get(country, ['English'])
        
        # Determine key regulators
        regulator_map = {
            'United States': ['SEC', 'CFTC'],
            'Canada': ['CSA', 'OSC'],
            'Hong Kong': ['SFC', 'HKEX'],
            'France': ['AMF', 'ACPR'],
            'Switzerland': ['FINMA', 'SIX'],
            'Japan': ['FSA', 'TSE']
        }
        key_regulators = regulator_map.get(country, ['SEC'])
        
        return {
            'company_name': company_name,
            'isin': isin,
            'ticker': ticker,
            'country': country,
            'hq_city': hq_city,
            'industry_level2': row['LEVEL2 SECTOR NAME'],
            'industry_level3': row['LEVEL3 SECTOR NAME'],
            'industry_level4': row['LEVEL4 SECTOR NAME'],
            'industry_level5': row['LEVEL5 SECTOR NAME'],
            'geography_description': row['GEOGRAPHIC DESCR.'],
            'primary_domain': primary_domain,
            'ir_url': f"https://investors.{primary_domain}",
            'localized_domains': [f"{primary_domain.split('.')[0]}.{domain_ext}"],
            'reporting_languages': reporting_languages,
            'key_regulators': key_regulators,
            'last_known_reports': {
                'annual': '2024',
                'sustainability': '2024',
                'tcfd': '2024'
            },
            'historical_years': self.historical_years,
            'notes': f"Company in {row['LEVEL2 SECTOR NAME']} sector - Using sonar-reasoning API"
        }
    
    def generate_sonar_reasoning_prompt(self, company_info: Dict[str, Any]) -> str:
        """Generate a prompt specifically optimized for Perplexity's sonar-deep-research model."""
        
        prompt = f"""EXTRACT ONLY REAL URLs FROM SEARCH RESULTS: You must extract ONLY URLs that appear in the search results provided to you. Do NOT construct, generate, or predict any URLs. Only return URLs that are explicitly mentioned in the search results.

        COMPANY DATA:
        - Company Name: {company_info['company_name']}
        - ISIN: {company_info['isin']}
        - Ticker: {company_info['ticker']}
        - Country: {company_info['country']}
        - Industry: {company_info['industry_level2']} - {company_info['industry_level3']}
        - Primary Domain: {company_info['primary_domain']}
        - Key Regulators: {', '.join(company_info['key_regulators'])}

        CRITICAL INSTRUCTIONS:
        1. ONLY extract URLs that are explicitly mentioned in the search results
        2. DO NOT construct, generate, or predict any URLs
        3. DO NOT use logical patterns to create URLs
        4. DO NOT assume URL structures or naming conventions
        5. ONLY return URLs that are directly visible in the search results

        WHAT TO LOOK FOR IN SEARCH RESULTS:
        - Direct links to PDF documents
        - Links to HTML pages containing documents
        - SEC EDGAR filing URLs
        - Company investor relations page URLs
        - Any URLs that lead to actual documents
        - URLs mentioned in search result snippets

        DOCUMENT TYPES TO PRIORITIZE (if found in search results):
        - Annual reports (10-K, Annual Report)
        - Quarterly reports (10-Q)
        - Sustainability/ESG reports
        - Proxy statements (DEF 14A)
        - Current reports (8-K)
        - Investor presentations
        - Climate risk disclosures
        - TCFD reports
        - GRI reports
        - CDP responses

        OUTPUT REQUIREMENTS:
        - Extract ONLY URLs found in search results
        - If fewer than 10 URLs are found, return only what is available
        - DO NOT pad the list with constructed URLs
        - Return ONLY a JSON array, no explanations
        - If no URLs are found, return an empty array []

        OUTPUT FORMAT:
        Return a JSON array with objects containing:
        - company: "{company_info['company_name']}"
        - legal_name: Company's legal name
        - source_type: ["company_site", "regulator_repo", "exchange_repo", "assurance_provider", "industry_assoc", "media_official", "other_secondary"]
        - doc_type: Document type (infer from URL context)
        - title: Document title (infer from URL context)
        - url: EXACT URL from search results
        - filetype: ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "html", "txt"]
        - year: "2022", "2023", or "2024" (infer from URL context)
        - publisher: Document publisher (infer from URL context)
        - language: Document language
        - last_modified: Date if mentioned in search results
        - is_primary_source: true/false
        - relevance_tags: Array of relevant tags
        - notes: Additional notes

        FINAL INSTRUCTION: 
        Carefully examine the search results and extract ONLY the URLs that are explicitly mentioned. Do not construct any URLs. If you find URLs, provide them in the JSON format. If you find no URLs, return an empty array.

        <think>
        [Examine the search results and identify only the URLs that are explicitly mentioned]
        </think>
        [
        // Only include URLs that are explicitly found in the search results
        ]"""
        
        return prompt
    
    def call_sonar_reasoning_api(self, prompt: str) -> Optional[Dict]:
        """Make API call to Perplexity's sonar-deep-research model."""
        if not self.api_key:
            print("ERROR: No API key provided. Set PERPLEXITY_API_KEY environment variable or pass api_key parameter.")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,  # Specify sonar-reasoning model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        try:
            print(f"Making API call to {self.model} model...")
            # Increased timeout for sonar-deep-research model (it needs more time for complex reasoning)
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            print(f"Response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse API response: {e}")
            return None
    
    def process_api_response(self, response: Dict, company_info: Dict[str, Any]) -> List[Dict]:
        """Process the API response and extract URLs."""
        try:
            # Extract content from API response
            content = response['choices'][0]['message']['content']
            
            # Save raw response for debugging
            debug_file = self.results_dir / f"{company_info['isin']}_raw_response.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== RAW API RESPONSE ===\n")
                f.write(f"Response structure: {list(response.keys())}\n")
                f.write(f"Content type: {type(content)}\n")
                f.write(f"Content length: {len(content) if content else 0}\n")
                f.write("\n=== CONTENT ===\n")
                f.write(str(content))
                f.write("\n\n=== END CONTENT ===\n")
            
            print(f"Raw response saved to: {debug_file}")
            print(f"Content preview: {str(content)[:200]}...")
            
            # Try to parse as JSON
            try:
                urls_data = json.loads(content)
                if isinstance(urls_data, list):
                    return urls_data
                else:
                    print("Warning: Response is not a JSON array")
                    print(f"Response type: {type(urls_data)}")
                    return []
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse response as JSON: {e}")
                print("Attempting to extract and fix JSON from response...")
                
                import re
                
                # First try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\[.*?)(?:\s*```|$)', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                    print(f"Found JSON in markdown code block, length: {len(json_content)}")
                    
                    # Try to fix incomplete JSON by finding complete objects
                    try:
                        # First try to parse as-is
                        urls_data = json.loads(json_content)
                        if isinstance(urls_data, list):
                            print("Successfully parsed complete JSON array!")
                            return urls_data
                    except json.JSONDecodeError:
                        print("JSON is incomplete, attempting to extract complete objects...")
                        
                        # Extract complete JSON objects from the array
                        objects = []
                        brace_count = 0
                        current_obj = ""
                        in_string = False
                        escape_next = False
                        
                        for char in json_content:
                            if escape_next:
                                current_obj += char
                                escape_next = False
                                continue
                                
                            if char == '\\':
                                current_obj += char
                                escape_next = True
                                continue
                                
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    
                            current_obj += char
                            
                            # If we've completed an object, try to parse it
                            if brace_count == 0 and current_obj.strip().startswith('{'):
                                try:
                                    obj = json.loads(current_obj.strip().rstrip(','))
                                    objects.append(obj)
                                    print(f"Successfully extracted object: {obj.get('title', 'Unknown')}")
                                except json.JSONDecodeError:
                                    pass  # Skip malformed objects
                                current_obj = ""
                        
                        if objects:
                            print(f"Successfully extracted {len(objects)} complete objects from incomplete JSON")
                            return objects
                        else:
                            print("No complete objects found in JSON")
                
                # Fallback: Try to find JSON array anywhere in the content
                json_match = re.search(r'(\[.*)', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                    print(f"Found JSON array start, length: {len(json_content)}")
                    
                    # Try to fix incomplete JSON by finding complete objects
                    try:
                        # First try to parse as-is
                        urls_data = json.loads(json_content)
                        if isinstance(urls_data, list):
                            print("Successfully parsed complete JSON array!")
                            return urls_data
                    except json.JSONDecodeError:
                        print("JSON is incomplete, attempting to extract complete objects...")
                        
                        # Extract complete JSON objects from the array
                        objects = []
                        brace_count = 0
                        current_obj = ""
                        in_string = False
                        escape_next = False
                        
                        for char in json_content:
                            if escape_next:
                                current_obj += char
                                escape_next = False
                                continue
                                
                            if char == '\\':
                                current_obj += char
                                escape_next = True
                                continue
                                
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    
                            current_obj += char
                            
                            # If we've completed an object, try to parse it
                            if brace_count == 0 and current_obj.strip().startswith('{'):
                                try:
                                    obj = json.loads(current_obj.strip().rstrip(','))
                                    objects.append(obj)
                                    print(f"Successfully extracted object: {obj.get('title', 'Unknown')}")
                                except json.JSONDecodeError:
                                    pass  # Skip malformed objects
                                current_obj = ""
                        
                        if objects:
                            print(f"Successfully extracted {len(objects)} complete objects from incomplete JSON")
                            return objects
                        else:
                            print("No complete objects found in JSON")
                
                # Fallback: Try to extract JSON from <think> tags
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1)
                    print("Found <think> content, searching for JSON within...")
                    
                    # Try to find JSON array in the think content
                    json_match = re.search(r'(\[.*?\])', think_content, re.DOTALL)
                    if json_match:
                        try:
                            urls_data = json.loads(json_match.group(1))
                            if isinstance(urls_data, list):
                                print("Successfully extracted JSON from <think> tags!")
                                return urls_data
                        except json.JSONDecodeError as json_err:
                            print(f"JSON in <think> tags is invalid: {json_err}")
                
                # Try to find JSON after </think> tag
                after_think_match = re.search(r'</think>\s*(\[.*)', content, re.DOTALL)
                if after_think_match:
                    json_content = after_think_match.group(1)
                    try:
                        urls_data = json.loads(json_content)
                        if isinstance(urls_data, list):
                            print("Successfully extracted JSON after </think> tag!")
                            return urls_data
                    except json.JSONDecodeError as json_err:
                        print(f"JSON after </think> tag is invalid: {json_err}")
                
                print("No valid JSON found in response")
                return []
                
        except (KeyError, IndexError) as e:
            print(f"Error processing API response: {e}")
            return []
    
    def save_response(self, company_info: Dict[str, Any], response: Dict, urls_data: List[Dict]) -> str:
        """Save the API response and processed URLs."""
        isin = company_info['isin']
        
        # Save raw API response
        response_file = self.results_dir / f"{isin}_sonar_reasoning_response.json"
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        
        # Save processed URLs as CSV
        if urls_data:
            urls_df = pd.DataFrame(urls_data)
            csv_file = self.results_dir / f"{isin}_sonar_reasoning_data_sources.csv"
            urls_df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"Saved {len(urls_data)} URLs to {csv_file}")
        else:
            print(f"No URLs found for {company_info['company_name']}")
        
        return str(response_file)
    
    def collect_data_for_company(self, company_info: Dict[str, Any]) -> bool:
        """Collect data for a single company using sonar-reasoning API."""
        print(f"\nCollecting data for {company_info['company_name']} ({company_info['isin']})...")
        
        # Generate prompt
        prompt = self.generate_sonar_reasoning_prompt(company_info)
        
        # Make API call
        response = self.call_sonar_reasoning_api(prompt)
        if not response:
            return False
        
        # Process response
        urls_data = self.process_api_response(response, company_info)
        
        # Save results
        self.save_response(company_info, response, urls_data)
        
        return len(urls_data) > 0
    
    def collect_data_for_all_companies(self, delay: float = 2.0) -> Dict[str, bool]:
        """Collect data for all companies using sonar-deep-research API."""
        results = {}
        
        print(f"Starting data collection for {len(self.companies_df)} companies using {self.model} model...")
        print(f"API delay between calls: {delay} seconds")
        
        for i, (_, row) in enumerate(self.companies_df.iterrows()):
            try:
                company_info = self._extract_company_info(row)
                success = self.collect_data_for_company(company_info)
                results[company_info['isin']] = success
                
                # Add delay between API calls to respect rate limits
                if i < len(self.companies_df) - 1:  # Don't delay after last call
                    print(f"Waiting {delay} seconds before next API call...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error processing company {row['NAME']}: {e}")
                results[row['Type']] = False
                continue
        
        return results
    

def main():
    """Main function to run the sonar-reasoning API data collection process."""
    print("Sonar-Reasoning API Climate Risk Data Collector")
    print("=" * 60)
    print("Model: Perplexity sonar-reasoning")
    print("API: https://api.perplexity.ai/chat/completions")
    print("=" * 60)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        print("WARNING: No PERPLEXITY_API_KEY found in .env file or environment variables.")
        print("Please add PERPLEXITY_API_KEY=your_api_key_here to your .env file")
        print("Or set environment variable: export PERPLEXITY_API_KEY='your_api_key_here'")
        return
    
    # Initialize sonar-reasoning API collector
    collector = SonarReasoningAPIClimateRiskDataCollector(api_key=api_key)
    
    print(f"Loaded {len(collector.companies_df)} companies")
    print(f"Historical focus: {collector.historical_years}")
    print(f"API URL: {collector.api_url}")
    print(f"Model: {collector.model}")
    print(f"Results directory: {collector.results_dir}")
    
    # Collect data for all companies
    print("\nStarting API data collection...")
    print("Note: sonar-deep-research model may take 2-3 minutes per company due to complex reasoning")
    print("Timeout set to 120 seconds")
    results = collector.collect_data_for_all_companies(delay=2.0)
    
    # Print results
    successful = sum(1 for success in results.values() if success)
    print(f"\nData collection completed!")
    print(f"Successful: {successful}/{len(results)} companies")
    print(f"Results saved in: {collector.results_dir}")
    
    return results

if __name__ == "__main__":
    main()
