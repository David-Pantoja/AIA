# Logic for parsing filing documents (HTML, XBRL) 

import json
from typing import Dict, List, Optional
import openai
from dotenv import load_dotenv  
import os
load_dotenv()

# Replace with your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_filing_document(document: str, form_type: str) -> Dict:
    """Parse a filing document using the OpenAI document API to extract key financial data."""
    try:
        if not document:
            print("Error: Empty document provided")
            return {}

        # Check if OpenAI API key is set
        if not openai.api_key:
            print("Error: OpenAI API key not found in environment variables")
            return {}

        # Truncate document to first 4000 characters to stay within token limits
        truncated_doc = document[:4000] + "..."
        
        # Prepare the prompt based on filing type
        base_prompt = f"""Extract the following key financial data from the following SEC {form_type} filing document.
Please provide all monetary values in billions (B) with 2 decimal places.
For example: 123.45B for 123.45 billion.

Document:
{truncated_doc}

Please provide the following information in JSON format:"""

        # Customize prompt based on filing type
        if form_type in ["10-K", "10-Q"]:
            prompt = base_prompt + """
- Revenue (in billions)
- Net Income (in billions)
- EPS (earnings per share)
- Cash (in billions)
- Assets (in billions)
- Liabilities (in billions)
- Management Discussion & Analysis summary (2-3 sentences)
- Risk Factors summary (2-3 sentences)"""
        elif form_type == "8-K":
            prompt = base_prompt + """
- Event Type
- Event Date
- Event Description (2-3 sentences)
- Financial Impact (if any)
- Key Takeaways (2-3 sentences)"""
        else:
            prompt = base_prompt + """
- Key Information (2-3 sentences)
- Financial Impact (if any)
- Important Details (2-3 sentences)"""

        print("Sending request to OpenAI API...")
        try:
            # Call the OpenAI API using the new format
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial data extraction assistant. Always respond with valid JSON. Format all monetary values in billions (B) with 2 decimal places."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            print("Received response from OpenAI API")
            
            # Check if response is valid
            if not response or not response.choices:
                print("Error: Empty response from OpenAI API")
                return {}

            content = response.choices[0].message.content.strip()
            print(f"Response content preview: {content[:200]}")

            # Try to parse the response as JSON
            try:
                parsed_data = json.loads(content)
                print("Successfully parsed JSON response")
                return parsed_data
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw response: {content}")
                return {}

        except openai.BadRequestError as e:
            print(f"OpenAI API error: {e}")
            return {}
        except openai.AuthenticationError as e:
            print(f"OpenAI API authentication error: {e}")
            return {}
        except openai.RateLimitError as e:
            print(f"OpenAI API rate limit error: {e}")
            return {}
        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            return {}

    except Exception as e:
        print(f"Unexpected error in parse_filing_document: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

def parse_filings(filings: List[Dict]) -> List[Dict]:
    """Parse a list of filing documents and return structured data."""
    parsed_filings = []
    for filing in filings:
        parsed_data = parse_filing_document(filing["document"], filing.get("form_type", "Unknown"))
        parsed_filings.append({
            "accession_number": filing["accession_number"],
            "filing_date": filing["filing_date"],
            "form_type": filing.get("form_type", "Unknown"),
            "financials": parsed_data
        })
    return parsed_filings 