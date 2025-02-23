import requests
import json
import pandas as pd
import streamlit as st
from snowflake_data import fetch_median_rates, fetch_unique_payers, fetch_unique_billing_codes
from database import get_cpt_code
import re as re

def init_llama():
    """
    Initialize and return a simple LLAMA model interface.
    This is a lightweight wrapper around the LLAMA API functionality.
    """
    class LlamaInterface:
        def __init__(self):
            self.api_url = 'http://localhost:8080/completion'
        
        def extract_keywords(self, text):
            """
            Extract relevant medical keywords from text using LLAMA.
            """
            prompt = (
                "Extract the most relevant medical procedure keywords from this text. "
                "Return only the key medical terms, separated by commas if there are multiple:\n\n"
                f"{text}"
            )
            
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        'prompt': prompt,
                        'n_predict': 100,
                        'temperature': 0.2,  # Lower temperature for more focused responses
                        'top_p': 0.9,
                        'repeat_penalty': 1.2,
                        'stream': False  # Don't stream for keyword extraction
                    }
                )
                response.raise_for_status()
                
                # Extract keywords from response
                keywords = response.json().get('content', '').strip()
                # Clean up the keywords
                keywords = re.sub(r'[^\w\s,]', '', keywords)
                keywords = [k.strip() for k in keywords.split(',')]
                # Return the first (most relevant) keyword
                return keywords[0] if keywords else None
                
            except Exception as e:
                st.error(f"Error extracting keywords: {str(e)}")
                return None

    return LlamaInterface()

def get_intro_prompt():
    """Returns the introductory message for the chatbot."""
    return (
        "Welcome to Caduceo, a Healthcare Cost Estimator Chatbot! 🏥\n\n"
        "I'm here to help you find the best hospitals for specific medical procedures "
        "based on your location, insurance, and other preferences. Let's get started!\n\n"
        "To begin, could you please tell me: \n"
        "What state are you located in?\n"
    )

def format_prompt(conversation_history):
    """Formats the conversation history with DataFrame handling."""
    prompt = "<|system|>\nYou're a healthcare AI assistant with access to data.\n"
    
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "assistant"
        
        # Handle DataFrame content
        if isinstance(msg["content"], pd.DataFrame):
            content = msg["content"].to_markdown(index=False)
        else:
            content = str(msg["content"])
            
        prompt += f"<|{role}|>\n{content}\n"
    
    prompt += "<|assistant|>\n"
    return prompt

def send_llama_request(prompt_config):
    """Sends a request to the LLaMA API with proper error handling."""
    try:
        response = requests.post(
            'http://localhost:8080/completion',
            json={
                'prompt': prompt_config,  # Directly use formatted prompt
                'n_predict': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
                'repeat_penalty': 1.2,
                'stream': True
            },
            stream=True
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error: {str(e)}")
def normalize_insurance_provider(provider_name):
    """
    Normalize insurance provider names to a standardized format.
    Example: "Cigna Health" -> "CIGNA", "Aetna Insurance" -> "AETNA".
    """
    provider_name = provider_name.lower()
    if "cigna" in provider_name:
        return "CIGNA"
    elif "aetna" in provider_name:
        return "AETNA"
    elif "blue cross" in provider_name or "bluecross" in provider_name:
        return "BCBS_SOUTH_CAROLINA"
    elif "unitedhealth" in provider_name or "united health" in provider_name or "united" in provider_name:
        return "UNITED_HEALTHCARE"
    # Add more mappings as needed
    return provider_name.upper()  # Default to uppercase if no specific mapping
def analyze_rate_fairness(quoted_rate, rate_data):
    """
    Analyzes whether a quoted rate is potentially fraudulent by comparing it to market data.
    
    Args:
        quoted_rate (float): The rate quoted to the user
        rate_data (pd.DataFrame): DataFrame containing negotiated rates
    
    Returns:
        dict: Analysis results including determination and supporting statistics
    """
    try:
        # Convert rates to numeric and clean data
        rate_data["NEGOTIATED_RATE"] = pd.to_numeric(rate_data["NEGOTIATED_RATE"], errors="coerce")
        rate_data = rate_data.dropna(subset=["NEGOTIATED_RATE"])
        
        # Calculate statistics
        avg_rate = rate_data["NEGOTIATED_RATE"].mean()
        min_rate = rate_data["NEGOTIATED_RATE"].min()
        max_rate = rate_data["NEGOTIATED_RATE"].max()
        median_rate = rate_data["NEGOTIATED_RATE"].median()
        std_dev = rate_data["NEGOTIATED_RATE"].std()
        
        # Calculate percentiles
        percentile_75 = rate_data["NEGOTIATED_RATE"].quantile(0.75)
        percentile_90 = rate_data["NEGOTIATED_RATE"].quantile(0.90)
        
        # Calculate how many standard deviations above mean
        z_score = (quoted_rate - avg_rate) / std_dev if std_dev > 0 else 0
        
        # Calculate percentage above average
        percent_above_avg = ((quoted_rate - avg_rate) / avg_rate) * 100
        
        # Determine if rate is potentially fraudulent
        is_fraudulent = False
        reasons = []
        
        if quoted_rate > percentile_90:
            is_fraudulent = True
            reasons.append(f"Rate is in the top 10% of all rates (above ${percentile_90:,.2f})")
            
        if z_score > 2:
            is_fraudulent = True
            reasons.append(f"Rate is {z_score:.1f} standard deviations above the mean")
            
        if percent_above_avg > 100:  # More than double the average
            is_fraudulent = True
            reasons.append(f"Rate is {percent_above_avg:.1f}% above the average")
            
        # Prepare analysis message
        analysis = {
            "is_fraudulent": is_fraudulent,
            "statistics": {
                "quoted_rate": quoted_rate,
                "average_rate": avg_rate,
                "median_rate": median_rate,
                "min_rate": min_rate,
                "max_rate": max_rate,
                "percent_above_average": percent_above_avg,
                "standard_deviations_above": z_score
            },
            "message": f"""Based on our analysis of {len(rate_data)} rates:
- Your quoted rate: ${quoted_rate:,.2f}
- Market average: ${avg_rate:,.2f}
- Median rate: ${median_rate:,.2f}
- Lowest rate: ${min_rate:,.2f}
- Highest rate: ${max_rate:,.2f}

Your rate is {percent_above_avg:.1f}% {'above' if percent_above_avg > 0 else 'below'} the market average.

{'⚠️ This rate appears potentially fraudulent for the following reasons:' if is_fraudulent else '✓ This rate appears to be within reasonable market ranges.'}
{chr(10).join('- ' + reason for reason in reasons) if reasons else ''}"""
        }
        
        return analysis
        
    except Exception as e:
        return {
            "is_fraudulent": None,
            "statistics": {},
            "message": f"Error analyzing rate: {str(e)}"
        }
def handle_user_input():
    """Handle user input in the chat interface with CPT code lookup integration"""
    STATE_ABBREVIATIONS = {
        "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA", "COLORADO": "CO", 
        "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", 
        "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", 
        "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", 
        "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", 
        "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", 
        "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", 
        "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", 
        "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"
    }
    VALID_STATES = set(STATE_ABBREVIATIONS.values())
    # Define the steps and corresponding placeholders
    steps = {
        "state": "Enter State:",
        "cpt_code": "Enter CPT Code or Describe Procedure:",
        "insurance": "Enter Insurance Provider (or type 'N/A'):",
        "rate": "Enter Quoted Rate (e.g., $500.00):",
        "continue": "Chat:"
    }

    # Determine the current step based on session state
    if st.session_state.state is None:
        current_step = "state"
    elif st.session_state.cpt_code is None:
        current_step = "cpt_code"
    elif st.session_state.quoted_rate is None:
        current_step = "rate"
    elif st.session_state.insurance is None:
        current_step = "insurance"
    
    else:
        current_step = "continue"
    user_input = st.chat_input(steps[current_step])
    if user_input:
        try:
            st.session_state.conversation.append({"role": "user", "content": user_input})

            # Step 1: Request state
            if current_step == "state":
                state_candidate = user_input.strip().upper()
                if state_candidate in STATE_ABBREVIATIONS:
                    state_candidate = STATE_ABBREVIATIONS[state_candidate]
                if state_candidate in VALID_STATES:
                    st.session_state.state = state_candidate
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": (
                            f"Great! You selected {st.session_state.state}. "
                            "Now, please describe the medical procedure you're interested in, "
                            "or if you know the CPT code, you can enter it directly."
                        )
                    })
                else:
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": "Invalid state. Please enter a valid U.S. state abbreviation or full state name."
                    })
                st.rerun()

            # Step 2: Request/lookup CPT code
            elif current_step == "cpt_code":
                # Fetch the list of valid billing codes from the database
                valid_codes = fetch_unique_billing_codes()
                
                # Check if the user input matches any of the valid billing codes
                if user_input.strip().upper() in valid_codes:
                    st.session_state.cpt_code = user_input.strip().upper()

                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": "Got it! Now, please enter the quoted rate you received in the format XXX.XX (for example: 500.00)"
                    })
                else:
                    # Try to find matching CPT codes based on description
                    with st.spinner("Looking up procedure codes..."):
                        matches, error = get_cpt_code(user_input)

                        if error:
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": f"Error looking up procedure: {error}"
                            })
                        elif matches:
                            match_text = "I found these matching procedures:\n\n"
                            shown_matches = 0
                            
                            for i, match in enumerate(matches, 1):
                                if all(key in match for key in ['HCPCS Code', 'HCPCS Short Descriptor']):
                                    match_text += f"{i}. CPT {match['HCPCS Code']}: {match['HCPCS Short Descriptor']}\n"
                                    shown_matches += 1
                                elif all(key in match for key in ['APC', 'APC Title']):
                                    match_text += f"{i}. APC {match['APC']}: {match['APC Title']}\n"
                                    shown_matches += 1
                                
                                if shown_matches >= 20:
                                    break

                            if shown_matches > 0:
                                match_text += "\nPlease select a procedure by entering its code (CPT or APC), or describe the procedure more specifically."
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": match_text
                                })
                            else:
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": "No valid matches found. Please try describing the procedure differently."
                                })
                    st.rerun()
            # After your CPT code handling section but before insurance handling
            elif current_step == "rate":
                # Remove any non-numeric characters except decimal point
                cleaned_input = ''.join(c for c in user_input if c.isdigit() or c == '.')
                
                try:
                    # Try to convert the cleaned input to a float
                    if cleaned_input.startswith('$'):
                        cleaned_input = cleaned_input[1:]
                    quoted_rate = float(cleaned_input)
                    
                    # Validate the rate is positive
                    if quoted_rate <= 0:
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": "Please enter a valid positive rate in the format XXX.XX"
                        })
                    else:
                        st.session_state.quoted_rate = quoted_rate
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": (
                                f"Thank you for providing your quoted rate of ${quoted_rate:.2f}. "
                                "Now, please provide your insurance provider name or type 'N/A' if not applicable."
                            )
                        })
                except ValueError:
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": "Please enter a valid rate in the format $XXX.XX (for example: $500.00)"
                    })
                st.rerun()
            # Step 3: Request insurance
            elif current_step == "insurance":
                entered_insurance = user_input.strip().lower()  # Normalize input to lowercase
                
                # Handle "N/A" case
                if entered_insurance in ["n/a", "na"]:
                    st.session_state.insurance = "N/A"
                    with st.spinner("Fetching data..."):
                        rate = fetch_median_rates(
                            st.session_state.state, st.session_state.insurance, st.session_state.cpt_code
                        )
                        if not rate.empty:
                            if "NEGOTIATED_RATE" not in rate.columns or "NPPES_ORGFRIENDLYNAME" not in rate.columns:
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": (
                                        "Error: Required columns not found in data. "
                                        "Available columns: " + ", ".join(rate.columns.tolist())
                                    )
                                })
                            else:
                                rate["NEGOTIATED_RATE"] = pd.to_numeric(rate["NEGOTIATED_RATE"], errors="coerce")
                                avg_rate = rate["NEGOTIATED_RATE"].mean()
                                min_rate = rate["NEGOTIATED_RATE"].min()
                                min_rate_row = rate.loc[rate["NEGOTIATED_RATE"].idxmin()]
                                practice_name = min_rate_row.get("NPPES_ORGFRIENDLYNAME", "N/A")
                                practice_city = min_rate_row.get("NPPES_CITY", "N/A")
                                practice_state = min_rate_row.get("NPPES_STATE", "N/A")
                                practice_type = min_rate_row.get("NUCC_TAXONOMY_CLASSIFICATION", "N/A")
                                payer_billing_code = min_rate_row.get("PAYERSET_BILLING_CODE_NAME", "N/A")
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": "Here is the data you requested:"
                                })
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": (
                                        f"The Average rate is ${avg_rate:,.2f}\n\n"
                                        f"The Lowest rate found is ${min_rate:,.2f}\n\n"
                                        f"The Lowest rate found was at {practice_name} ({practice_type}), which is at {practice_city}, {practice_state}"
                                    )
                                })

                                # Perform fraud detection analysis if quoted rate is available
                                if st.session_state.quoted_rate is not None:
                                    with st.spinner("Analyzing rate fairness..."):
                                        analysis = analyze_rate_fairness(st.session_state.quoted_rate, rate)
                                        st.session_state.conversation.append({
                                            "role": "assistant",
                                            "content": analysis["message"]
                                        })
                        else:
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": "No data available for the given query."
                            })
                    st.session_state.state = None
                    st.session_state.insurance = None
                    st.session_state.cpt_code = None
                    st.rerun()

                # Normalize insurance provider names using the existing normalize_insurance_provider function
                normalized_insurance = normalize_insurance_provider(entered_insurance)

                # Fetch valid insurance providers
                valid_insurance_providers = fetch_unique_payers()

                # Check if the normalized insurance provider is valid
                if normalized_insurance in [provider.upper() for provider in valid_insurance_providers]:
                    st.session_state.insurance = normalized_insurance
                    with st.spinner("Fetching data..."):
                        rate = fetch_median_rates(
                            st.session_state.state, st.session_state.insurance, st.session_state.cpt_code
                        )
                        if not rate.empty:
                            if "NEGOTIATED_RATE" not in rate.columns or "NPPES_ORGFRIENDLYNAME" not in rate.columns:
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": (
                                        "Error: Required columns not found in data. "
                                        "Available columns: " + ", ".join(rate.columns.tolist())
                                    )
                                })
                            else:
                                rate["NEGOTIATED_RATE"] = pd.to_numeric(rate["NEGOTIATED_RATE"], errors="coerce")
                                avg_rate = rate["NEGOTIATED_RATE"].mean()
                                min_rate = rate["NEGOTIATED_RATE"].min()
                                min_rate_row = rate.loc[rate["NEGOTIATED_RATE"].idxmin()]
                                practice_name = min_rate_row.get("NPPES_ORGFRIENDLYNAME", "N/A")
                                practice_city = min_rate_row.get("NPPES_CITY", "N/A")
                                practice_state = min_rate_row.get("NPPES_STATE", "N/A")
                                practice_type = min_rate_row.get("NUCC_TAXONOMY_CLASSIFICATION", "N/A")
                                payer_billing_code = min_rate_row.get("PAYERSET_BILLING_CODE_NAME", "N/A")
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": "Here is the data you requested:"
                                })
                                st.session_state.conversation.append({
                                    "role": "assistant",
                                    "content": (
                                        f"The Average rate is ${avg_rate:,.2f}\n\n"
                                        f"The Lowest rate found is ${min_rate:,.2f}\n\n"
                                        f"The Lowest rate found was at {practice_name} ({practice_type}), which is at {practice_city}, {practice_state}"
                                    )
                                })

                                # Perform fraud detection analysis if quoted rate is available
                                if st.session_state.quoted_rate is not None:
                                    with st.spinner("Analyzing rate fairness..."):
                                        analysis = analyze_rate_fairness(st.session_state.quoted_rate, rate)
                                        st.session_state.conversation.append({
                                            "role": "assistant",
                                            "content": analysis["message"]
                                        })

                                # Append the explanation request to the conversation history (but don't display it)
                                explanation_prompt = f"Explain what : {payer_billing_code} is in detail and give me details of {practice_name} such as address and contact number and the rating."
                                formatted_prompt = format_prompt(st.session_state.conversation + [{"role": "user", "content": explanation_prompt}])

                                # Send the request to LLaMA
                                response = send_llama_request(formatted_prompt)

                                # Collect response from streaming API
                                llama_response = ""
                                container = st.empty()  # Create an empty container to stream the response

                                # Read streaming response line by line
                                for chunk in response.iter_lines():
                                    if chunk:
                                        try:
                                            # Remove "data: " prefix if present
                                            chunk_str = chunk.decode("utf-8").strip()
                                            if chunk_str.startswith("data: "):
                                                chunk_str = chunk_str[6:].strip()

                                            # Parse JSON and extract content
                                            json_data = json.loads(chunk_str)
                                            if "content" in json_data:
                                                llama_response += json_data["content"]
                                                # Update the container with the streaming response
                                                container.markdown(f"**Assistant:**\n{llama_response}")
                                        except json.JSONDecodeError:
                                            continue  # Skip malformed chunks

                                # Append LLaMA's final response to the conversation history
                                st.session_state.conversation.append({"role": "assistant", "content": llama_response})
                        else:
                            st.session_state.conversation.append({
                                "role": "assistant",
                                "content": "No data available for the given query."
                            })
                    st.session_state.chat_mode = "continue"        

                else:
                    # Handle unrecognized insurance providers
                    matching_providers = [
                        provider for provider in valid_insurance_providers 
                        if entered_insurance in provider.lower()
                    ]
                    if matching_providers:
                        formatted_list = "\n".join(f"- {provider}" for provider in matching_providers)
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": (
                                f"I found multiple insurance providers matching '{user_input}':\n\n"
                                f"{formatted_list}\n\n"
                                "Please enter the exact name of your insurance provider or type 'N/A' if not applicable."
                            )
                        })
                    else:
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": "Insurance provider not recognized. Please enter a valid insurance provider name or type 'N/A' if not applicable."
                        })
                    st.rerun()            
            elif current_step == "continue":
                # Format conversation history into a prompt for LLaMA
                formatted_prompt = format_prompt(st.session_state.conversation)
                try:
                    # Send the request to LLaMA
                    response = send_llama_request(formatted_prompt)
                    
                    # Create a placeholder for streaming text
                    message_placeholder = st.empty()
                    # Initialize full response storage
                    llama_response = ""
                    
                    # Read streaming response line by line
                    for chunk in response.iter_lines():
                        if chunk:
                            try:
                                # Remove "data: " prefix if present
                                chunk_str = chunk.decode("utf-8").strip()
                                if chunk_str.startswith("data: "):
                                    chunk_str = chunk_str[6:].strip()
                                
                                # Parse JSON and extract content
                                json_data = json.loads(chunk_str)
                                if "content" in json_data:
                                    # Append new content to full response
                                    llama_response += json_data["content"]
                                    # Update the placeholder with full response so far
                                    message_placeholder.markdown(llama_response)
                                    
                            except json.JSONDecodeError:
                                continue  # Skip malformed chunks
                    
                    # After streaming is complete, append the full response to conversation
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": llama_response
                    })
                    
                except Exception as e:
                    st.error(f"Error in continued conversation: {str(e)}")
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error. Please try again."
                    })
            st.rerun()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Chat error: {str(e)}\n\n{error_details}")
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "Sorry, I encountered an error. Please try again."
            })
            st.rerun()