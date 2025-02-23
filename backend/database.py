from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
from typing import Tuple, List, Dict, Any, Optional
import re 

def connect_to_mongo() -> Tuple[Optional[MongoClient], Optional[str]]:
    """Connect to MongoDB Atlas cluster"""
    uri = "mongodb+srv://jacobtruong25:dlwn8e4Td8MG1L7K@cluster1.9taaq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
    try:
        client = MongoClient(uri, 
                           server_api=ServerApi('1'), 
                           tlsCAFile=certifi.where())
        client.admin.command('ping')
        return client, None
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_cpt_code(description: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Get relevant CPT/APC codes based on full procedure description
    
    Args:
        description: User's description of medical procedure
        
    Returns:
        Tuple containing:
        - List of matching procedures (empty if none found)
        - Error message if any occurred (None if successful)
    """
    try:
        # Clean and validate input
        clean_description = description.strip()
        if not clean_description:
            return [], "Please provide a procedure description"
            
        # Split and escape search terms
        terms = [re.escape(term) for term in clean_description.split() if term]
        if not terms:
            return [], "No valid search terms found"
            
        # Connect to MongoDB
        client, error = connect_to_mongo()
        if error:
            return [], error
            
        try:
            db = client.Hack2025
            matches = []
            
            # Build search query for CPT codes with correct field names
            cpt_query = {
                "$and": [
                    {"HCPCS Short Descriptor": {"$exists": True}},
                    {"CPT/HCPCS Code": {"$exists": True}},  # Updated field name
                    {"$or": [
                        {"HCPCS Short Descriptor": {"$regex": term, "$options": "i"}}
                        for term in terms
                    ]}
                ]
            }
            
            # Execute CPT query with correct field projection
            cpt_results = db.cptcosts.find(cpt_query, {
                'CPT/HCPCS Code': 1,  # Updated field name
                'HCPCS Short Descriptor': 1,
                '_id': 0
            })
            
            for doc in cpt_results:
                if 'CPT/HCPCS Code' in doc and 'HCPCS Short Descriptor' in doc:  # Updated field name check
                    # Create a standardized format for the matches
                    matches.append({
                        'HCPCS Code': doc['CPT/HCPCS Code'],  # Map to consistent key name
                        'HCPCS Short Descriptor': doc['HCPCS Short Descriptor']
                    })
            
            # Build search query for APC codes
            apc_query = {
                "$and": [
                    {"APC Title": {"$exists": True}},
                    {"APC": {"$exists": True}},
                    {"$or": [
                        {"APC Title": {"$regex": term, "$options": "i"}}
                        for term in terms
                    ]}
                ]
            }
            
            # Execute APC query
            apc_results = db.apccosts.find(apc_query, {
                'APC': 1,
                'APC Title': 1,
                '_id': 0
            })
            
            matches.extend(apc_results)
            
            # Debug logging
            if not matches:
                print("No matches found")
            else:
                print(f"Found {len(matches)} matches")
                for m in matches:
                    print(f"Match fields: {m.keys()}")
            
            return matches, None
            
        finally:
            client.close()
            
    except Exception as e:
        print(f"Error in get_cpt_code: {str(e)}")  # Added error logging
        return [], f"Search error: {str(e)}"