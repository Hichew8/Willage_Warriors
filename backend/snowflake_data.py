import snowflake.connector
import pandas as pd
import streamlit as st

# Secure credentials using Streamlit Secrets
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        
    )

# Query Snowflake for median rates by state
@st.cache_data
def fetch_median_rates(state, insurance_provider, billing_code):
    if insurance_provider == 'N/A':
        # Return an empty DataFrame if insurance provider is 'N/A'
        return pd.DataFrame(columns=[
            "PAYER",
            "BILLING_CODE_TYPE",
            "BILLING_CODE",
            "NEGOTIATED_TYPE",
            "NEGOTIATED_RATE",
            "NPPES_ORGFRIENDLYNAME",
            "NPPES_CITY",
            "NPPES_STATE",
            "PAYERSET_BILLING_CODE_NAME",
            "NUCC_TAXONOMY_CLASSIFICATION",
            "PAYERSET_BILLING_CODE_CATEGORY",
            "PAYERSET_BILLING_CODE_SUBCATEGORY"
        ])
    
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Use the appropriate database and schema
        cursor.execute("USE DATABASE _PAYERSET_PRICE_TRANSPARENCY;")
        cursor.execute("USE SCHEMA CUSTOMER_SHARE;")
        
        query = """
            SELECT 
                r.PAYER,
                r.BILLING_CODE_TYPE,
                r.BILLING_CODE,
                r.NEGOTIATED_TYPE,
                r.NEGOTIATED_RATE,
                r.NPPES_ORGFRIENDLYNAME,
                r.NPPES_CITY,
                r.NPPES_STATE,
                r.PAYERSET_BILLING_CODE_NAME,
                r.PAYERSET_BILLING_CODE_CATEGORY,
                r.NUCC_TAXONOMY_CLASSIFICATION,
                r.PAYERSET_BILLING_CODE_SUBCATEGORY
            FROM CUSTOMER_SHARE.PAYERSET_PRICE_TRANSPARENCY r
            WHERE  
                r.NPPES_STATE = %s
                AND r.BILLING_CODE = %s
                AND r.PAYER = %s
            LIMIT 1000
        """

        print("Executing query:", query)
        print("With parameters:", (state, billing_code, insurance_provider))

        cursor.execute(query, (state, billing_code, insurance_provider))
        
        # Fetch results
        data = cursor.fetchall()
        print("Query results:", data)

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            "PAYER",
            "BILLING_CODE_TYPE",
            "BILLING_CODE",
            "NEGOTIATED_TYPE",
            "NEGOTIATED_RATE",
            "NPPES_ORGFRIENDLYNAME",
            "NPPES_CITY",
            "NPPES_STATE",
            "PAYERSET_BILLING_CODE_NAME",
            "PAYERSET_BILLING_CODE_CATEGORY",
            "NUCC_TAXONOMY_CLASSIFICATION",
            "PAYERSET_BILLING_CODE_SUBCATEGORY"
        ])

        return df

    except snowflake.connector.errors.ProgrammingError as e:
        print("Snowflake SQL Error:", e)
        return pd.DataFrame()  # Return an empty DataFrame on error

    finally:
        cursor.close()
        conn.close()


@st.cache_data
def fetch_unique_payers():
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    cursor.execute("USE DATABASE _PAYERSET_PRICE_TRANSPARENCY;")
    cursor.execute("USE SCHEMA CUSTOMER_SHARE;")
    
    query = """
        SELECT DISTINCT PAYER
        FROM CUSTOMER_SHARE.PAYERSET_PRICE_TRANSPARENCY
        WHERE PAYER IS NOT NULL
        ORDER BY PAYER
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Extract the string value from each tuple result
    return [row[0] for row in data]  # Changed from {row[0]} to row[0]

@st.cache_data
def fetch_unique_billing_codes():
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    cursor.execute("USE DATABASE _PAYERSET_PRICE_TRANSPARENCY;")
    cursor.execute("USE SCHEMA CUSTOMER_SHARE;")
    
    query = """
        SELECT DISTINCT BILLING_CODE
        FROM CUSTOMER_SHARE.PAYERSET_PRICE_TRANSPARENCY
        WHERE BILLING_CODE IS NOT NULL
        ORDER BY BILLING_CODE
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Extract the string value from each tuple result
    return [row[0] for row in data]