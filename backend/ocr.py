# ocr.py
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import time

# Azure credentials
AZURE_ENDPOINT = "https://willagewarriors.cognitiveservices.azure.com/"
AZURE_KEY = "GAxEu5gMr2O0s7p20T6gN90645esMCAzeRvhkeEKlYxaEF6TUXbmJQQJ99BBACYeBjFXJ3w3AAAFACOGUTJT"

client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

def analyzeImageAndExtractText(image_file):
    """
    Accepts a file-like object (e.g., BytesIO from Streamlit uploader) and extracts:
      - OCR text (via the Read API)
      - An image caption (via the Description feature)
    """
    # --- OCR: Extract Text ---
    # Use the image file directly (no need to open via a file path)
    readOp = client.read_in_stream(image_file, raw=True)
    operationLocation = readOp.headers["Operation-Location"]
    operationId = operationLocation.split("/")[-1]

    # Wait for the OCR processing to complete
    while True:
        result = client.get_read_result(operationId)
        if result.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    extractedText = []
    if result.status == "succeeded":
        for page in result.analyze_result.read_results:
            for line in page.lines:
                extractedText.append(line.text)
    extractedText = "\n".join(extractedText)

    # --- Image Captioning ---
    # Reset file pointer before reusing the file
    image_file.seek(0)
    features = [VisualFeatureTypes.description]
    analysis = client.analyze_image_in_stream(image_file, visual_features=features)
    caption = analysis.description.captions[0].text if analysis.description.captions else "No caption available"

    return {"caption": caption, "extractedText": extractedText}

