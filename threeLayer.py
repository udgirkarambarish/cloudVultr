import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
import fitz  # PyMuPDF for PDF metadata
from datetime import datetime, timedelta
import easyocr
import google.generativeai as genai
import cv2
from pyzbar.pyzbar import decode
import numpy as np

# Initialize the Gemini API with the hardcoded API key
genai.configure(api_key='AIzaSyAIlrYllmYs9Lhjt_CaLa4-tTVJ-7CcyNA')


def layerOne(path, score):
    # Preprocess images by resizing them to 224x224 and normalizing them
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None  # Skip files that aren't valid images
        image = cv2.resize(image, (224, 224))  # Resize to 224x224 for consistency
        image = image.astype('float32') / 255  # Normalize to [0, 1]
        return image

    # Load the trained model
    model = load_model("fake_invoice_detection_model.h5")

    # Function to predict if an image is fake or authentic
    def predict_fake_or_authentic(image_path, curr_score):
        image = preprocess_image(image_path)
        if image is None:
            return "Image could not be processed.", score
        
        # Expand image dimensions to match the model's input shape
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        
        # If prediction > 0.5, classify as 'Fake'; otherwise, 'Authentic'
        if prediction[0][0] > 0.5:
            return "Fake", curr_score
        else:
            curr_score += 35
            return "Authentic", curr_score

    # Example usage: Test a new image
    curr_score = 0
    image_path = path
    result, curr_score = predict_fake_or_authentic(image_path, curr_score)

    print(f"The image is: {result}")
    print(f"Current score: {curr_score}")

    return score + curr_score

def layerTwo(path, score):
    # Function to detect file type
    def detect_file_type(file_path):
        if file_path.lower().endswith('.pdf'):
            return 'pdf'
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            return 'image'
        return None

    # Function to extract image metadata
    def get_image_metadata(image_path, curr_score):
        """Extract basic metadata including file creation and modification dates."""
        metadata = {}
        try:
            # Open the image and try to extract EXIF data
            image = Image.open(image_path)
            exif_data = image._getexif()

            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    metadata[tag_name] = value

            # Add file creation and modification dates from the file system
            file_stats = os.stat(image_path)
            creation_date = datetime.fromtimestamp(file_stats.st_ctime)
            modification_date = datetime.fromtimestamp(file_stats.st_mtime)
            metadata["FileCreationDate"] = creation_date.strftime("%Y-%m-%d %H:%M:%S")
            metadata["FileModificationDate"] = modification_date.strftime("%Y-%m-%d %H:%M:%S")

            # Authenticity Check based on creation and modification dates
            time_diff = modification_date - creation_date
            threshold = timedelta(hours=2)

            if time_diff > threshold:
                metadata["SuspiciousModification"] = f"Image was modified {time_diff} after creation. This might be suspicious."
            else:
                metadata["SuspiciousModification"] = "Image seems to be consistent with creation and modification dates."
                curr_score += 35

        except Exception as e:
            metadata["Error"] = f"Error fetching metadata: {e}"

        return metadata, curr_score

    # Function to extract PDF metadata using PyMuPDF
    def check_pdf_metadata(file_path, curr_score):
        suspicious_metadata = []
        mod_date = None
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata

            # Display PDF metadata
            print("PDF Metadata:")
            for key, value in metadata.items():
                print(f"{key}: {value}")

            # Check for recent edits based on the 'modDate' field
            mod_date_str = metadata.get("modDate")
            if mod_date_str:
                try:
                    if mod_date_str.startswith("D:"):
                        mod_date_str = mod_date_str[2:]
                    mod_date_str = mod_date_str.split('+')[0]
                    mod_date = datetime.strptime(mod_date_str, '%Y%m%d%H%M%S')

                    if datetime.now() - mod_date < timedelta(days=30):
                        suspicious_metadata.append("Document was modified recently.")
                        print("Warning: Document was modified within the last 30 days.")
                    else:
                        curr_score += 35
                except Exception as e:
                    print(f"Error parsing modification date: {e}")

            author = metadata.get("author", "")
            producer = metadata.get("producer", "")
            if "unknown" in author.lower() or "unknown" in producer.lower():
                suspicious_metadata.append("Author or producer information is missing or unknown.")
                print("Warning: Author or producer metadata appears suspicious.")
        
        except Exception as e:
            print(f"Error processing PDF metadata: {e}")
        
        return suspicious_metadata, curr_score

    # Main function to process the file based on its type and return updated curr_score
    def process_file(file_path, curr_score):
        file_type = detect_file_type(file_path)
        if file_type == 'pdf':
            print("Processing PDF file...")
            suspicious_metadata, curr_score = check_pdf_metadata(file_path, curr_score)
            if suspicious_metadata:
                print("\nSuspicious Metadata Detected in PDF:")
                for warning in suspicious_metadata:
                    print(warning)
            else:
                print("No suspicious metadata detected in PDF.")
        elif file_type == 'image':
            print("Processing image file...")
            metadata, curr_score = get_image_metadata(file_path, curr_score)
            if "Error" in metadata:
                print(f"Error processing image metadata: {metadata['Error']}")
            else:
                print("\nImage Metadata:")
                for key, value in metadata.items():
                    print(f"{key}: {value}")
        else:
            print("Unsupported file format or unable to detect file type.")

        return curr_score

    # Initialize curr_score and process the file to get the updated score
    curr_score = process_file(path, 0)
    return score + curr_score

def layerThree(path, score):
    # Initialize the EasyOCR Reader for English
    reader = easyocr.Reader(['en'])

    # Path to the image file you want to test
    image_path = path # Replace with your image path

    # Initialize the score variable
    curr_score = 0

    # Function to enhance the image for OCR and QR code extraction
    def enhance_image(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Apply Gaussian blur to reduce noise
        enhanced = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
        return enhanced

    # Enhance the image
    enhanced_image = enhance_image(image_path)

    # Save the enhanced image temporarily for EasyOCR to read
    temp_path = "enhanced_temp.png"
    cv2.imwrite(temp_path, enhanced_image)

    # Step 1: Extract text from the enhanced image using OCR
    extracted_text = reader.readtext(temp_path, detail=0)
    extracted_text_str = " ".join(extracted_text)

    # Step 2: Detect and extract QR code data from the enhanced image
    def extract_qr_data(enhanced_image):
        # Convert enhanced image to a format compatible with pyzbar
        qr_codes = decode(enhanced_image)
        if qr_codes:
            return qr_codes[0].data.decode("utf-8")
        return None

    qr_data = extract_qr_data(enhanced_image)

    # Clean up the temporary file
    os.remove(temp_path)

    # Step 3: Send both OCR and QR extracted data to AI model for authentication
    def authenticate_with_ai(ocr_text, qr_text):
        prompt = (
            f"Authenticate the following extracted data from a bill or invoice. "
            f"Only compare the actual values for 'Date' and 'Amount' fields between OCR and QR data, and ignore any differences in formatting or labels. "
            f"For instance, ignore extra words like 'Invoice Date' or 'QR Code Date', and focus solely on the date itself. "
            f"Also, ignore minor differences in decimal places for amounts (727.20 vs. 727.2 should be considered a match). "
            f"Flag the document as 'fake' only if there is a clear and exact difference in these specific values.\n\n"
            f"OCR Extracted Data: {ocr_text}\n"
            f"QR Code Extracted Data: {qr_text}\n\n"
            f"Provide a concise result and note any mismatched fields only if they exist.\n"
            f"Result:"
        )

        # Use Gemini model to process the prompt
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, 'text') else "Unknown result"

    # Run the AI-based authentication
    if qr_data:  # If QR data is available, proceed with AI-based comparison
        ai_result = authenticate_with_ai(extracted_text_str, qr_data)
        print("Gemini AI Authentication Result:", ai_result)

        # Update curr_score based on the AI authentication result
        if "authentic" in ai_result.lower():
            curr_score += 30
            print("Image authenticated. Current Score:", curr_score)
            return score + curr_score
        else:
            print("Image flagged as fake. Current Score:", curr_score)
            return score
    else:
        print("No QR Code found. Unable to authenticate using AI.")
        print("Current Score:", curr_score)
        return score
