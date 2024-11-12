from flask import Flask
from threeLayer import *

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host='45.32.19.96', port=5000)
import boto3
from botocore.client import Config

# Set up custom S3 client for VultroObjects 
s3_client = boto3.client(
    's3',
    aws_access_key_id='9KJQ2JY4OWRTDKCW52LQ',  # Your Access Key
    aws_secret_access_key='hjpHbaR4ZsFIaumrJXcFSbOHwcuKhlbxFezoIpF8',  # Your Secret Key
    endpoint_url='https://sgp1.vultrobjects.com',  # Your custom endpoint
    config=Config(signature_version='s3v4')  # Specify the signature version
)
from flask import Flask, request, jsonify
import boto3
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Set up custom S3 client for VultroObjects
s3_client = boto3.client(
    's3',
    aws_access_key_id='9KJQ2JY4OWRTDKCW52LQ',
    aws_secret_access_key='hjpHbaR4ZsFIaumrJXcFSbOHwcuKhlbxFezoIpF8',
    endpoint_url='https://sgp1.vultrobjects.com',
    config=Config(signature_version='s3v4')
)

# Set the name of your S3 bucket
bucket_name = 'bill'

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    
    # Upload to S3
    s3_client.upload_fileobj(file, bucket_name, filename)
    
    return jsonify({"message": "File uploaded successfully", "filename": filename}), 200

# Verification endpoint logic
@app.route('/verification', methods=['POST'])
def verification():
    s3_bucket = 'bill'
    files_in_s3 = s3_client.list_objects_v2(Bucket=s3_bucket).get('Contents', [])
    
    for file_info in files_in_s3:
        file_path = file_info['Key']  # Get file path

        # Check file format and apply different processing based on PDF or image
        if file_path.lower().endswith('.pdf'):
            # Process PDF files with only layerTwo function
            score = layerTwo(file_path, 0)  # Run layerTwo function from threeLayer.py

            # Scoring logic for PDFs
            if score == 35:
                # Copy to authentic bucket
                s3_client.copy_object(
                    Bucket='authentic_bill',
                    CopySource={'Bucket': s3_bucket, 'Key': file_path},
                    Key=file_path
                )
            else:
                # Copy to fake bucket
                s3_client.copy_object(
                    Bucket='fake_bill',
                    CopySource={'Bucket': s3_bucket, 'Key': file_path},
                    Key=file_path
                )

            # Delete from original bucket after copying
            s3_client.delete_object(Bucket=s3_bucket, Key=file_path)

        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Initialize score for image processing
            score = 0
            
            # Run all three layers sequentially for image files and update score
            score += layerOne(file_path, 0)
            score += layerTwo(file_path, 0)
            score += layerThree(file_path, 0)

            # Scoring logic for images
            if score > 65:
                # Copy to authentic bucket
                s3_client.copy_object(
                    Bucket='authentic_bill',
                    CopySource={'Bucket': s3_bucket, 'Key': file_path},
                    Key=file_path
                )
            else:
                # Copy to fake bucket
                s3_client.copy_object(
                    Bucket='fake_bill',
                    CopySource={'Bucket': s3_bucket, 'Key': file_path},
                    Key=file_path
                )

            # Delete from original bucket after copying
            s3_client.delete_object(Bucket=s3_bucket, Key=file_path)


    return jsonify({"message": "Verification completed!"}), 200

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='45.32.19.96', port=5000)
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

            return "Fake", score

        else:

            score += 35

            return "Authentic", score



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

                    tag_name = TAGS.get(tag, tag)

                    metadata[tag_name] = value



            # Add file creation and modification dates from the file system

            file_stats = os.stat(image_path)

            creation_date = datetime.fromtimestamp(file_stats.st_ctime)

            modification_date = datetime.fromtimestamp(file_stats.st_mtime)

            metadata["FileCreationDate"] = creation_date.strftime("%Y-%m-%d %H:%M:%S")

            metadata["FileModificationDate"] = modification_date.strftime("%Y-%m-%d %H:%M:%S")



            # Authenticity Check based on creation and modification dates

            time_diff = modification_date - creation_date



            # Define a threshold (e.g., 2 hours) to flag suspicious differences

            threshold = timedelta(hours=2)

            

            if time_diff > threshold:

                metadata["SuspiciousModification"] = f"Image was modified {time_diff} after creation. This might be suspicious."

            else:

                metadata["SuspiciousModification"] = "Image seems to be consistent with creation and modification dates."

                curr_score += 35  # Increase curr_score by 35 if the image is authentic



        except Exception as e:

            metadata["Error"] = f"Error fetching metadata: {e}"



        return metadata, curr_score



    # Function to extract PDF metadata using PyMuPDF

    def check_pdf_metadata(file_path, curr_score):

        suspicious_metadata = []

        mod_date = None  # Initialize the mod_date variable

        try:

            # Open the PDF file

            doc = fitz.open(file_path)

            

            # Extract metadata

            metadata = doc.metadata

            

            # Display PDF metadata

            print("PDF Metadata:")

            for key, value in metadata.items():

                print(f"{key}: {value}")



            # Check for recent edits based on the 'modDate' field

            mod_date_str = metadata.get("modDate")

            if mod_date_str:

                try:

                    # Remove the 'D:' prefix and strip off the timezone part (e.g., '+00'00)

                    if mod_date_str.startswith("D:"):

                        mod_date_str = mod_date_str[2:]  # Remove the "D:" prefix

                    mod_date_str = mod_date_str.split('+')[0]  # Remove the timezone part

                    

                    # Attempt to parse modDate using standard PDF format

                    mod_date = datetime.strptime(mod_date_str, '%Y%m%d%H%M%S')



                    print(f"Parsed modification date: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")



                    # Check if the date was successfully parsed

                    if mod_date:

                        # Check if the modification date is within the last 30 days

                        if datetime.now() - mod_date < timedelta(days=30):

                            suspicious_metadata.append("Document was modified recently.")

                            print("Warning: Document was modified within the last 30 days.")

                        else:

                            curr_score += 35  # Increase curr_score if PDF is deemed authentic

                    else:

                        print("Unable to parse modification date.")



                except Exception as e:

                    print(f"Error parsing modification date: {e}")



            # Check for suspicious author or producer information

            author = metadata.get("author", "")

            producer = metadata.get("producer", "")

            if "unknown" in author.lower() or "unknown" in producer.lower():

                suspicious_metadata.append("Author or producer information is missing or unknown.")

                print("Warning: Author or producer metadata appears suspicious.")

        

        except Exception as e:

            print(f"Error processing PDF metadata: {e}")

        

        return suspicious_metadata, curr_score



    # Main function to process the file based on its type

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



        print(f"\nCurrent Score: {curr_score}")



    # Example usage - Initialize curr_score before calling the function

    curr_score = 0

    process_file(path, curr_score)

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

        return 0

from flask import Flask, render_template, request, redirect, url_for
import boto3
import os

app = Flask(__name__)

# Configure your S3 client here
s3 = boto3.client(
    's3',
    endpoint_url='https://sgp1.vultrobjects.com',
    aws_access_key_id='9KJQ2JY4OWRTDKCW52LQ',
    aws_secret_access_key='hjpHbaR4ZsFIaumrJXcFSbOHwcuKhlbxFezoIpF8'
)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Define the S3 bucket and upload file
            bucket_name = 'bill'  # Change this to your bucket name
            s3.upload_fileobj(file, bucket_name, file.filename)
            return "File uploaded successfully!"
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
