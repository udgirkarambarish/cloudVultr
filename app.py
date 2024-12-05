from flask import Flask
from threeLayer import *
import boto3
from botocore.client import Config
from flask import Flask, request, jsonify
import boto3
from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import datetime
import fitz  # PyMuPDF for PDF metadata
from datetime import datetime, timedelta
import easyocr
import google.generativeai as genai
from pyzbar.pyzbar import decode
import tensorflow as tf
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Initialize the Gemini API with the hardcoded API key
genai.configure(api_key='')

s3 = boto3.client(
    's3',
    # endpoint_url='https://s3.ap-south-1.amazonaws.com',  # Your Object Storage hostname
    aws_access_key_id='',     # Your access key
    aws_secret_access_key=''  # Your secret key
)

# Helper function to get image from S3
def get_image_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    return response['Body'].read()  # Return image data

# Helper function to get PDF from S3
def get_pdf_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    return response['Body'].read()

@app.route('/verification', methods=['GET', 'POST'])
def verification():
    result_status = ''
    bucket_name = 'general-bills'
    local_path = None  # Define this variable to track the local file path

    try:
        # Fetch the most recent file from the 'general-bills' bucket
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='', MaxKeys=1, Delimiter='/')
        if 'Contents' in response:
            # Get the most recent file
            most_recent_file = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
            file_key = most_recent_file['Key']

            print(f"Most recent file key: {file_key}")  # Debug

            # Download the file
            file_name = secure_filename(file_key.split('/')[-1])
            local_path = os.path.join(os.getcwd(), file_name)
            s3.download_file(bucket_name, file_key, local_path)

            # Process the file based on type
            if file_name.lower().endswith('.pdf'):
                score = layerTwo(local_path)
                target_bucket = 'authentic-bills' if score else 'fake-bills'
            elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                score1 = layerOne(local_path)
                print(score1)
                score2 = layerTwo(local_path)
                print(score2)
                score3 = layerThree(local_path)
                print(score3)
                target_bucket = 'authentic-bills' if (score1 and score2) or score3 else 'fake-bills'
            else:
                raise ValueError("Unsupported file type")

            # Copy and delete the file
            print(f"Copying file to bucket: {target_bucket}")  # Debug
            s3.copy_object(Bucket=target_bucket, CopySource={'Bucket': bucket_name, 'Key': file_key}, Key=file_key)
            s3.delete_object(Bucket=bucket_name, Key=file_key)
            result_status = target_bucket
        else:
            print("No files found in the 'general-bills' bucket")
            return render_template('verification.html', results=None)

        result = {'file': file_name, 'status': result_status}
        return render_template('verification.html', results=[result])

    except Exception as e:
        print(f"Unexpected error: {e}")
        return render_template('verification.html', results=None)

    finally:
        # Ensure the locally downloaded file is deleted
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            print(f"Deleted local file: {local_path}")

# In layerOne function:
def layerOne(path):
    print("Starting Layer One Processing...")

    import numpy as np
    import tensorflow as tf
    import cv2  # OpenCV for image processing

    # Function to preprocess image
    def preprocess_image(image_path):
        # Load the image from the given local path
        image = cv2.imread(image_path)

        # Check if the image was successfully loaded
        if image is None:
            raise ValueError(f"Image at path '{image_path}' could not be loaded. Please check the file path.")

        # If the image is grayscale (i.e., it has only 1 channel), convert it to 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale image has 2 dimensions
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

        # Resize image to a fixed size expected by the model (e.g., 224x224)
        image = cv2.resize(image, (224, 224))

        # Normalize the image to scale pixel values to [0, 1] range
        image = image.astype(np.float32) / 255.0

        # Convert the image from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    # Predict authenticity of the image
    def predict_image(local_path):
        # Fetch and preprocess the image
        image = preprocess_image(local_path)  # Load and preprocess the local image
        image = np.expand_dims(image, axis=0)

        # Perform inference
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Decision based on prediction threshold
        if prediction > 0.5:
            print(f"Prediction: {prediction} (Authentic)")
            return 1  # No change to score, image considered authentic
        else:
            print(f"Prediction: {prediction} (Fake/Low Quality)")
            return 0  # Add 35 to score for low-quality prediction

    # Call predict_image with the local image path
    print("Starting prediction on local image...")
    score = predict_image(path)
    print("Prediction complete.")

    return score

# Main function to process the file
def layerTwo(path):
    import os
    from PIL import Image
    from PIL.ExifTags import TAGS
    from datetime import datetime, timedelta
    import fitz  # PyMuPDF
    from werkzeug.utils import secure_filename

    # Function to detect file type
    def detect_file_type(file_path):
        print('detect_file_type')
        if file_path.lower().endswith('.pdf'):
            return 'pdf'
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            return 'image'
        return None

    # Function to extract image metadata
    def get_image_metadata(image_path):
        print('get_image_metadata')
        metadata = {}
        try:
            # Open the image and extract EXIF data
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    metadata[tag_name] = value

            # Add file creation and modification dates
            file_stats = os.stat(image_path)
            creation_date = datetime.fromtimestamp(file_stats.st_ctime)
            modification_date = datetime.fromtimestamp(file_stats.st_mtime)
            metadata["FileCreationDate"] = creation_date.strftime("%Y-%m-%d %H:%M:%S")
            metadata["FileModificationDate"] = modification_date.strftime("%Y-%m-%d %H:%M:%S")

            # Authenticity Check
            time_diff = modification_date - creation_date
            threshold = timedelta(hours=2)
            if time_diff > threshold:
                metadata["SuspiciousModification"] = f"Image modified {time_diff} after creation."
                return 0
            else:
                metadata["SuspiciousModification"] = "Image seems authentic."
                return 1  # Increase score for authenticity
        except Exception as e:
            metadata["Error"] = f"Error fetching metadata: {e}"

    # Function to extract PDF metadata
    def check_pdf_metadata(file_path):
        print('check_pdf_metadata')
        suspicious_metadata = []
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            print("PDF Metadata:", metadata)

            # Check for recent edits
            mod_date_str = metadata.get("modDate")
            if mod_date_str:
                try:
                    if mod_date_str.startswith("D:"):
                        mod_date_str = mod_date_str[2:]
                    mod_date_str = mod_date_str.split('+')[0]
                    mod_date = datetime.strptime(mod_date_str, '%Y%m%d%H%M%S')

                    if datetime.now() - mod_date < timedelta(days=30):
                        suspicious_metadata.append("Document modified recently.")
                        return 0  # Return no score change for suspicious edits
                    else:
                        return 1  # Return 35 for a document that seems authentic
                except Exception as e:
                    print(f"Error parsing modDate: {e}")

            # Check author/producer metadata
            if "unknown" in metadata.get("author", "").lower() or "unknown" in metadata.get("producer", "").lower():
                suspicious_metadata.append("Suspicious author or producer information.")
        except Exception as e:
            print(f"Error processing PDF metadata: {e}")

    file_type = detect_file_type(path)
    if file_type == 'pdf':
        print("Starting check_pdf_metadata --------------")
        return check_pdf_metadata(path)
    elif file_type == 'image':
        print("Starting get_image_metadata --------------")
        return get_image_metadata(path)
    else:
        print("Unsupported file format.")



def layerThree(path):

    # Initialize the EasyOCR Reader for English
    reader = easyocr.Reader(['en'])

    # Function to enhance the image for OCR and QR code extraction
    def enhance_image(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise
        enhanced = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Threshold
        return enhanced

    # Enhance the image
    enhanced_image = enhance_image(path)

    # Save the enhanced image temporarily for EasyOCR to read
    temp_path = "enhanced_temp.png"
    cv2.imwrite(temp_path, enhanced_image)

    try:
        # Step 1: Extract text from the enhanced image using OCR
        extracted_text = reader.readtext(temp_path, detail=0)
        extracted_text_str = " ".join(extracted_text)

        # Step 2: Detect and extract QR code data from the enhanced image
        def extract_qr_data(enhanced_image):
            qr_codes = decode(enhanced_image)
            if qr_codes:
                return qr_codes[0].data.decode("utf-8")
            return None

        qr_data = extract_qr_data(enhanced_image)

        # Step 3: Send both OCR and QR extracted data to AI model for authentication
        def authenticate_with_ai(ocr_text, qr_text):
            prompt = (
                f"Authenticate the following extracted data from a bill or invoice. "
                f"Only compare the actual values for 'Date' and 'Amount' fields between OCR and QR data, "
                f"and ignore any differences in formatting or labels.\n\n"
                f"OCR Extracted Data: {ocr_text}\n"
                f"QR Code Extracted Data: {qr_text}\n\n"
                f"Provide a concise result and note any mismatched fields only if they exist.\n"
                f"Result:"
            )
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip() if response and hasattr(response, 'text') else "Unknown result"

        # Run the AI-based authentication
        if qr_data:  # If QR data is available, proceed with AI-based comparison
            ai_result = authenticate_with_ai(extracted_text_str, qr_data)
            print("Gemini AI Authentication Result:", ai_result)

            # Update curr_score based on AI result
            if "authentic" in ai_result.lower():
                print("authentic")
                print("completed 3 -------------------------------------------------------------------------------------------")

                return 1
            else:
                print("fake")
                print("completed 3 -------------------------------------------------------------------------------------------")
                return 0
        else:
            print("No QR Code found. Unable to authenticate using AI.")
            print("completed 3 -------------------------------------------------------------------------------------------")
            return 0

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                bucket_name = 'general-bills'  # Ensure this is your exact bucket name
                
                # Sanitize the filename
                import os
                filename = os.path.basename(file.filename)
                
                # Upload file to the bucket
                s3.upload_fileobj(file, bucket_name, filename)
                return "File uploaded successfully!"
        except s3.exceptions.NoSuchBucket:
            return "Error: The bucket does not exist."
        except s3.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            return f"ClientError occurred: {error_code} - {e.response['Error']['Message']}", 404
        except Exception as e:
            return f"An error occurred: {e}", 500
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)