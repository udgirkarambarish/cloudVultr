# Running the Flask Application on Your IP Address

To run this application on your machine or server with your own IP address, you will need to update the `host` parameter in the code to reflect your specific IP.

## Steps to Run
1. git clone <repository_url>
   cd <repository_folder>

2. Set Up Virtual Environment 
   python3 -m venv myenv
   source myenv/bin/activate    # On Linux/Mac
   myenv\scripts\activate        #On Windows

3. Installation
   pip install -r requirements.txt

4. Configure Environment Variables
   export FLASK_APP=app.py
   export FLASK_ENV=development     # For development mode
   export DATABASE_URL=mysql://<user>:<password>@<host>/<database>
   export AWS_ACCESS_KEY_ID=<your_aws_access_key>
   export AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>

   Modify IP Address 
   python
   # Run the app with your Ip
     if __name__ == '__main__':
         app.run(debug=True,
     host='YOUR_IP_ADDRESS', port=5000)

5. Initialise The Database
   flask db init
   flask db migrate
   flask db upgrade

6. Run The App Locally 
   flask run

7. OR Deploy With Gunicorn (Recommended)
   sudo fuser -k 8000/tcp
   gunicorn --workers 4 --bind 0.0.0.0:8000 app:app

8. Access The App On Your Server IP Address or http://localhost:8000

### Known Issues with TensorFlow and GPU Compatibility

Our project encountered a challenge with TensorFlow. The primary issue relates to TensorFlow’s attempt to utilize GPU acceleration. While we tried disabling GPU to force TensorFlow to run on the CPU, the application still seeks a GPU-accelerated environment. This results in potential performance issues for our Flask app.

### Current Limitations

Our web application is accessible at [http://65.20.76.106:8000/](http://65.20.76.106:8000/). However, **User registration and login functionalities are currently not working due to GPU problem**.
