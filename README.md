# Running the Flask Application on Your IP Address

To run this application on your machine or server with your own IP address, you will need to update the `host` parameter in the code to reflect your specific IP.

## Steps to Run

1. **Locate the Code Section**  
   Find the following code block at the bottom of the application file:

    ```python
    # Run the app
    if __name__ == '__main__':
        app.run(debug=True, host='45.32.19.96', port=5000)
    ```

2. **Modify the IP Address**  
   Replace `'45.32.19.96'` with your own IP address:

    ```python
    # Run the app with your IP
    if __name__ == '__main__':
        app.run(debug=True, host='YOUR_IP_ADDRESS', port=5000)
    ```

   Replace `YOUR_IP_ADDRESS` with the IP address of the machine where you want the Flask app to be accessible.

3. **Run the Application**  
   After making the change, save the file and run it to start the application on your specified IP.

## Installation

To install the required dependencies for this project, run the following command:

```bash
pip install -r requirements.txt
```

### Known Issues with TensorFlow and GPU Compatibility

Our project encountered a challenge with TensorFlow. The primary issue relates to TensorFlow’s attempt to utilize GPU acceleration. While we tried disabling GPU to force TensorFlow to run on the CPU, the application still seeks a GPU-accelerated environment. This results in potential performance issues for our Flask app.

### Current Limitations

Our web application is accessible at [http://65.20.76.106:8000/](http://65.20.76.106:8000/). However, **User registration and login functionalities are currently not working due to GPU problem**.
