import boto3
from botocore.exceptions import ClientError

# Initialize the S3 client
s3 = boto3.client(
    's3',
    endpoint_url='https://del1.vultrobjects.com',
    aws_access_key_id='9CDGUYGZRK12D4VO084M',
    aws_secret_access_key='KTiaIBhDCDG7anxdmt5u5Bs72jukKShs1DifZBG8'
)

# List buckets to confirm the connection
try:
    response = s3.list_buckets()
    print("Buckets:", [bucket['Name'] for bucket in response['Buckets']])
except ClientError as e:
    error_code = e.response['Error']['Code']
    error_message = e.response['Error']['Message']
    print(f"ClientError occurred: Code: {error_code}, Message: {error_message}")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
