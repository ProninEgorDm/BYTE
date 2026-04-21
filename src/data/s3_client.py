import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError, ClientError
import os
from io import BytesIO
from PIL import Image


class MinIOS3Client:
    
    def __init__(self, endpoint_url='http://localhost:9000', access_key='minioadmin', 
                 secret_key='minioadmin123', region_name='us-east-1'):
        """
        
        Args:
            endpoint_url (str): MinIO endpoint URL.
            access_key (str): MinIO access key.
            secret_key (str): MinIO secret key.
            region_name (str): AWS region name.
        """
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name
        self.client = None
        self.connect()
    
    def connect(self):
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name
            )
            print(" Connected to MinIO successfully.")
        except Exception as e:
            print(f" Failed to connect: {e}")
    
    def create_bucket(self, bucket_name):
        try:
            self.client.create_bucket(Bucket=bucket_name)
            print(f" Bucket '{bucket_name}' created.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"  Bucket '{bucket_name}' already exists.")
            else:
                print(f" Error creating bucket: {e}")
    
    def list_buckets(self):
        try:
            response = self.client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            print(f" Buckets: {buckets}")
            return buckets
        except ClientError as e:
            print(f" Error listing buckets: {e}")
            return []
    
    def list_objects(self, bucket_name, prefix=''):
        try:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            objects = [obj['Key'] for obj in response.get('Contents', [])]
            print(f"📁 Objects in '{bucket_name}' (prefix='{prefix}'): {len(objects)} found")
            return objects
        except ClientError as e:
            print(f" Error listing objects: {e}")
            return []
    
    def upload_file(self, bucket_name, file_path, object_key):
        try:
            self.client.upload_file(file_path, bucket_name, object_key)
            print(f" Uploaded '{file_path}' to '{bucket_name}/{object_key}'.")
        except ClientError as e:
            print(f" Error uploading file: {e}")
    
    def download_file(self, bucket_name, object_key, file_path):
        try:
            self.client.download_file(bucket_name, object_key, file_path)
            print(f" Downloaded '{bucket_name}/{object_key}' to '{file_path}'.")
        except ClientError as e:
            print(f" Error downloading file: {e}")
    
    def upload_dataframe(self, df, bucket_name, object_key, format='parquet'):
        buffer = BytesIO()
        try:
            if format == 'parquet':
                df.to_parquet(buffer, index=False)
            elif format == 'csv':
                df.to_csv(buffer, index=False)
            else:
                print(f" Unsupported format: {format}")
                return
            
            buffer.seek(0)
            self.client.put_object(Bucket=bucket_name, Key=object_key, Body=buffer.getvalue())
            print(f" Uploaded DataFrame to '{bucket_name}/{object_key}' ({format}).")
        except ClientError as e:
            print(f" Error uploading DataFrame: {e}")
    
    def download_dataframe(self, bucket_name, object_key, format='parquet'):
        try:
            response = self.client.get_object(Bucket=bucket_name, Key=object_key)
            buffer = BytesIO(response['Body'].read())
            
            if format == 'parquet':
                df = pd.read_parquet(buffer)
            elif format == 'csv':
                df = pd.read_csv(buffer)
            else:
                print(f" Unsupported format: {format}")
                return None
            
            print(f"✅ Downloaded DataFrame from '{bucket_name}/{object_key}' ({format}).")
            return df
        except ClientError as e:
            print(f" Error downloading DataFrame: {e}")
            return None
    
    def upload_image(self, image_path, bucket_name, object_key):
        try:
            with open(image_path, 'rb') as f:
                self.client.put_object(Bucket=bucket_name, Key=object_key, Body=f.read())
            print(f" Uploaded image '{image_path}' to '{bucket_name}/{object_key}'.")
        except ClientError as e:
            print(f" Error uploading image: {e}")
        except FileNotFoundError:
            print(f" Image file not found: {image_path}")
    
    def download_image(self, bucket_name, object_key, save_path):
        try:
            response = self.client.get_object(Bucket=bucket_name, Key=object_key)
            image_data = response['Body'].read()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(image_data)
            print(f" Downloaded image from '{bucket_name}/{object_key}' to '{save_path}'.")
        except ClientError as e:
            print(f" Error downloading image: {e}")
    
    def get_image_as_pil(self, bucket_name, object_key):
        try:
            response = self.client.get_object(Bucket=bucket_name, Key=object_key)
            image_data = response['Body'].read()
            image = Image.open(BytesIO(image_data))
            print(f" Loaded image from '{bucket_name}/{object_key}' as PIL Image.")
            return image
        except ClientError as e:
            print(f" Error loading image: {e}")
            return None
    
    def delete_object(self, bucket_name, object_key):
        try:
            self.client.delete_object(Bucket=bucket_name, Key=object_key)
            print(f" Deleted '{bucket_name}/{object_key}'.")
        except ClientError as e:
            print(f" Error deleting object: {e}")
    
    def delete_bucket(self, bucket_name):
        try:
            self.client.delete_bucket(Bucket=bucket_name)
            print(f" Deleted bucket '{bucket_name}'.")
        except ClientError as e:
            print(f" Error deleting bucket: {e}")
    
    def object_exists(self, bucket_name, object_key):
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError:
            return False