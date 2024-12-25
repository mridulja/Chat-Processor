import os
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def verify_setup():
  # Load environment variables
  load_dotenv()
  
  # Check environment variables
  required_vars = ['SPREADSHEET_ID', 'GOOGLE_CLOUD_PROJECT', 'OPENAI_API_KEY']
  for var in required_vars:
      value = os.getenv(var)
      if not value:
          print(f"❌ Missing environment variable: {var}")
          return False
      print(f"✅ Found {var}")
  
  # Check Google Cloud authentication
  try:
      credentials, project = default()
      print(f"✅ Google Cloud authentication successful")
      print(f"✅ Project ID: {project}")
  except Exception as e:
      print(f"❌ Google Cloud authentication failed: {str(e)}")
      return False
  
  # Try to access the spreadsheet
  try:
      service = build('sheets', 'v4', credentials=credentials)
      sheet = service.spreadsheets().get(
          spreadsheetId=os.getenv('SPREADSHEET_ID')
      ).execute()
      print(f"✅ Successfully accessed spreadsheet: {sheet['properties']['title']}")
  except HttpError as e:
      print(f"❌ Failed to access spreadsheet: {str(e)}")
      return False
  
  return True

if __name__ == "__main__":
  print("Verifying setup...")
  if verify_setup():
      print("\n✅ All systems go! Your setup is complete and working.")
  else:
      print("\n❌ Setup verification failed. Please check the errors above.")