import streamlit as st
import os
from dotenv import load_dotenv
import openai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from datetime import datetime
import json
import logging
from google.oauth2 import service_account
import re
import glob
from tkinter import filedialog
import tkinter as tk
import shutil
import time
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configure Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

def setup_google_sheets():
    try:
        # Load credentials from the service account file
        credentials = service_account.Credentials.from_service_account_file(
            'ywca-chatanalyzer-e829d0e72c29.json',  # Update this path
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        # Build the service
        service = build('sheets', 'v4', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"Failed to setup Google Sheets: {str(e)}")
        return None

def create_or_get_sheet(service):
    if not service:
        st.error("Google Sheets service not initialized")
        return None
        
    try:
        # Try to get the existing sheet
        sheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        return sheet
    except HttpError as e:
        if e.resp.status == 404:
            # Create new spreadsheet if it doesn't exist
            sheet_body = {
                'properties': {'title': 'YWCA Chat Summary'},
                'sheets': [{
                    'properties': {
                        'title': 'Chat Summaries',
                        'gridProperties': {'frozenRowCount': 1}
                    }
                }]
            }
            try:
                sheet = service.spreadsheets().create(body=sheet_body).execute()
                # Add headers
                headers = [['Timestamp', 'Name', 'Email', 'Context', 'Action Required', 'Headline']]
                service.spreadsheets().values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range='A1:F1',
                    valueInputOption='RAW',
                    body={'values': headers}
                ).execute()
                return sheet
            except HttpError as create_error:
                st.error(f"Failed to create sheet: {str(create_error)}")
                return None
        else:
            st.error(f"Failed to access sheet: {str(e)}")
            return None

def analyze_chat_with_gpt(chat_text):
    prompt = f"""Please analyze this chat and extract the following information:
    1. Context of the conversation, context should be clear and concise
    2. Name of the visitor, if name is found else NA
    3. Email address, if email is found else NA
    4. Action required by YWCAGLA team, make sure action is clear and concise, if action is found else NA
    5. A brief headline summarizing the conversation

    Chat:
    {chat_text}

    Please respond in a JSON format with these keys: context, name, email, action_required, headline"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Safely parse JSON response
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response. Please try again.")
        return None

def clear_previous_formatting(service, spreadsheet_id):
    """
    Clear formatting from all rows in the spreadsheet.
    
    Args:
        service: Google Sheets API service instance
        spreadsheet_id (str): ID of the target spreadsheet
    """
    try:
        # Get the sheet metadata to find the sheet ID
        sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_id = sheet_metadata['sheets'][0]['properties']['sheetId']
        
        # Create the request to clear formatting
        request = {
            'requests': [{
                'updateCells': {
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': 1  # Skip header row
                    },
                    'fields': 'userEnteredFormat.backgroundColor'
                }
            }]
        }
        
        # Execute the request
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=request
        ).execute()
        
    except Exception as e:
        logger.error(f"Error clearing previous formatting: {str(e)}")

def append_to_sheet(service, data):
    """
    Append new data to the Google Sheet and highlight newly added rows.
    
    Args:
        service: Google Sheets API service instance
        data (dict): Analysis data to be appended
    """
    try:
        # Get the first sheet's ID and metadata
        sheet_metadata = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        properties = sheet_metadata.get('sheets', '')[0].get('properties', '')
        sheet_id = properties.get('sheetId')
        sheet_title = properties.get('title', 'Sheet1')
        
        # Replace empty or missing values with "NA"
        name = data['name'].strip() if data.get('name') and data['name'].strip() else "NA"
        email = data['email'].strip() if data.get('email') and data['email'].strip() else "NA"
        
        # Prepare row data
        values = [[
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            email,
            data['context'],
            data['action_required'],
            data['headline']
        ]]
        
        # Clear previous formatting
        clear_previous_formatting(service, SPREADSHEET_ID)
        
        # Append the new row
        append_result = service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f'{sheet_title}!A:F',
            valueInputOption='RAW',
            body={'values': values},
            insertDataOption='INSERT_ROWS'
        ).execute()
        
        # Get the row number where data was inserted
        updated_range = append_result.get('updates', {}).get('updatedRange', '')
        match = re.search(r'!A(\d+)', updated_range)
        if match:
            row_number = int(match.group(1))
            
            # Prepare formatting request for the new row
            request = {
                'requests': [{
                    'updateCells': {
                        'range': {
                            'sheetId': sheet_id,
                            'startRowIndex': row_number - 1,  # 0-based index
                            'endRowIndex': row_number,
                            'startColumnIndex': 0,
                            'endColumnIndex': 6  # A through F
                        },
                        'rows': [{
                            'values': [{
                                'userEnteredFormat': {
                                    'backgroundColor': {
                                        'red': 0.9,
                                        'green': 0.9,
                                        'blue': 1.0  # Light blue background
                                    }
                                }
                            } for _ in range(6)]  # One for each column
                        }],
                        'fields': 'userEnteredFormat.backgroundColor'
                    }
                }]
            }
            
            # Apply the formatting
            service.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body=request
            ).execute()
            
    except HttpError as e:
        st.error(f"Error appending to sheet: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def get_sheet_data(service):
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='A:F'
        ).execute()
        values = result.get('values', [])
        if values:
            df = pd.DataFrame(values[1:], columns=values[0])
            return df
        return pd.DataFrame()
    except HttpError as e:
        st.error(f"Error fetching sheet data: {str(e)}")
        return pd.DataFrame()

def process_json_chat(json_data):
    """Convert JSON chat format to a single string for analysis."""
    chat_text = ""
    for message in json_data:
        from_who = message.get('from', '')
        text = message.get('text', '').replace('<p>', '\n').replace('</p>', '')
        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        chat_text += f"{from_who}: {text}\n"
    return chat_text

def select_folder():
    """Open folder selection dialog and return the chosen path."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.wm_attributes('-topmost', 1)  # Make sure window is on top
        folder_path = filedialog.askdirectory(parent=root, title="Select Folder containing JSON files")
        root.destroy()
        return folder_path
    except Exception as e:
        logger.error(f"Error in folder selection: {str(e)}")
        return None

def ensure_processed_folder(source_folder):
    """Create 'processed' folder if it doesn't exist."""
    processed_folder = os.path.join(source_folder, "processed")
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    return processed_folder

def process_json_files(folder_path, service):
    """
    Process multiple JSON chat files from a specified folder and analyze their contents.
    
    This function performs the following operations:
    1. Identifies all JSON files in the specified folder
    2. Processes each file through GPT analysis
    3. Moves processed files to a 'processed' subfolder with modified filenames
    4. Tracks processing statistics and errors
    
    Args:
        folder_path (str): Path to the folder containing JSON chat files
        service (googleapiclient.discovery.Resource): Initialized Google Sheets API service
        
    Returns:
        tuple: (summary_data, error_files, processed_folder)
            - summary_data (dict): Statistics about processed files
            - error_files (list): List of files that encountered errors
            - processed_folder (str): Path to the folder containing processed files
    """
    try:
        # Find all JSON files in the specified folder, excluding already processed ones
        json_files = [f for f in glob.glob(os.path.join(folder_path, "*.json")) 
                     if "processed" not in f]
        
        if not json_files:
            st.warning("No JSON files found in the selected folder.")
            return
        
        # Initialize processing environment
        processed_folder = ensure_processed_folder(folder_path)
        progress_bar = st.progress(0)
        processed_count = 0
        total_files = len(json_files)
        
        # Initialize tracking metrics
        summary_data = {
            'total_files': total_files,
            'processed': 0,
            'errors': 0,
            'names_found': 0,
            'emails_found': 0,
            'actions_required': 0
        }
        error_files = []
        
        # Process each JSON file
        for index, file_path in enumerate(json_files):
            file_handle = None
            try:
                # Verify file accessibility before processing
                if not os.access(file_path, os.R_OK):
                    raise IOError(f"File is being used by another process: {file_path}")
                
                # Read and parse JSON chat data
                with open(file_path, 'r', encoding='utf-8') as file_handle:
                    chat_data = json.load(file_handle)
                
                # Convert JSON structure to plain text for analysis
                chat_text = process_json_chat(chat_data)
                
                if chat_text:
                    # Perform GPT analysis on the chat content
                    analysis = analyze_chat_with_gpt(chat_text)
                    if analysis:
                        # Record analysis results in Google Sheets
                        append_to_sheet(service, analysis)
                        processed_count += 1
                        
                        # Update processing statistics
                        summary_data['processed'] += 1
                        if analysis.get('name') and analysis['name'] != 'NA':
                            summary_data['names_found'] += 1
                        if analysis.get('email') and analysis['email'] != 'NA':
                            summary_data['emails_found'] += 1
                        if analysis.get('action_required') and analysis['action_required'] != 'NA':
                            summary_data['actions_required'] += 1
                        
                        # Ensure file handle is closed before file operations
                        if file_handle:
                            file_handle.close()
                        
                        # Brief pause to ensure file system operations complete
                        time.sleep(0.1)
                        
                        try:
                            # Generate new filename with metadata
                            original_filename = os.path.basename(file_path)
                            filename_without_ext = os.path.splitext(original_filename)[0]
                            current_date = datetime.now().strftime("%Y%m%d")
                            
                            # Add email username to filename if available
                            email_prefix = ""
                            if analysis.get('email') and analysis['email'] != 'NA':
                                email_prefix = f"_{analysis['email'].split('@')[0]}"
                            
                            # Construct new filename with format: original_YYYYMMDD_emailuser.json
                            new_filename = f"{filename_without_ext}_{current_date}{email_prefix}.json"
                            new_path = os.path.join(processed_folder, new_filename)
                            
                            # Move processed file to new location
                            shutil.move(file_path, new_path)
                        except PermissionError:
                            st.warning(f"Could not move file {os.path.basename(file_path)}. Please close any programs that might be using it.")
                            continue
                
            except Exception as e:
                # Track any errors encountered during processing
                error_files.append((file_path, str(e)))
                summary_data['errors'] += 1
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
            finally:
                # Update progress indicator
                progress_bar.progress((index + 1) / total_files)
                # Ensure proper file handle cleanup
                if file_handle and not file_handle.closed:
                    file_handle.close()
        
        return summary_data, error_files, processed_folder
        
    except Exception as e:
        # Handle any unexpected errors during overall processing
        st.error(f"Error processing JSON files: {str(e)}")
        return None, None, None

def categorize_chats(context):
    """Categorize chat based on context."""
    categories = {
        'Child Development': ['child', 'children', 'daycare', 'preschool', 'early education', 'head start'],
        'Senior Services': ['senior', 'elderly', 'aging', 'medicare', 'retirement'],
        'SACS': ['domestic violence', 'abuse', 'shelter', 'safety', 'crisis', 'assault', 'violence'],
        'Youth Development': ['youth', 'teen', 'adolescent', 'after school', 'summer program'],
        'Volunteering': ['volunteer', 'help out', 'giving back', 'community service'],
        'Housing': ['housing', 'shelter', 'apartment', 'rent', 'living space'],
        'Employment': ['job', 'career', 'employment', 'work', 'hiring', 'position'],
        'General Inquiry': ['information', 'contact', 'question', 'inquiry', 'hours', 'location'],
        'Donations': ['donate', 'contribution', 'giving', 'support', 'charity'],
        'Events': ['event', 'program', 'workshop', 'seminar', 'meeting']
    }
    
    context = context.lower()
    for category, keywords in categories.items():
        if any(keyword in context for keyword in keywords):
            return category
    return 'Other'

def analyze_chat_data(df):
    """Analyze the chat data and return statistics."""
    try:
        if df.empty:
            return None
            
        # Create a mapping of possible column names to standardized names
        column_mapping = {
            'Date': 'Timestamp',
            'Timestamp': 'Timestamp',
            'Name': 'Name',
            'email': 'Email',
            'Email': 'Email',
            'Context': 'Context',
            'Action Needed': 'Action Required',
            'Action Required': 'Action Required',
            'headline': 'Headline',
            'Headline': 'Headline'
        }
        
        # Rename columns based on mapping
        df_columns = df.columns.to_list()
        rename_dict = {}
        for old_col in df_columns:
            if old_col in column_mapping:
                rename_dict[old_col] = column_mapping[old_col]
        
        df = df.rename(columns=rename_dict)
        
        # Verify required columns exist
        required_columns = {'Timestamp', 'Name', 'Email', 'Context', 'Action Required', 'Headline'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.error(f"Still missing columns after mapping: {missing_columns}")
            logger.error(f"Current columns: {set(df.columns)}")
            return None
        
        # Add category column
        df['Category'] = df['Context'].apply(categorize_chats)
        
        # Initialize statistics dictionary
        stats = {
            'total_chats': len(df),
            'categories': {},
            'email_collection': {
                'total_emails': len(df[df['Email'].fillna('NA').str.lower() != 'na']),
                'by_category': {}
            },
            'action_required': {
                'total': len(df[df['Action Required'].fillna('NA').str.lower() != 'na']),
                'by_category': {}
            },
            'timeline': {}
        }
        
        # Category statistics
        category_counts = df['Category'].value_counts()
        for category in category_counts.index:
            stats['categories'][category] = {
                'count': int(category_counts[category]),
                'percentage': round(category_counts[category] / len(df) * 100, 2),
                'emails_collected': len(df[(df['Category'] == category) & 
                                        (df['Email'].fillna('NA').str.lower() != 'na')]),
                'actions_required': len(df[(df['Category'] == category) & 
                                        (df['Action Required'].fillna('NA').str.lower() != 'na')])
            }
        
        # Timeline analysis (by month)
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Month'] = df['Timestamp'].dt.strftime('%Y-%m')
            monthly_counts = df['Month'].value_counts().sort_index()
            stats['timeline'] = monthly_counts.to_dict()
        except Exception as e:
            logger.warning(f"Error processing timeline data: {str(e)}")
            stats['timeline'] = {}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in analyze_chat_data: {str(e)}")
        return None

def display_analytics(df):
    """Display analytics in the Streamlit interface."""
    try:
        if df.empty:
            st.warning("No data available for analysis.")
            return
        
        stats = analyze_chat_data(df)
        if not stats:
            st.warning("Unable to analyze chat data. Please check if the sheet has the correct column names.")
            st.info("Required columns: Timestamp/Date, Name, Email, Context, Action Required/Needed, Headline")
            return
        
        # Overview metrics with more professional styling
        st.markdown("### 📊 Key Metrics Overview")
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric(
                "Total Conversations",
                f"{stats.get('total_chats', 0):,}",
                help="Total number of chat conversations processed"
            )
        with metrics_cols[1]:
            emails = stats.get('email_collection', {}).get('total_emails', 0)
            email_rate = (emails / stats.get('total_chats', 1)) * 100
            st.metric(
                "Email Collection Rate",
                f"{email_rate:.1f}%",
                help=f"Collected {emails:,} email addresses"
            )
        with metrics_cols[2]:
            actions = stats.get('action_required', {}).get('total', 0)
            action_rate = (actions / stats.get('total_chats', 1)) * 100
            st.metric(
                "Action Required Rate",
                f"{action_rate:.1f}%",
                help=f"{actions:,} conversations require follow-up"
            )
        
        # Category Analysis Section
        st.markdown("### 📑 Category Analysis")
        
        # Full-width table
        if stats.get('categories'):
            # Prepare data for the table
            table_data = []
            for category, data in stats['categories'].items():
                table_data.append({
                    "Category": category,
                    "Total Chats": f"{data['count']:,}",
                    "Distribution": f"{data['percentage']:.1f}%",
                    "Emails Collected": f"{data['emails_collected']:,}",
                    "Actions Required": f"{data['actions_required']:,}",
                    "Engagement Rate": f"{(data['emails_collected'] / data['count'] * 100):.1f}%" if data['count'] > 0 else "0%"
                })
            
            # Convert to DataFrame for better display
            table_df = pd.DataFrame(table_data)
            
            # Display the table with custom formatting
            st.markdown("#### Detailed Category Statistics")
            st.dataframe(
                table_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Total Chats": st.column_config.TextColumn("Total Chats", width="small"),
                    "Distribution": st.column_config.TextColumn("Distribution", width="small"),
                    "Emails Collected": st.column_config.TextColumn("Emails Collected", width="small"),
                    "Actions Required": st.column_config.TextColumn("Actions Required", width="small"),
                    "Engagement Rate": st.column_config.TextColumn("Engagement Rate", width="small"),
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Add some spacing
        st.markdown("---")
        
        # Centered pie chart with custom width
        if stats.get('categories'):
            st.markdown("#### Distribution of Conversations by Category")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:  # Center column for the pie chart
                categories_df = pd.DataFrame.from_dict(stats['categories'], orient='index')
                fig = px.pie(
                    values=categories_df['count'],
                    names=categories_df.index,
                    hole=0.4
                )
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=12
                )
                fig.update_layout(
                    showlegend=False,
                    height=500,  # Fixed height for better proportions
                    margin=dict(t=30, l=30, r=30, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Timeline Analysis
        if stats.get('timeline'):
            st.markdown("### 📈 Conversation Volume Trends")
            timeline_df = pd.DataFrame.from_dict(
                stats['timeline'], 
                orient='index', 
                columns=['count']
            ).reset_index()
            timeline_df.columns = ['Month', 'Number of Conversations']
            
            fig = px.line(
                timeline_df,
                x='Month',
                y='Number of Conversations',
                title="Monthly Conversation Volume"
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Conversations",
                hovermode='x unified',
                title_x=0.5,
                margin=dict(t=50, l=0, r=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        logger.error(f"Error in display_analytics: {str(e)}")
        st.error("An error occurred while displaying analytics. Please try refreshing the page.")

def main():
    # Initialize session state variables
    if 'clear_clicked' not in st.session_state:
        st.session_state.clear_clicked = False
    if 'chat_text' not in st.session_state:
        st.session_state.chat_text = ""
    if 'text_area_key' not in st.session_state:
        st.session_state.text_area_key = 'default'
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = ""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    st.title("YWCA Chat Processor")
    
    try:
        # Initialize Google Sheets service
        service = setup_google_sheets()
        create_or_get_sheet(service)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Process Single Chat", 
            "Batch Process JSON Files", 
            "View All Records",
            "Analytics Dashboard"
        ])
        
        with tab1:
            st.subheader("Process Single Chat")
            
            # Add option to either paste JSON or chat text
            input_type = st.radio("Select input type:", ["Chat Text", "JSON"])
            
            if input_type == "Chat Text":
                chat_text = st.text_area(
                    "Paste the chat conversation here:", 
                    value="" if st.session_state.clear_clicked else st.session_state.chat_text,
                    height=300,
                    key=st.session_state.get('text_area_key', 'default')
                )
            else:
                json_input = st.text_area(
                    "Paste the JSON chat data here:",
                    height=300,
                    key="json_input"
                )
                try:
                    if json_input:
                        chat_data = json.loads(json_input)
                        chat_text = process_json_chat(chat_data)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your input.")
                    chat_text = ""
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Process Chat"):
                    if chat_text:
                        with st.spinner("Processing chat..."):
                            analysis = analyze_chat_with_gpt(chat_text)
                            if analysis:
                                st.subheader("Analysis Results")
                                st.write("Headline:", analysis['headline'])
                                st.write("Name:", analysis.get('name', 'NA'))
                                st.write("Email:", analysis.get('email', 'NA'))
                                st.write("Context:", analysis['context'])
                                st.write("Action Required:", analysis['action_required'])
                                append_to_sheet(service, analysis)
                                st.success("Chat summary has been added to the Google Sheets document!")
                    else:
                        st.error("Please provide chat content to process.")
            
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.chat_text = ""
                    st.session_state.clear_clicked = True
                    st.session_state.text_area_key = datetime.now().isoformat()
                    st.rerun()
        
        with tab2:
            st.subheader("Batch Process JSON Files")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text_input("Selected Folder:", 
                             value=st.session_state.selected_folder,
                             key="folder_display",
                             disabled=True)
            
            with col2:
                if st.button("Select Folder", key="select_folder"):
                    folder_path = select_folder()
                    if folder_path:
                        st.session_state.selected_folder = folder_path
                        st.session_state.processing_complete = False
                        st.rerun()
            
            # Add Process Files button
            if st.session_state.selected_folder and not st.session_state.processing_complete:
                if st.button("Process Files", key="process_files"):
                    with st.spinner("Processing JSON files..."):
                        summary_data, error_files, processed_folder = process_json_files(
                            st.session_state.selected_folder, 
                            service
                        )
                        
                        if summary_data:
                            st.session_state.processing_complete = True
                            # Display summary in a nice format
                            st.success("Processing Complete!")
                            
                            # Create three columns for summary stats
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Files", summary_data['total_files'])
                                st.metric("Successfully Processed", summary_data['processed'])
                            
                            with col2:
                                st.metric("Names Found", summary_data['names_found'])
                                st.metric("Emails Found", summary_data['emails_found'])
                            
                            with col3:
                                st.metric("Actions Required", summary_data['actions_required'])
                                st.metric("Errors", summary_data['errors'])
                            
                            # Show processed files location
                            st.info(f"✅ Processed files moved to: {processed_folder}")
                            
                            # Show errors if any
                            if error_files:
                                with st.expander("View Error Details"):
                                    for file_path, error in error_files:
                                        st.error(f"❌ {os.path.basename(file_path)}: {error}")
            
            # Add New Folder button after processing is complete
            if st.session_state.processing_complete:
                if st.button("Process New Folder", key="new_folder"):
                    st.session_state.selected_folder = ""
                    st.session_state.processing_complete = False
                    st.rerun()

        with tab3:
            st.subheader("View All Records")
            if st.button("Refresh Data"):
                df = get_sheet_data(service)
                if not df.empty:
                    st.dataframe(df)
                else:
                    st.info("No records found in the sheet.")

        with tab4:
            st.title("📊 Chat Analytics Dashboard")
            
            # Get the data
            df = get_sheet_data(service)
            
            if not df.empty:
                display_analytics(df)
            else:
                st.warning("No data available for analysis. Please process some chats first.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again or contact support.")

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# - .env