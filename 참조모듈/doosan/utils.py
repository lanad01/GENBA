##################################################
### Library Import
##################################################

### Built-in Modules
import os
import re
import difflib
from collections import Counter
from datetime import datetime
from IPython.display import Markdown, display

### Third-party Library
import cv2
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from langchain.load import dumps, loads
from langchain.core.messages import ChatMessage
from pytz import timezone
from sentence_transformers import util
from IPython.display import Markdown, display

##################################################
### Doc Parsing
##################################################
def clean_text(text):
    """
    Clean and preprocess the input text by removing unwanted characters,
    extra spaces, and other artifacts.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned and processed text.
    """

    # Replace specific unwanted patterns
    text = text.replace(' (부품-주거-순서)', '')
    text = text.replace(' (경고 메시지 포함)', '')

    # Remove specific special Unicode characters
    text = re.sub(r'\uf06c' , ' '  , text) # Special character
    text = re.sub(r'\uf0d8' , ' '  , text) # Special character
    text = re.sub(r'\uf0fc' , ' '  , text) # Special character
    text = re.sub(r'\uf0d3' , ' '  , text) # Special character
    text = re.sub(r'\ufea0' , ' '  , text) # Special character
    text = re.sub(r'\uf0b0' , ' '  , text) # Special character
    text = re.sub(r'\ufffd' , ' '  , text) # Special character

    # Remove cid patterns (e.g., cid:nnn)
    text = re.sub(r'cid:\d+', ' '  , text)

    # Normalize spaces and punctuation
    text = re.sub(r'\*'     , ' '  , text)
    text = re.sub(r'\.{3,}' , '...', text)
    text = re.sub(r'-{3,}'  , ' '  , text)
    text = re.sub(r'\|{10,}', ' '  , text)
    text = re.sub(r'\s+'    , ' '  , text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def find_keyword(text, kw_list):
    """
    Find and return the first keyword from a given keyword list
    that appears in the input text.

    Args:
        text    (str) : The input text to search.
        kw_list (list): A list of keywords to search for.

    Returns:
        str: The first matching keyword found in the text (case-insensitive).
              Returns None if no keyword is found.
    """

    # Iterate over the keyword list
    for idx, kw in enumerate(kw_list):
        # Check if the keyword exists in the text (case-insensitive)
        if kw.lower() in text.lower():
            return kw  # Return the matching keyword

    # Return None if no keywords are found

    return None

def find_header(text_list, ratio=0.5):
    """
    Identify and return the header text from a list of text lines
    based on repetitive patterns in the beginning of the lines.

    Args:
        text_list (list) : A list of text strings (e.g., from PDF pages).
        ratio     (float): Threshold ratio for determining headers.
                           Default is 0.5 (50% of the text lines must start the same way).

    Returns:
        str: Identified header text, if any.
    """

    header = ""  # Initialize header text

    while True:
        # Remove empty strings from the text list
        text_list = [text for text in text_list if text != ""]

        # If the text list has 1 or fewer elements, stop processing
        if len(text_list) <= 1:
            break

        # Count the occurrences of the first character/word of each text
        cnt = Counter([text[0] for text in text_list])
        most_val = cnt.most_common(1)[0][0]  # Most common starting value
        most_cnt = cnt.most_common(1)[0][1]  # Frequency of the most common value
        digit_cnt = sum([text[0].isdigit() for text in text_list])  # Count lines starting with a digit

        # If the most common starting value exceeds the ratio threshold
        if most_cnt / len(text_list) > ratio:
            # Remove the common starting value from all lines
            text_list = [text[1:] if text[0] == most_val else text for text in text_list]
            header += most_val  # Append the most common value to the header

        # If the starting value is numeric and exceeds the ratio threshold
        elif digit_cnt / len(text_list) > ratio:
            # Remove leading digits from all lines
            text_list = [re.sub(r'^\d+', '', text) if text[0].isdigit() else text for text in text_list]
            header += r'\d*'  # Append a placeholder for numeric patterns to the header

        # If no conditions are met, exit the loop
        else:
            break

    # Strip whitespace and handle cases where the header may not be valid
    if header.strip():
        header = header
    else:
        header = re.escape(header).replace(r'\\\\d\\*', r'\d*')  # Escape special characters in header

    return header

def page_chunking(text_list, page_list, chunking_num=2, overlap_num=1):
    """
    Create text chunks from a list of pages with overlap.

    This function divides a PDF file's pages into chunks of size `chunking_num`,
    with `overlap_num` pages overlapping between consecutive chunks.

    Args:
        text_list    (list): A list of text strings, one for each page.
        page_list    (list): A list of corresponding page numbers.
        chunking_num (int) : Number of pages in each chunk.
        overlap_num  (int) : Number of overlapping pages between chunks.

    Returns:
        list: A list of tuples, where each tuple contains:
              - Chunked text (str)
              - List of page numbers in the chunk
    """

    chunks = []  # List to store the resulting chunks
    total_pages = len(text_list)

    # Iterate through the pages to create chunks
    for start in range(0, total_pages, chunking_num - overlap_num):
        # Determine the end of the current chunk
        end = min(start + chunking_num, total_pages)

        # Extract text and page numbers for the current chunk
        chunk_text = " ".join(text_list[start:end])
        chunk_pages = page_list[start:end]

        # Append the chunk to the result
        chunks.append((chunk_text, chunk_pages))

        # Break if the end of the text list is reached
        if end == total_pages:
            break

    return chunks

def page_chunking(text_list, page_list, chunking_num=2, overlap_num=1):
    """
    Divide a PDF's text into chunks with overlap.

    This function divides a list of text pages into chunks of size `chunking_num`,
    with `overlap_num` pages overlapping between consecutive chunks.

    Args:
        text_list    (list): A list of text strings, one for each page.
        page_list    (list): A list of corresponding page numbers.
        chunking_num (int) : Number of pages per chunk.
        overlap_num  (int) : Number of overlapping pages between consecutive chunks.

    Returns:
        tuple: Two lists:
            - chunk_list: A list of chunked text strings.
            - new_page_list: A list of page numbers corresponding to each chunk.
    """

    # Initialize the results
    chunk_list = []
    new_page_list = []

    # Case 1: If text_list has 1 or fewer elements, return it as-is
    if len(text_list) <= 1:
        chunk_list = text_list.copy()
        new_page_list = page_list.copy()

    # Case 2: If text_list length is less than or equal to chunking_num, treat the entire text as one chunk
    elif len(text_list) <= chunking_num:
        chunk = " ".join(text_list)  # Merge all text into a single chunk
        chunk_list = text_list.copy()
        new_page_list = page_list.copy()

    # Case 3: Chunking with overlap
    else:
        idx = 0
        while 1:
            # Create the chunk by joining text in the range [idx, end_idx)
            chunk = " ".join(text_list[idx:idx+chunking_num])
            chunk_list.append(chunk)
            new_page_list.append(page_list[idx])
            if idx + chunking_num >= len(text_list):
                break
            idx += chunking_num - overlap_num  # Move by chunking_num - overlap_num to create overlapping chunks

    return chunk_list, new_page_list

def curves_to_edges(cs):
    """
    Convert curves to their corresponding edge coordinates.

    Args:
        cs (list): A list of curve objects.

    Returns:
        list: A list of edge coordinates for each curve.
    """

    edges = []
    for c in cs:
        # Convert each curve into its rectangle edges using pdfplumber utilities
        edges += pdfplumber.utils.rect_to_edges(c)

    return edges

def is_word_in_bbox(word_bbox, bbox):
    """
    Check if a word's bounding box is inside a table's bounding box.

    Args:
        word_bbox (dict): A dictionary with keys 'x0', 'top', 'x1', 'bottom' representing the word's bounding box.
        bbox (tuple): A tuple (bx0, by0, bx1, by1) representing the table's bounding box.

    Returns:
        bool: True if the word's bounding box is entirely inside the table's bounding box, False otherwise.
    """

    x0, y0, x1, y1 = word_bbox['x0'], word_bbox['top'], word_bbox['x1'], word_bbox['bottom']
    bx0, by0, bx1, by1 = bbox

    # Check if the word's bounding box lies entirely within the table's bounding box
    return x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1

def detect_paragraphs(image):
    """
    Detect paragraphs in an image and return their bounding box coordinates.

    Args:
        image (PIL.Image): An image (converted from a PDF page).

    Returns:
        list: A list of bounding box coordinates representing detected paragraphs.
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Perform morphological operations to enhance paragraph detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    dilated = cv2.dilate(binary, kernel, iterations=9)

    # Extract contours from the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def convert_box_to_pdf_coordinates(image_size, pdf_page_size, box):
    """
    Convert detected object bounding box coordinates from image space to PDF space.

    Note:
        PDF coordinates are flipped vertically compared to image coordinates.
        This function accounts for this difference by inverting the y-axis.

    Args:
        image_size    (tuple): The width and height of the image (e.g., as processed by OpenCV).
        pdf_page_size (tuple): The width and height of the actual PDF file.
        box           (tuple): The bounding box coordinates in the image space (x, y, width, height).

    Returns:
        tuple: Transformed bounding box coordinates in PDF space (x_min, y_min, x_max, y_max).
    """

    # Extract image and PDF dimensions
    img_width, img_height = image_size
    pdf_width, pdf_height = pdf_page_size

    # Extract bounding box coordinates
    x, y, w, h = box

    # Convert image coordinates to PDF coordinates
    x_min = (x / img_width) * pdf_width
    y_min = (y / img_height) * pdf_height
    x_max = ((x + w) / img_width) * pdf_width
    y_max = ((y + h) / img_height) * pdf_height

    # Adjust for the PDF coordinate system (invert the y-axis)
    return (x_min, pdf_height - y_max, x_max, pdf_height - y_min)

def are_similar(x1, x2, threshold=10):
    """
    Determine whether two values are similar based on a threshold.

    Args:
        x1        (float): The first value.
        x2        (float): The second value.
        threshold (float): The maximum allowed difference between the two values (default: 10).

    Returns:
        bool: True if the absolute difference between x1 and x2 is less than or equal to the threshold, False otherwise.
    """

    return abs(x1 - x2) <= threshold

def group_x0(data, threshold=10):
    """
    Group coordinates based on the x0 value to sort a multi-column document.

    The grouping helps in parsing the document in a top-left to bottom-left,
    then top-right to bottom-right order based on x0 and y0 coordinates.

    Args:
        data      (list): A list of coordinates or items, where each item is expected
                          to have the x0 value at index 0.
        threshold (int) : The threshold for grouping x0 coordinates.
                          Larger values result in broader groupings.

    Returns:
        list: A list of grouped indices based on the x0 values.
    """

    grouped_indices = []  # To store groups of indices
    current_group = []  # Current group being processed

    # Iterate through the data to group indices
    for idx, item in enumerate(data):
        # If the current group is empty or the current item's x0 is similar to the last group's x0
        if not current_group or are_similar(item[0], data[current_group[-1]][0], threshold):
            current_group.append(idx)  # Add to the current group
        else:
            # Save the current group and start a new group
            grouped_indices.append(current_group)
            current_group = [idx]

    # Append the last group if it exists
    if current_group:
        grouped_indices.append(current_group)

    return grouped_indices

def remove_similar_texts(input_text, text_list, threshold=0.6, model=None, device=None):
    """
    Remove texts from a list that have a cosine similarity above a given threshold with the input text.

    Args:
        input_text (str)                : The input text to compare.
        text_list  (list)               : A list of texts to filter based on similarity.
        threshold  (float)              : Similarity threshold (default: 0.6). Texts with similarity above this value will be removed.
        model      (SentenceTransformer): A preloaded SentenceTransformer model for encoding text.
        device     (str)                : The device to use for computations ('cuda' or 'cpu').

    Returns:
        list: A list of texts with low similarity to the input text.
    """

    # Encode the input text and the text list into embeddings
    input_embedding = model.encode(input_text, convert_to_tensor=True).to(device)
    list_embeddings = model.encode(text_list, convert_to_tensor=True).to(device)

    # Compute cosine similarity between input text and the text list
    cosine_scores = util.cos_sim(input_embedding, list_embeddings).squeeze(0)

    # Filter texts based on similarity threshold
    filtered_texts = [
        text for idx, text in enumerate(text_list)
        if cosine_scores[idx] < threshold
    ]

    return filtered_texts

##################################################
### Streamlit
##################################################

def now_time():
    """
    Get the current time in Korea Standard Time (KST).

    Returns:
        str: The current time formatted as "YYYY-MM-DD HH:MM:SS".
    """

    return datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

def print_history():
    """
    Display the previous query/response history in the Streamlit app.

    This function iterates through the session's stored messages and their types,
    and displays them accordingly (e.g., text or table).
    """

    for message, message_type in zip(st.session_state["messages"], st.session_state["message_type"]):
        # Check the type of the message
        if isinstance(message, str):
            if message_type == "text":
                st.write(message)
        else:
            if message_type == "text":
                st.chat_message(message.role).write(message.content)
            elif message_type == "table":
                # Convert markdown table to DataFrame and display it
                df = markdown_to_dataframe(message).reset_index(drop=True)
                st.table(df)

def add_history(role, content, message_type):
    """
    Add a message to the current message history.

    Args:
        role         (str): The sender of the message (e.g., "user", "assistant").
        content      (str): The content of the message.
        message_type (str): The type of the message ("text" or "table").
    """

    # Append the new message to the session state
    if role:
        st.session_state["messages"].append(ChatMessage(role=role, content=content))
    else:
        st.session_state["messages"].append(content)
    st.session_state["message_type"].append(message_type)

def save_log(key, value):
    """
    Save a key-value pair to the log in session state.

    Args:
        key (str): The key for the log dictionary.
        value (Any): The value to store in the log dictionary.
    """

    st.session_state["log"][key] = value

def load_queue():
    """
    Load the queue from a pickle file.

    Returns:
        pd.DataFrame: The loaded queue as a Pandas DataFrame.
    """

    return pd.read_pickle('queue.pkl')

def add_queue():
    """
    Add a new entry to the queue and save it to the queue file.

    Returns:
        int: The ID of the new queue entry.
    """

    queue = load_queue()
    new_id = len(queue)  # Generate a new ID
    new_queue = pd.DataFrame([{'id': new_id, 'status': 'wait'}])  # New queue entry
    pd.concat([queue, new_queue]).to_pickle('queue.pkl')  # Save updated queue to file

    return new_id

def change_status(id, status):
    """
    Change the status of a specific queue entry.

    Args:
        id (int): The ID of the queue entry to update.
        status (str): The new status to set.
    """
    queue = load_queue()
    queue.loc[queue['id'] == id, 'status'] = status  # Update the status
    queue.to_pickle('queue.pkl')  # Save updated queue to file

def split_text_and_table(text):
    """
    Separate plain text and table text from a given input.

    Args:
        text (str): The input text containing both plain text and table text.

    Returns:
        tuple: A tuple containing:
            - plain_text (str): The plain text part.
            - table_text (str): The table text part.
    """

    plain_text_lines = []
    table_text_lines = []
    is_table = False

    # Process each line
    for line in text.strip().splitlines():
        if is_table and "---" in line:
            continue
        elif line.startswith('|'):
            is_table = True
            table_text_lines.append(line)
        else:
            plain_text_lines.append(line)

    # Combine lines into single strings
    plain_text = "\n".join(plain_text_lines).strip()
    table_text = "\n".join(table_text_lines).strip()

    return plain_text, table_text

def markdown_to_dataframe(markdown_text, n=1):
    """
    Convert a markdown table into a Pandas DataFrame.

    Args:
        markdown_text (str): Text containing a markdown-style table.
        n (int): Row number to start indexing from (default: 1).

    Returns:
        pd.DataFrame: A DataFrame representing the markdown table.
    """

    # Split the markdown text into lines
    lines = markdown_text.strip().splitlines()

    # Extract headers
    headers = [col.strip() for col in lines[0].strip().split('|')[1:-1]]

    # Extract rows
    rows = []
    for line in lines[n:]:  # Skip header and separator lines
        split_line = [val.strip() for val in line.strip().split('|')[1:-1]]
        if len(split_line) == len(headers):
            rows.append(split_line)
        else:
            # Handle rows with missing or extra columns
            new_split_line = [val for val in split_line if val]
            if len(new_split_line) == len(headers):
                rows.append(new_split_line)

    return pd.DataFrame(rows, columns=headers)

def save_evaluation_to_excel(file_name, new_eval):
    """
    Save feedback or evaluation data to an Excel file.

    Args:
        file_name (str): The name of the Excel file to save the evaluation data.
        new_eval (dict): The evaluation data to be saved.
    """

    # Check if the file already exists
    if os.path.exists(file_name):
        # Load existing data from the Excel file
        df = pd.read_excel(file_name)
    else:
        # Create a new DataFrame with default columns if the file doesn't exist
        df = pd.DataFrame(columns=["No", "WTG_model", "question", "question_time", "select_answer", "answer", "score", "ref_doc", "true_answer"])

    # Assign a new entry number
    new_entry_no = len(df) + 1
    new_eval["No"] = new_entry_no

    # Append the new evaluation data
    df = pd.concat([df, pd.DataFrame([new_eval], columns=df.columns)], ignore_index=True)

    # Save the updated DataFrame back to the Excel file
    df.to_excel(file_name, index=False)

##################################################
### LLM
##################################################
def get_unique_union(documents: list) -> list:
    """
    Remove duplicate documents and return a unique list of documents.

    Args:
        documents (list): A list of lists containing documents.

    Returns:
        list: A list of unique documents.
    """

    # Flatten the nested list and serialize each Document object to a string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]

    # Remove duplicates by converting to a set, then back to a list
    unique_docs = list(set(flattened_docs))

    # Deserialize the unique document strings back into Document objects
    return [loads(doc) for doc in unique_docs]

def make_ew_content(ew_info: str, WTG_model: str, db) -> str:
    """
    Generate markdown-formatted content for Error/Warning-related queries.

    Args:
        ew_info   (str)  : Error/Warning information.
        WTG_model (str)  : Wind turbine generator (WTG) model name.
        db        (FAISS): Vector Store (Vector Database) to search for documents.

    Returns:
        str: Markdown-formatted content containing error/warning information.
    """

    # Initialize content with a heading
    content = "질문 내용과 관련된 에러/경고의 정보입니다.\n"

    # Retrieve metadata for the relevant document
    ew_metadata = [
        v.metadata
        for k, v in db.docstore._dict.items()
        if v.page_content == ew_info
    ][0]

    # Add metadata information to the content
    content += f"\n에러/경고 관련 문서: {ew_metadata['file_path']}/{ew_metadata['file_name']}\n\n"

    # Define error/warning lists based on WTG_model
    if WTG_model in ('01.WINDS3000_91_100', '02.WINDS3000_134'):
        if ew_info.startswith('Error'):
            ew_list = [
                "Error text",
                "Error stop class",
                "Time to first auto error reset",
                "Auto error call"
            ]
        elif ew_info.startswith('Warning'):
            ew_list = [
                "Warning text",
                "Description"
            ]
        else:
            ew_list = [
                "ERROR CODE",
                "ERROR 그룹명",
                "ERROR 명",
                "설명",
                "부품ID",
                "부품 ITEMPATH",
                "부품명"
            ]

        for idx in range(len(ew_list)):
            try:
                pattern = re.compile(
                    f'{ew_list[idx]}[\\w\\W]*?{ew_list[idx + 1]}',
                    re.IGNORECASE
                )
                match = re.search(pattern, ew_info)
                if match:
                    content += f"{match.group(0).split(',')[0]}\n"
            except IndexError:  # Handle the last item
                pattern = re.compile(
                    f'{ew_list[idx]}[\\w\\W]*',
                    re.IGNORECASE
                )
                match = re.search(pattern, ew_info)
                if match:
                    content += f"{match.group(0)}\n"

    elif WTG_model == '03.WINDS5560_140':  # Specific handling for model 03.WINDS5560_140
        if "Source:" in ew_info:
            source_info = ew_info.split("Source:")[1].split("Reason:")[0].strip()
            content += f"Source: {source_info}\n"
        if "Reason:" in ew_info:
            reason_info = ew_info.split("Reason:")[1].strip()
            content += f"Reason: {reason_info}\n"

    return content

def printmd(text: str):
    """
    Display text in Markdown format.

    Args:
        text (str): The text to be displayed in Markdown format.
    """

    display(Markdown(text))