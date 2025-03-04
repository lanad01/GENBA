################
### Library Import
################

### Built-in Modules
import os  # For operating system interactions
import re  # For regular expressions
import time  # For time-related functions
import warnings  # For managing warnings
from datetime import datetime  # For date and time handling
warnings.filterwarnings("ignore")  # Suppress warnings

### Third-party Libraries
import cv2  # For image processing
import numpy as np  # For numerical computations
import pandas as pd  # For handling data in tabular form
import pdfplumber  # For parsing PDF documents
import pymupdf4llm  # For working with PDFs in NLP
import torch  # For tensor computations and GPU support
from contextlib import redirect_stdout  # For redirecting stdout

# LangChain Libraries
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks

# PDF to Image
from pdf2image import convert_from_path  # For converting PDF pages to images
from pytz import timezone

### Local Libraries
from model import *  # Import local model utilities
from utils import *  # Import local utility functions

################
################
### Setting
################

### GPU Setting
os.environ["USER_AGENT"] = "myagent"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Paths
data_path = "../data"  # Path for manual documents, classification files, and vector DBs

### Document Classification File Load
WTG_doc_clf = pd.read_excel(f"{data_path}/WTG_doc_clf.xlsx")

### Embedding Model Creation
embedding = load_embedding(device)

### Vector DB Creation
# Error/Warning DBs
EW_100 = gen_db(embedding)  # 01.WINDS3000_91_100 model
EW_134 = gen_db(embedding)  # 02.WINDS3000_134 model
EW_140 = gen_db(embedding)  # 03.WINDS5560_140 model

# Manual/Drawing DBs
MD_100 = gen_db(embedding)  # 01.WINDS3000_91_100 model
MD_134 = gen_db(embedding)  # 02.WINDS3000_134 model
MD_140 = gen_db(embedding)  # 03.WINDS5560_140 model

### Text Splitter Creation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=500, 
    length_function=len, 
    is_separator_regex=False
)

### Sentence Transformer Model for Header/Footer Removal
model_ST = load_sentence_transformer(device)

### PDF Parsing Evaluation Prompt
prompt = load_prompt_evaluation_context()

################
### PDF Parsing Logic
################

### Load Prompt and Model
prompt = load_prompt_evaluation_context()  # Load prompt for PDF parsing evaluation
llm = load_llm()  # Load LLM
chain = create_chain_evaluation_context(prompt, llm)  # Create evaluation chain

### Error Code List for Model 03.WINDS5560_140
WINDS5560_error_list = []  # List to handle duplicate error codes for 03.WINDS5560_140 model

################
### Document Parsing & Vector DB Loading
################
for iter_num, doc_info in WTG_doc_clf.iterrows():
    # Log start time
    start_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[LOG] [{start_time}] Document Parsing ({iter_num + 1}/{len(WTG_doc_clf):,}) START")

    ################
    ### Document Information
    ################
    file_name = doc_info["file_name"]  # File name
    file_path = doc_info["file_path"]  # File path
    model_name = doc_info["model_name"]  # WTG model name
    file_ext = doc_info["file_ext"]  # File extension
    priority = doc_info["priority"]  # Reference document priority
    loading_type = doc_info["loading_type"]  # Loading type (parsing, link, exclude)
    doc_clf = doc_info["doc_clf"]  # Document classification
    note = doc_info["note"]  # Notes or remarks

    # Print document information for logging
    print(f"File Name: {file_name}")
    print(f"File Path: {file_path}")
    print(f"Model Name: {model_name}")
    print(f"Priority: {priority}")
    print(f"File Extension: {file_ext}")

    # Construct the full file path
    file_full_path = f"{data_path}/{file_path}/{file_name}"

    # Initialize data structures
    chunk_list = []
    page_list = []
    db_texts = []
    db_metadatas = []
    db_ids = []

    ################
    ### 1. Loading Type: Parsing
    ################
    if loading_type == "parsing":
        ### 1.1 Priority 1: Error & Warning List
        if priority == 1:
            # For Model 01.WINDS3000_91_100
            if model_name == "01.WINDS3000_91_100":
                text_list = []
                toc_list = []

                # Document Loader
                loader = PyMuPDFLoader(file_full_path)  # Load the document
                docs = loader.load()

                for doc in docs:
                    # Extract page content
                    text = doc.page_content
                    text_list.append(text)

                ################
                ### Remove Header & Footer
                ################
                header = find_header(text_list, ratio=0.99)
                footer = find_header([text[::-1] for text in text_list], ratio=0.99)

                # Remove header
                if header:
                    text_list = [re.sub(header, "", text) for text in text_list]

                # Remove footer
                if footer:
                    text_list = [
                        re.sub(footer[::-1], "", text[::-1])[::-1] for text in text_list
                    ]

                # Merge the entire document content
                docs_contents = "\n".join(text_list)

                ################
                ### TOC (Table of Contents) Extraction
                ################
                # Define separators for TOC
                toc_start_sep = "Index\n\n"
                toc_final_sep = "1 General\n"

                # Extract TOC string
                toc_str = docs_contents.split(toc_start_sep)[1].split(toc_final_sep)[0]
                toc_str = re.findall(r"\w+[.\d]*\s*.*\s*\.{5,}\s*\b*", toc_str)

                # Assign TOC and page numbers
                toc_list = []
                page_list = []

                for toc in toc_str:
                    toc, page_num = re.split(r"\s?[.]{5,}\s?", toc)
                    toc_list.append(toc.strip())
                    page_list.append(int(page_num))

                # Merge document content excluding TOC
                docs_contents = toc_list[0] + docs_contents.split(toc_list[0])[-1]

                ################
                ### Chunk Assignment
                ################
                chunk_list = []

                for idx in range(len(toc_list)):
                    # Split content based on current and next TOC entries
                    if idx != len(toc_list) - 1:
                        text = docs_contents.split(toc_list[idx])[-1].split(toc_list[idx + 1])[0]
                    else:
                        text = docs_contents.split(toc_list[idx])[-1]


                    chunk = ''
                    if text.startswith((" \nerror text", " \nwarning text")):
                        # Process error/warning text
                        chunk = "\n\n".join(
                            [val.strip().replace("\n", " ") for val in text.split("\n\n") if val.strip() != ""]
                        )
                    chunk = clean_text(chunk)  # Clean the chunk
                    chunk_list.append(chunk)

                # Remove empty chunks
                chunk_list, page_list, toc_list = [
                    chunk for chunk in chunk_list if len(chunk) > 0
                ], [
                    page for chunk, page in zip(chunk_list, toc_list) if len(chunk) > 0
                ]

                # Prepend TOC title to chunks
                chunk_list = [f"{toc}: {chunk}" for chunk, toc in zip(chunk_list, toc_list)]

                # Remove numbering and extra whitespace from TOC titles in chunks
                chunk_list = [re.sub(r"^\d+[.\d]*\s*", "", chunk) for chunk in chunk_list]

            ################
            ### Model-Specific Processing: 02.WINDS3000_134
            ################
            elif model_name == "02.WINDS3000_134":
                # Load the Excel file
                df = pd.read_excel(file)
                # Filter required columns
                df = df[["ERROR CODE", "ERROR 그룹명", "ERROR 명", "설명", "부품ID", "부품 ITEMPATH", "부품명"]]

                for _, row in df.iterrows():
                    # Combine row data into a chunk
                    chunk_list.append(", ".join([f"{index}: {value}" for index, value in zip(row.index, row.values) if value != ""]))

            ################
            ### Model-Specific Processing: 03.WINDS5560_140
            ################
            elif model_name == "03.WINDS5560_140":
                text_list = []
                toc_list = []

                # Load the document
                loader = PyMuPDFLoader(file)
                docs = loader.load()

                # Extract text content from pages
                for doc in docs:
                    text = doc.page_content
                    text_list.append(text)


                ################
                ### Header & Footer Removal
                ################
                # Extract text content from pages
                text_list = [doc.page_content for doc in docs]

                # Identify header and footer patterns
                header = find_header(text_list, 0.99)
                footer = find_header([text[::-1] for text in text_list], 0.99)

                # Remove header
                if header:
                    text_list = [re.sub(header, "", text) for text in text_list]

                # Remove footer
                if footer:
                    text_list = [re.sub(footer[::-1], "", text[::-1])[::-1] for text in text_list]

                # Merge the entire document content
                docs_contents = "\n".join(text_list)

                ################
                ### TOC (Table of Contents) Extraction
                ################
                # Extract TOC string from the document content
                toc_str = docs_contents.split("Index")[-1]
                toc_str = re.findall(r"status code no. \d*, \d* \d*", toc_str)


                for toc in toc_str:
                    toc_parts = toc.split("no. ")
                    toc = toc_parts[1].split(", ")[0]
                    page_num = toc_parts[1].split(", ")[2]
                    toc_list.append(toc.)
                    page_list.append(int(page_num))

                # Sort TOC and page numbers
                sorted_dict = sorted(zip(page_list, toc_list))
                page_list = [key for key, val in sorted_dict]
                toc_list = [val for key, val in sorted_dict]



                ################
                ### Error Code Processing
                ################
                for idx in range(len(toc_list)):
                    error_code = toc_list[idx]
                    page_num = page_list[idx]

                    # Skip duplicate error codes
                    if error_code in WINDS5560_error_list:
                        continue

                    # Skip specific error codes
                    if error_code == "10021":  # 페이지가 잘린 에러 코드 제외
                        continue

                    # Adjust page number for specific cases
                    if page_num > 958:  # Index IV 페이지 번호가 잘못된 경우 수정
                        page_list[idx] += 1
                        page_num += 1

                    # Extract text for the current page
                    text = text_list[page_num - 1]

                    ################
                    ### Extract Detailed Error Information
                    ################
                    try:
                        # Extract error details when Source, Reason, Remarks exist
                        error_text = re.findall(f"{error_code}[\\w\\W]*Remarks", text)[0].replace("View\nEdit\n", "")
                        error_title = "Error: " + error_text.split("\n")[0]
                        error_content = "\n".join(
                            [val for val in re.findall(r".*:.*", error_text) if not val.startswith(("Source", "Reason"))]
                        )
                        error_source = re.findall(r"Source: [\\w\\W]*\nReason", error_text)[0].split("\nReason")[0]
                        error_reason = re.findall(r"Reason: [\\w\\W]*\nRemarks", error_text)[0].split("\nRemarks")[0]

                        # Create chunk with error details
                        chunk = "\n".join([error_title, error_content, error_source, error_reason])
                        chunk = clean_text(chunk)

                    except Exception:
                        # Extract error details when Source, Reason, Remarks are absent
                        if idx < len(toc_list) - 1:
                            error_text = re.findall(f"{error_code}[\\w\\W]*", text)[0].split(toc_list[idx + 1])[0]
                        else:
                            error_text = re.findall(f"{error_code}[\\w\\W]*", text)[0]

                        error_title = "Error: " + error_text.split("\n")[0]
                        error_content = "\n".join(re.findall(r".*:.*", error_text))

                        # Create chunk with error details
                        chunk = "\n".join([error_title, error_content])
                        chunk = clean_text(chunk)

                    # Add the chunk to the list
                    chunk_list.append(chunk)
                    WINDS5560_error_list.append(error_code)


        ################
        ### 1.2 Priority 3 (Wiring Diagram)
        ################
        elif priority == 3:
            if model_name in ("01.WINDS3000_91_100", "02.WINDS3000_134"):
                # Document Load
                loader = PyMuPDFLoader(file)
                docs = loader.load()

                # Extract keywords from TOC
                kw_pattern = re.compile(r"\n\+\w.+\n")
                kw_list = []
                toc_last_page = 0  # Initialize TOC last page

                for doc_idx in range(len(docs)):
                    content = docs[doc_idx].page_content

                    if "structure identifier overview" in content.lower() and "list of contents" not in content:
                        toc_last_page = doc_idx  # Identify Table of Content Last Page
                        kw_list += [val.replace("+", "").strip() for val in kw_pattern.findall(content)]

                ################
                ### Chunking
                ################
                for doc_idx in range(len(docs)):
                    # Apply only to pages after the TOC
                    if doc_idx <= toc_last_page:
                        continue

                    # Extract content and keyword
                    content = docs[doc_idx].page_content
                    content = "\n".join(re.split(r"\d{2,4}-\d{2}-\d{2}", content)[:-1])  # Remove date-like patterns
                    kw = find_keyword(content, kw_list)

                    ################
                    ### Model-Specific Processing
                    ################
                    if model_name == "01.WINDS3000_91_100":
                        if kw and "subtitle" in content:
                            # Process content for the specific model
                            chunk = clean_text(content)
                            chunk_list.append(chunk)

                    elif model_name == "02.WINDS3000_134":

                        if kw and "Cover sheet" not in content:
                            content = content.split(kw)[-1].split("\n")[0]
                        else:
                            subtitle = re.findall(r"Date\n.+\n", content)[0]
                            subtitle = subtitle.replace("Date", "").strip()
                            continue

                    # Add subtitle and content to chunk
                    chunk = f"{subtitle}\n{content}"
                    chunk_list.append(chunk)
                    page_list.append(doc_idx + 1)

            # 03.WINDS5560_140
            elif model_name == "03.WINDS5560_140":
                # Document Load
                docs = pdfplumber.open(file)

                for page in docs.pages:
                    # Extract tables from the page
                    tb = page.extract_tables()

                    if tb != []:
                        df = pd.DataFrame(tb[0])  # Use the first table extracted

                        try:
                            # Extract specific values from the table
                            val1 = df.iloc[19][2].replace(" ", "")
                            val2 = df.iloc[19][3].replace(" ", "")
                            val3 = df.iloc[19][4].replace(" ", "")
                            val4 = df.iloc[19][3].replace(" ", "")
                            val5 = df.iloc[19][4].replace(" ", "")
                            title = df.iloc[7][2]
                            # 데이터 길이 조건 확인 및 제목 생성
                            if len(val1) == 8 and len(val2) == 9 and len(val3) == 8 and len(val4) == 8 and len(val5) == 4 and not title.lower().startswith('error'):
                                chunk_list.append(title)
                                page_list.append(page.page_number)

                        except:
                            continue


        ################
        ### 1.3 Priority: 2 (부품 매뉴얼 & 업체자료) & 4 (설치 매뉴얼 & 기타자료)
        ################
        elif priority in (2, 4):
            if file_ext in ("docx", "doc"):
                text_list = []

                ################
                ### Document Load
                ################
                if file_ext == "doc":
                    # Convert `.doc` to `.docx` for compatibility
                    docx_file_name = re.sub(r"[.]\w*$", "[CONVERT].docx", file_name)
                    docx_file_path = re.sub(r"^01.WTG", "01.WTG_CONVERT", file_path)
                    file = f"{data_path}/{docx_file_path}/{docx_file_name}"

                # Load document using UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(file, mode="elements", strategy="fast")
                docs = loader.load()

                ################
                ### Element-Level Text Processing
                ################
                for doc in [doc for doc in docs if doc.metadata.get("category") not in ("Header", "Footer", "PageBreak")]:
                    # Handle table-like elements
                    if "text_as_html" in doc.metadata:
                        if doc.page_content == '':
                            continue

                        table = pd.read_html(
                            doc.metadata["table_text_as_html"], header=0, encoding="utf-8"
                        )[0]

                        # Remove unnecessary columns
                        table = table[[col for col in table.columns if not col.startswith(("Unnamed", "검사 결과", "노트 사항"))]]

                        # Drop rows and columns with all missing values
                        table = table.dropna(axis=0, how="all")  # Drop rows where all values are NaN
                        table = table.dropna(axis=1, how="all")  # Drop columns where all values are NaN

                        # Replace NaN values with empty strings
                        table = table.fillna("")

                        # Skip tables where all column names are the same after removing suffix numbers
                        if len(set([re.sub(r"\.\d*$", "", col) for col in table.columns])) == 1:
                            continue

                        # Skip tables where all column names match the first value in the column
                        if any([col == table[col].unique()[0] for col in table.columns]):
                            continue

                        # Skip empty tables
                        if len(table) == 0:
                            continue

                        # Process each row of the table and add to `text_list`
                        for _, row in table.iterrows():
                            row_text = " ".join(
                                [f"{index}: {value}" for index, value in zip(row.index, row.values) if value != ""]
                            )
                            text_list.append(row_text)

                    # Process text-based elements and add to `text_list`
                    else:
                        text_list.append(doc.page_content)

                ################
                ### Merge and Chunk Text
                ################

                # Merge all text into a single string
                all_text = "\n".join([text.strip() for text in text_list])
                all_text = clean_text(all_text)

                # Split the merged text into chunks
                chunk_list = [chunk.page_content for chunk in text_splitter.create_documents([all_text])]

            ################
            ### PDF File Processing
            ################

            elif file_ext == "pdf":
                # Suppress PyMuPDF4LLM output
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    llama_reader = pymupdf4llm.LlamaMarkdownReader()
                    llama_docs = llama_reader.load_data(file)

                text_list_4llm = []
                page_list_4llm = []

                for page_num, doc in enumerate(llama_docs):
                    text = doc.text
                    text = clean_text(text)
                    text_list_4llm.append(text)
                    page_list_4llm.append(page_num)


                    ################
                    ### Remove Header & Footer
                    ################

                    header = find_header(text_list_4llm)
                    footer = find_header([text[::-1] for text in text_list_4llm])

                    if header:
                        text_list_4llm = [re.sub(header, "", text) for text in text_list_4llm]
                    if footer:
                        text_list_4llm = [re.sub(footer, "", text[::-1])[::-1] for text in text_list_4llm]

                    text_list = []
                    page_list = []
                    ################
                    ### PDF to Image Conversion
                    ################

                    dpi = 144
                    images = convert_from_path(file, dpi=dpi)

                    ################
                    ### Document Load and Header/Footer Processing
                    ################

                    docs = pdfplumber.open(file)
                    text_list_header = []

                    for page in docs.pages:
                        text_header = page.extract_text()
                        text_list_header.append(text_header)

                    header = find_header(text_list_header)
                    footer = find_header([text[::-1] for text in text_list_header])

                    ################
                    ### PDF Processing with Plumber
                    ################

                    with pdfplumber.open(file) as pdf:
                        text_list = []
                        total_bboxes = []

                        for page_num, image in enumerate(images):
                            contours = detect_paragraphs(image)
                            pdf_page = pdf.pages[page_num]
                            img_width, img_height = image.size

                            # 문단 처리
                            text_list_tmp = []
                            text_list_tmp2 = []
                            box_list = []

                            for contour in contours:
                                # 이미지 좌표의 문단 사각형
                                x, y, w, h = cv2.boundingRect(contour)

                                # 이미지 좌표 > PDF 좌표 변환
                                pdf_box = convert_box_to_pdf_coordinates(
                                    (img_width, img_height),
                                    (pdf_page.width, pdf_page.height),
                                    (x, y, w, h),
                                )

                                # PDF 좌표의 문단 사각형
                                x0, y0, x1, y1 = pdf_box
                                y0 = pdf_page.height - y0
                                y1 = pdf_page.height - y1

                                bbox_info = pdf_page.bbox
                                a1, a2, a3, a4 = bbox_info
                                x0 = x0 + a1
                                y0 = y0 + a2
                                x1 = x1 + a1
                                y1 = y1 + a2

                                if y0 > y1:
                                    pdf_box = (x0, y1, x1, y0)
                                else:
                                    pdf_box = (x0, y0, x1, y1)

                                box_list.append(pdf_box)

                                page = pdf_page

                                if page.within_bbox(pdf_box).find_tables():
                                    tables = page.extract_tables()
                                    try:
                                        ts = {
                                            "vertical_strategy": "explicit",
                                            "horizontal_strategy": "explicit",
                                            "explicit_vertical_lines": curves_to_edges(page.curves + page.edges),
                                            "explicit_horizontal_lines": curves_to_edges(page.curves + page.edges),
                                            "intersection_y_tolerance": 10,
                                        }
                                    except:
                                        ts = {
                                            "vertical_strategy": "explicit",
                                            "horizontal_strategy": "explicit",
                                            "explicit_vertical_lines": curves_to_edges(page.curves + page.rect_edges),
                                            "explicit_horizontal_lines": curves_to_edges(page.curves + page.rect_edges),
                                            "intersection_y_tolerance": 10,
                                        }

                                    bboxes = [
                                        table.bbox
                                        for table in page.find_tables(table_settings=ts)
                                        if table.bbox not in total_bboxes
                                    ]
                                    total_bboxes.extend(bboxes)

                                    # Extract words not within table bounding boxes
                                    word_list = []
                                    for word in page.within_bbox(pdf_box).extract_words():
                                        word_bbox = {
                                            "x0": word["x0"],
                                            "top": word["top"],
                                            "x1": word["x1"],
                                            "x1": word["x1"],
                                            "bottom": word["bottom"]
                                        }

                                        if any(is_word_in_bbox(word_bbox, bbox) for bidx, bbox in enumerate(bboxes, 1)):
                                            for bidx, bbox in enumerate(bboxes, 1):
                                                if is_word_in_bbox(word_bbox, bbox) and f"[SEP{bidx}]" not in word_list:
                                                    word_list.append(f"[SEP{bidx}]")
                                        else:
                                            word_list.append(word["text"])

                                    # Process words within table bounding boxes and append to word_list
                                    table2text_dict = {}
                                    for idx, table in enumerate(tables, 1):
                                        table = pd.DataFrame(table[1:], columns=table[0])
                                        table = table.dropna(axis=0, how="all")  # Drop rows with all NaN
                                        table = table.dropna(axis=1, how="all")  # Drop columns with all NaN
                                        table = table.fillna("")  # Replace NaN with empty string

                                        # Skip invalid tables
                                        if None in table.columns:
                                            continue
                                        if len(set([re.sub(r"\.\d*$", "", col) for col in table.columns])) == 1:
                                            continue
                                        if any(
                                            col == np.unique(table[col].values)[0] for col in table.columns if len(table) > 0
                                        ):
                                            continue
                                        if len(table) == 0:
                                            continue

                                        # Convert table rows to text
                                        table2text = "\n".join(
                                            [
                                                f"{index}: {value}"
                                                for _, row in table.iterrows()
                                                for index, value in zip(row.index, row.values)
                                                if value != ""
                                            ]
                                        )
                                        table2text_dict[f"[SEP{idx}]"] = table2text

                                    # Combine words and table texts
                                    if table2text_dict:
                                        text = "\n".join(
                                            [table2text_dict.get(word, word) for word in word_list]
                                        )
                                    else:
                                        text = "\n".join(word_list)
                                    text = re.sub(r"\[SEP\d*\]", "", text)

                                # 페이지에 테이블이 존재하지 않는 경우
                                else:
                                    text = page.within_bbox(pdf_box).extract_text()
                                text = clean_text(text)
                                text_list_tmp.append(text)

                            # 1. x0 오름차순 정렬
                            sorted_data1_with_indices = sorted(
                                enumerate(box_list), key=lambda x: x[1][0]
                            )  # (index, value)
                            sorted_data1 = [x[1] for x in sorted_data1_with_indices]
                            sorted_indices = [x[0] for x in sorted_data1_with_indices]

                            # data2를 같은 인덱스 순서로 정렬
                            sorted_data2 = [text_list_tmp[i] for i in sorted_indices]

                            # 2. x0 유사성에 따라 그룹화
                            threshold = 10
                            grouped_indices = group_x0(sorted_data1, threshold)

                            # 3. 각 그룹 내에서 y0 내림차순 정렬
                            final_sorted_indices = []
                            for group in grouped_indices:
                                group_sorted = sorted(group, key=lambda idx: sorted_data1[idx][1], reverse=True)  # y0 내림차순
                                final_sorted_indices.extend(group_sorted)

                            # 최종 데이터 정렬
                            final_data1 = [sorted_data1[i] for i in final_sorted_indices]
                            final_data2 = [sorted_data2[i] for i in final_sorted_indices]
                            tmp_text = "\n".join([text for text in final_data2 if text.strip()])


                            tmp_text = [text for text in final_data2 if text.strip()]

                            if len(tmp_text) > 0:
                                if len(header) > 0:
                                    tmp_text = remove_similar_texts(header, tmp_text, threshold=0.6, model=model_ST, device=device)

                            if len(tmp_text) > 0:
                                if len(footer[::-1]) > 0:
                                    result = remove_similar_texts(footer[::-1], tmp_text, threshold=0.6, model=model_ST, device=device)
                                else:
                                    result = tmp_text

                            text2 = " ".join(result)
                            
                            text_list.append(text2)
                            page_list.append(page.page_number)

                    text_list_fin = []

                    for i in range(len(text_list_4llm)):
                        answer = chain.invoke({"context1": text_list_4llm[i], "context2": text_list[i]})
                        clean_answer = clean_text(answer)

                        if len(clean_answer) < 10:
                            text_list_fin.append(text_list_4llm[i])
                        else:
                            text_list_fin.append(clean_answer)

                    chunk_list, page_list = page_chunking(text_list_fin, page_list)

    ##########
    ### 2. Loading type Link
    ##########
    elif loading_type == "link":
        pass

    ##########
    ### 3. Loading type Exclude
    ##########
    elif loading_type == "exclude":
        continue

    ##########
    ### Convert to VectorDB Format
    ##########
    meta_dic = {
        "file_name": file_name,
        "file_path": file_path if note != "dup" else ", ".join(
            [f for f in WTG_doc_clf[WTG_doc_clf["file_name"].str.lower() == file_name.lower()]["file_path"].unique()]
        ),
        "file_ext": file_ext,
        "model_name": model_name,
        "priority": priority,
    }

    if page_list:
        db_texts = chunk_list.copy()
        db_metadatas = [
            dict(meta_dic, **{"page_num": page_list[i]}) for i in range(len(chunk_list))
        ]
        db_ids = [f"{file_path}/{file_name}" for _ in range(len(chunk_list))]
    elif loading_type == "parsing":
        db_texts = chunk_list.copy()
        db_metadatas = [meta_dic for _ in range(len(chunk_list))]
        db_ids = [f"{file_path}/{file_name}_{i}" for i in range(len(chunk_list))]

    elif loading_type == "link":
        db_texts = [f"{file_path}/{file_name}"]
        db_metadatas = [meta_dic.copy()]
        db_ids = [f"{file_path}/{file_name}"]

    ##########
    ### Data Insertion
    ##########
    if len(db_ids):
        if priority == 1:
            if model_name == "01.WINDS3000_91_100":
                _ = EW_100.add_texts(
                    texts=db_texts,
                    metadatas=db_metadatas,
                    ids=db_ids,
                )
            elif model_name == "02.WINDS3000_134":
                _ = EW_134.add_texts(
                    texts=db_texts,
                    metadatas=db_metadatas,
                    ids=db_ids,
                )


            elif model_name == "03.WINDS5560_140":
                _ = EW_140.add_texts(
                    texts=db_texts,
                    metadatas=db_metadatas,
                    ids=db_ids,
                )
        else:
            if model_name == "01.WINDS3000_91_100":
                _ = MD_100.add_texts(
                    texts=db_texts,
                    metadatas=db_metadatas,
                    ids=db_ids,
                )
            elif model_name == "02.WINDS3000_134":
                _ = MD_134.add_texts(
                    texts=db_texts,
                    metadatas=db_metadatas,
                    ids=db_ids,
                )
            elif model_name == "03.WINDS5560_140":
                _ = MD_140.add_texts(
                    texts=db_texts,
                    metadatas=db_metadatas,
                    ids=db_ids,
                )

    ##########
    ### Save DB
    ##########

    EW_100.save_local(
        folder_path=f"{data_path}/db_faiss",
        index_name="EW_100"
    )
    EW_134.save_local(
        folder_path=f"{data_path}/db_faiss",
        index_name="EW_134"
    )
    EW_140.save_local(
        folder_path=f"{data_path}/db_faiss",
        index_name="EW_140"
    )
    MD_100.save_local(
        folder_path=f"{data_path}/db_faiss",
        index_name="MD_100"
    )
    MD_134.save_local(
        folder_path=f"{data_path}/db_faiss",
        index_name="MD_134"
    )
    MD_140.save_local(
        folder_path=f"{data_path}/db_faiss",
        index_name="MD_140"
    )

    finish_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    total_time = str(
        datetime.strptime(finish_time, "%Y-%m-%d %H:%M:%S")
        - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    )

    print(f"[LOG] [{start_time}] Document Parsing ({iter_num+1}/{len(WTG_doc_clf):,}) FINISH / Duration of Time: {total_time}")
