import io
import base64
import logging
import concurrent.futures

import fitz
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_PROMPT = """
Extract the full markdown text from the given image, following these guidelines:
- Respond only with markdown, no additional commentary.  
- Capture all the text, respecting titles, headers, subheaders, equations, etc.
- If there are tables in this page, convert each one into markdown table format and include it in the response.
- If there are images, provide a brief description of what is shown in each image, and include it in the response.
- if there are charts, for each chart include a markdown table with the data represents the chart, a column for each of the variables of the cart and the relevant estimated values
          
"""

def image_to_markdown(file_object, client, model="gpt-4o",  prompt = DEFAULT_PROMPT):
    """
    Process a single image file and convert its content to markdown using OpenAI's API.

    Args:
        file_object (io.BytesIO): The image file object.
        client (OpenAI): The OpenAI client instance.
        model (str, optional): by default is gpt-4o
        prompt (str, optional): The prompt to send to the API. Defaults to DEFAULT_PROMPT.

    Returns:
        str: The markdown representation of the image content, or None if an error occurs.
    """
    # Log that we're about to process a page
    logging.info("About to process a page")

    base64_image = base64.b64encode(file_object.read()).decode('utf-8')

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        # Extract the markdown content from the response
        markdown_content = response.choices[0].message.content
        logging.info("Page processed successfully")
        return markdown_content

    except Exception as e:
        logging.error(f"An error occurred while processing the image: {e}")
        return None


def is_visual_page(page):
    """
    Check if a page contains visual content.

    Args:
        page (fitz.Page): The page object to check.

    Returns:
        bool: True if the page contains visual content, False otherwise.
    """
    has_images = bool(page.get_images(full=True))
    has_drawings = len(page.get_drawings()) > 20
    has_text = bool(page.get_text().strip())

    return has_images or has_drawings or not has_text


def page_to_image(page):
    """
    Convert a page of a PDF file to an image file.

    Args:
        page (fitz.Page): The page object to convert.

    Returns:
        bytes: The image file in bytes.
    """
    zoom_x = 2.0  # Horizontal zoom
    zoom_y = 2.0  # Vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # Zoom factor 2 in each dimension

    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def page_to_markdown(page, gap_threshold=10):
    """
    Convert a page of a PDF file to markdown format.

    Args:
        page (fitz.Page): The page object to convert.
        gap_threshold (int, optional): The threshold for vertical gaps between text blocks. Defaults to 10.

    Returns:
        str: The markdown representation of the page.
    """
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda block: (block[1], block[0]))

    markdown_page = []
    previous_block_bottom = 0

    for block in blocks:
        y0 = block[1]
        y1 = block[3]
        block_text = block[4]

        # Check if there's a large vertical gap between this block and the previous one
        if y0 - previous_block_bottom > gap_threshold:
            markdown_page.append("")

        markdown_page.append(block_text)
        previous_block_bottom = y1

    return "\n".join(markdown_page)


def ocr(
    pdf_file, 
    api_key, 
    model="gpt-4o", 
    base_url= 'https://api.openai.com/v1', 
    prompt=DEFAULT_PROMPT, 
    pages_list = None,
    use_llm_for_all = False,
    ):
    """
    Convert a PDF file to a list of markdown-formatted pages using OpenAI's API.

    Args:
        pdf_file (io.BytesIO): The PDF file object.
        api_key (str): The OpenAI API key.
        model (str, optional): by default is gpt-4o
        base_url (str): You can use this one to point the client whereever you need it like Ollama
        prompt (str, optional): The prompt to send to the API. Defaults to DEFAULT_PROMPT.
        pages_list (list, optional): A list of page numbers to process. If provided, only these pages will be converted. Defaults to None, which processes all pages.
        use_llm_for_all (bool, optional): If True, all pages will be processed using the LLM, regardless of visual content. Defaults to False.
    Returns:
        list: A list of strings, each containing the markdown representation of a PDF page.
    """
    client = OpenAI(api_key=api_key, base_url = base_url)  # Create OpenAI client

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    pages_list = pages_list or list(range(1, doc.page_count + 1))  # Default to all pages if not provided

    # List to store markdown content for each page
    markdown_pages = [None] * len(pages_list)

    image_files = {}
    for page_num in pages_list:
        page = doc.load_page(page_num - 1)
        if not use_llm_for_all and not is_visual_page(page):
            logging.info(f"The content of Page {page.number + 1} will be extracted using text parsing.")
            # Extract text using traditional OCR
            markdown_content = page_to_markdown(page)
            if markdown_content:
                markdown_pages[page_num - 1] = markdown_content
            else:
                logging.warning(f"Page {page.number + 1} is empty or contains no text.")
                markdown_pages[page_num - 1] = f"Page {page.number + 1} is empty or contains no text."

        else:
            logging.info(f"The content of page {page.number + 1} will be extracted using the LLM.")
            # Convert page to image
            image_file = page_to_image(page)
            image_files[page_num - 1] = io.BytesIO(image_file)

    if image_files:
        # Process each image file in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each image file
            future_to_page = {executor.submit(image_to_markdown, img_file, client, model, prompt): page_num 
                            for page_num, img_file in image_files.items()}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    markdown_content = future.result()
                    if markdown_content:
                        markdown_pages[page_num] = markdown_content
                    else:
                        markdown_pages[page_num] = f"Error processing page {page_num + 1}."
                except Exception as e:
                    logging.error(f"Error processing page {page_num + 1}: {e}")
                    markdown_pages[page_num] = f"Error processing page {page_num + 1}: {str(e)}"

    return markdown_pages
