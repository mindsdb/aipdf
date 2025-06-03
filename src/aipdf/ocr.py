import asyncio
import base64
import concurrent.futures
import io
import logging
import os

import fitz
from httpx import Timeout
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

DEFAULT_PROMPT = """
Extract the full markdown text from the given image, following these guidelines:
- Respond only with markdown, no additional commentary.  
- Capture all the text, respecting titles, headers, subheaders, equations, etc.
- If there are tables in this page, convert each one into markdown table format and include it in the response.
- If there are images, provide a brief description of what is shown in each image, and include it in the response.
- if there are charts, for each chart include a markdown table with the data represents the chart, a column for each of the variables of the cart and the relevant estimated values
          
"""
DEFAULT_DRAWING_AREA_THRESHOLD = 0.1  # 10% of the page area
DEFAULT_GAP_THRESHOLD = 10  # 10 points


def get_openai_client(api_key=None, base_url='https://api.openai.com/v1', is_async=False, timeout=Timeout(None), **kwargs):
    """
    Get an OpenAI client instance.

    Args:
        api_key (str): The OpenAI API key.
        base_url (str): The base URL for the OpenAI API.
        is_async (bool): Whether to create an asynchronous client.
        timeout (Timeout): Timeout for the OpenAI API calls.
        **kwargs: Additional keyword arguments.

    Returns:
        OpenAI or AsyncOpenAI: An instance of the OpenAI client.
    """
    if not api_key:
        api_key = os.getenv("AIPDF_API_KEY")

    if not api_key:
        raise ValueError("API key is required. Please provide it as an argument or set the AIPDF_API_KEY environment variable.")

    if base_url and "openai.azure.com" in base_url:
        if is_async:
            return AsyncAzureOpenAI(api_key=api_key, azure_endpoint=base_url, **kwargs)
        else:
            return AzureOpenAI(api_key=api_key, azure_endpoint=base_url, **kwargs)

    if is_async:
        return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
    else:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)


def _prepare_image_messages(file_object, prompt):
    """
    Helper function to prepare messages for OpenAI API call.

    Args:
        file_object (io.BytesIO): The image file object.
        prompt (str): The prompt to send to the API.

    Returns:
        list: The messages list for the API call.
    """
    base64_image = base64.b64encode(file_object.read()).decode('utf-8')

    return [
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


def _validate_and_extract_content(response):
    """
    Helper function to validate OpenAI API response and extract content.

    Args:
        response: The response object from OpenAI API.

    Returns:
        str or None: The extracted content, or None if validation fails.
    """
    # Validate the response structure before accessing choices
    if not response:
        logging.error(f"Received empty response from OpenAI API: {response}")
        return None

    if not hasattr(response, 'choices') or not response.choices:
        logging.error(f"Response does not contain choices or choices is empty. Response: {response}")
        return None

    if len(response.choices) == 0:
        logging.error(f"Response choices list is empty. Response: {response}")
        return None

    first_choice = response.choices[0]
    if not hasattr(first_choice, 'message') or not first_choice.message:
        logging.error(f"Response choice does not contain message. First choice: {first_choice}")
        return None

    if not hasattr(first_choice.message, 'content'):
        logging.error(f"Response message does not contain content. Message: {first_choice.message}")
        return None

    markdown_content = first_choice.message.content

    # Additional check for empty or None content
    if not markdown_content:
        logging.warning(f"Response content is empty or None. Content: {repr(markdown_content)}")
        return None

    return markdown_content


def image_to_markdown(file_object, client, model="gpt-4o",  prompt=DEFAULT_PROMPT):
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
    logging.debug("About to process a page")

    messages = _prepare_image_messages(file_object, prompt)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        markdown_content = _validate_and_extract_content(response)

        if markdown_content:
            logging.debug("Page processed successfully")
            return markdown_content
        else:
            logging.warning("Page is empty or contains no text.")
            return None

    except Exception as e:
        logging.error(f"An error occurred while processing the image: {e}")
        return None


async def image_to_markdown_async(file_object, client, model="gpt-4o", prompt=DEFAULT_PROMPT):
    """
    Asynchronously process a single image file and convert its content to markdown using OpenAI's API.

    Args:
        file_object (io.BytesIO): The image file object.
        client (AsyncOpenAI): The AsyncOpenAI client instance.
        model (str, optional): by default is gpt-4o
        prompt (str, optional): The prompt to send to the API. Defaults to DEFAULT_PROMPT.

    Returns:
        tuple: A tuple containing the page number and the markdown representation of the image content, or None if an error occurs.
    """
    # Log that we're about to process a page
    logging.debug("About to process a page")

    messages = _prepare_image_messages(file_object, prompt)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )

        markdown_content = _validate_and_extract_content(response)

        if markdown_content:
            logging.debug("Page processed successfully")
            return markdown_content
        else:
            logging.warning("Page is empty or contains no text.")
            return None

    except Exception as e:
        logging.error(f"An error occurred while processing the image: {e}")
        return None


def is_visual_page(page, drawing_area_threshold=DEFAULT_DRAWING_AREA_THRESHOLD):
    """
    Determine if a page is visual based on presence of images or large drawings.

    Args:
        page (fitz.Page): The page object to analyze.
        drawing_area_threshold (float): Minimum fraction of page area that drawings must cover to be visual.

    Returns:
        bool: True if visual page, False otherwise.
    """
    page_area = page.rect.width * page.rect.height

    # Rule 1: If even one image is included, it is a visual page
    images = page.get_images(full=True)
    if len(images) > 0:
        return True

    # Rule 2: If large enough area is covered by real drawings, it is a visual page
    drawing_area = 0
    for d in page.get_drawings():
        rect = d.get("rect")  # Get the bounding box the contains the drawing
        if rect:
            area = rect.width * rect.height
            # Ignore tiny drawings
            if area > 5000:  # minimum size in points² (~0.7% of page if full-page)
                drawing_area += area

    drawing_fraction = drawing_area / page_area

    if drawing_fraction > drawing_area_threshold:
        return True

    # Rule 3: If the page does not contain any text, it is a visual page
    # These could be scanned images or pages with other complex layouts
    if not page.get_text().strip():
        return True

    # Otherwise, it's a text page
    return False


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


def page_to_markdown(page, gap_threshold=DEFAULT_GAP_THRESHOLD):
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


def process_pages(pdf_file, pages_list=None, use_llm_for_all=False, drawing_area_threshold=DEFAULT_DRAWING_AREA_THRESHOLD, gap_threshold=DEFAULT_GAP_THRESHOLD):
    """
    Process the pages of a PDF file to determine which ones are visual and which ones are text-based.

    Args:
        pdf_file (io.BytesIO): The PDF file object.
        pages_list (list, optional): A list of page numbers to process. If provided, only these pages will be converted. Defaults to None, which processes all pages.
        use_llm_for_all (bool, optional): If True, all pages will be processed using the LLM, regardless of visual content. Defaults to False.
        drawing_area_threshold (float): Minimum fraction of page area that drawings must cover to be visual.
        gap_threshold (int): The threshold for vertical gaps between text blocks.

    Returns:
        tuple: A tuple containing a list of markdown-formatted pages and a dictionary of image files.
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    pages_list = pages_list or list(range(1, doc.page_count + 1))  # Default to all pages if not provided

    # List to store markdown content for each page
    markdown_pages = [None] * len(pages_list)

    image_files = {}
    for page_num in pages_list:
        page = doc.load_page(page_num - 1)
        if not use_llm_for_all and not is_visual_page(page, drawing_area_threshold=drawing_area_threshold):
            logging.debug(f"The content of Page {page.number + 1} will be extracted using text parsing.")
            # Extract text using traditional OCR
            markdown_content = page_to_markdown(page, gap_threshold=gap_threshold)
            if markdown_content:
                markdown_pages[page_num - 1] = markdown_content
            else:
                logging.warning(f"Page {page.number + 1} is empty or contains no text.")
                markdown_pages[page_num - 1] = f"Page {page.number + 1} is empty or contains no text."

        else:
            logging.debug(f"The content of page {page.number + 1} will be extracted using the LLM.")
            # Convert page to image
            image_file = page_to_image(page)
            image_files[page_num - 1] = io.BytesIO(image_file)

    return markdown_pages, image_files


def ocr(
    pdf_file,
    api_key = None,
    model="gpt-4o",
    base_url='https://api.openai.com/v1',
    prompt=DEFAULT_PROMPT,
    pages_list=None,
    use_llm_for_all=False,
    drawing_area_threshold=DEFAULT_DRAWING_AREA_THRESHOLD,
    gap_threshold=DEFAULT_GAP_THRESHOLD,
    logging_level=logging.INFO,
    timeout=Timeout(None),
    **kwargs
    ):
    """
    Convert a PDF file to a list of markdown-formatted pages using text parsing and OpenAI's API.
    The OpenAI API is called in parallel using threading for each image file.
    This function is synchronous.

    Args:
        pdf_file (io.BytesIO): The PDF file object.
        api_key (str): The OpenAI API key.
        model (str, optional): by default is gpt-4o
        base_url (str): You can use this one to point the client whereever you need it like Ollama
        prompt (str, optional): The prompt to send to the API. Defaults to DEFAULT_PROMPT.
        pages_list (list, optional): A list of page numbers to process. If provided, only these pages will be converted. Defaults to None, which processes all pages.
        use_llm_for_all (bool, optional): If True, all pages will be processed using the LLM, regardless of visual content. Defaults to False.
        drawing_area_threshold (float): Minimum fraction of page area that drawings must cover to be visual.
        gap_threshold (int): The threshold for vertical gaps between text blocks.
        logging_level (int): The logging level. Defaults to logging.INFO.
        timeout (Timeout): Timeout for the OpenAI API calls.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of strings, each containing the markdown representation of a PDF page.
    """
    # Set up logging
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    client = get_openai_client(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

    # Identify the maximum number of workers for parallel processing
    max_workers = os.getenv("AIPDF_MAX_CONCURRENT_REQUESTS", None)
    if max_workers:
        logging.debug("The maximum number of concurrent requests is set to %s", max_workers)
        max_workers = int(max_workers)

    markdown_pages, image_files = process_pages(
        pdf_file,
        pages_list=pages_list,
        use_llm_for_all=use_llm_for_all,
        drawing_area_threshold=drawing_area_threshold,
        gap_threshold=gap_threshold
    )

    if image_files:
        if max_workers:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        else:
            executor = concurrent.futures.ThreadPoolExecutor()

        # Process each image file in parallel
        with executor:
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


async def ocr_async(
    pdf_file,
    api_key = None,
    model="gpt-4o",
    base_url='https://api.openai.com/v1',
    prompt=DEFAULT_PROMPT,
    pages_list=None,
    use_llm_for_all=False,
    drawing_area_threshold=DEFAULT_DRAWING_AREA_THRESHOLD,
    gap_threshold=DEFAULT_GAP_THRESHOLD,
    logging_level=logging.INFO,
    timeout=Timeout(None),
    **kwargs
    ):
    """
    Convert a PDF file to a list of markdown-formatted pages using text parsing and OpenAI's API.
    The OpenAI API is called asynchronously for each image file.
    This function is asynchronous.

    Args:
        pdf_file (io.BytesIO): The PDF file object.
        api_key (str): The OpenAI API key.
        model (str, optional): by default is gpt-4o
        base_url (str): You can use this one to point the client whereever you need it like Ollama
        prompt (str, optional): The prompt to send to the API. Defaults to DEFAULT_PROMPT.
        pages_list (list, optional): A list of page numbers to process. If provided, only these pages will be converted. Defaults to None, which processes all pages.
        use_llm_for_all (bool, optional): If True, all pages will be processed using the LLM, regardless of visual content. Defaults to False.
        drawing_area_threshold (float): Minimum fraction of page area that drawings must cover to be visual.
        gap_threshold (int): The threshold for vertical gaps between text blocks.
        logging_level (int): The logging level. Defaults to logging.INFO.
        timeout (Timeout): Timeout for the OpenAI API calls.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of strings, each containing the markdown representation of a PDF page.
    """
    # Set up logging
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    client = get_openai_client(api_key=api_key, base_url=base_url, is_async=True, timeout=timeout, **kwargs)

    # Set up a semaphore for limiting concurrent requests if specified
    semaphore = None
    max_concurrent_requests = os.getenv("AIPDF_MAX_CONCURRENT_REQUESTS", None)
    if max_concurrent_requests:
        logging.debug("The maximum number of concurrent requests is set to %s", max_concurrent_requests)
        max_concurrent_requests = int(max_concurrent_requests)
        semaphore = asyncio.Semaphore(max_concurrent_requests)

    markdown_pages, image_files = process_pages(
        pdf_file,
        pages_list=pages_list,
        use_llm_for_all=use_llm_for_all,
        drawing_area_threshold=drawing_area_threshold,
        gap_threshold=gap_threshold
    )

    if image_files:
        # Process each image file in parallel
        tasks = []

        async def task_wrapper(img_file, page_num):
            if semaphore:
                async with semaphore:
                    markdown_content = await image_to_markdown_async(img_file, client, model, prompt)
            else:
                markdown_content = await image_to_markdown_async(img_file, client, model, prompt)
            return page_num, markdown_content

        tasks = [task_wrapper(img_file, page_num) for page_num, img_file in image_files.items()]

        # Collect results as they complete
        results = await asyncio.gather(*tasks)

        for page_num, markdown_content in results:
            if markdown_content:
                markdown_pages[page_num] = markdown_content
            else:
                markdown_pages[page_num] = f"Error processing page {page_num + 1}."

    return markdown_pages
