# AIPDF: Simple PDF Data Extraction with AI Vision Models

Screw traditional OCRs or heavy libraries to get data from PDFs, GenAI does a better job!

AIPDF is a stand-alone, minimalistic, yet powerful pure Python library that leverages OpenAI's vision models (or compatible alternatives) to extract data from PDFs and convert it into various formats such as Markdown or JSON. 

## Installation

```bash
pip install aipdf
```

in macOS you will need to install poppler
```bash
brew install poppler 
```

## Quick Start


```python
import requests
import io
from aipdf.ocr import ocr

# Your OpenAI API key   
api_key = 'your_openai_api_key'
# from url
pdf_file = io.BytesIO(requests.get('https://arxiv.org/pdf/2410.02467').content)

# extract markdown
markdown_pages = ocr(pdf_file, api_key, prompt="get markdown format, extract tables and turn charts into tables")

```

We chose that you pass a file object, because that way it is flexible for you to use this with any type of file system, s3, localfiles, urls etc


```python
from aipdf.ocr import ocr
import io

# Your OpenAI API key   
api_key = 'your_openai_api_key'

file = open('somepdf.pdf', 'rb')
markdown_pages = ocr(file, api_key, prompt="extract json structure from this file, extract tables and turn charts into tables")

```

## Customization

You can easily customize the extraction process by providing a custom prompt:

```python
custom_prompt = "Extract a json of this document, including all tables from and charts."
markdown_pages = ocr(pdf_file, api_key, prompt=custom_prompt)
```

## Why AIPDF?

1. **Simplicity**: AIPDF provides a straightforward API that requires minimal setup and configuration.
2. **Flexibility**: Extract data into Markdown, which can be easily converted to other formats like JSON or HTML.
3. **Power of AI**: Leverages state-of-the-art vision models for accurate and intelligent data extraction.
4. **Customizable**: Tailor the extraction process to your specific needs with custom prompts.
5. **Efficient**: Utilizes parallel processing for faster extraction of multi-page PDFs.

## Requirements

- Python 3.7+
- OpenAI API key (or compatible vision model API)

We will keep this super clean, only 3 required libraries:

- openai library to talk to completion endpoints
- pdf2image library (for PDF to image conversion)
- Pillow (PIL) library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any problems or have any questions, please open an issue on the GitHub repository.

---

AIPDF makes PDF data extraction simple, flexible, and powerful. Try it out and simplify your PDF processing workflow today!
