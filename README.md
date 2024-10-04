# AIPDF: Simple PDF OCR with GPT-like Multimodal Models

Screw traditional OCRs or heavy libraries to get data from PDFs, GenAI does a better job!

AIPDF is a stand-alone, minimalistic, yet powerful pure Python library that leverages multi-modal gen AI models (OpenAI, llama3 or compatible alternatives) to extract data from PDFs and convert it into various formats such as Markdown or JSON. 

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
from aipdf.ocr import ocr

# Your OpenAI API key   
api_key = 'your_openai_api_key'

file = open('somepdf.pdf', 'rb')
markdown_pages = ocr(file, api_key, prompt="extract markdown, extract tables and turn charts into tables")

```

##  Ollama

You can use with any ollama multi-modal models 

```python
ocr(pdf_file, api_key='ollama', model="llama3.2", base_url= 'http://localhost:11434/v1', prompt=DEFAULT_PROMPT)
```
## Any file system

We chose that you pass a file object, because that way it is flexible for you to use this with any type of file system, s3, localfiles, urls etc

### From url
```python

pdf_file = io.BytesIO(requests.get('https://arxiv.org/pdf/2410.02467').content)

# extract markdown
pages = ocr(pdf_file, api_key, prompt="extract tables and turn charts into tables, return each table in json")

```
### From S3

```python

s3 = boto3.client('s3', config=Config(signature_version='s3v4'),
                  aws_access_key_id=access_token,
                  aws_secret_access_key='', # Not needed for token-based auth
                  aws_session_token=access_token)


pdf_file = io.BytesIO(s3.get_object(Bucket=bucket_name, Key=object_key)['Body'].read())
# extract markdown
pages = ocr(pdf_file, api_key, prompt="extract tables and turn charts into tables, return each table in json")
```


## Why AIPDF?

1. **Simplicity**: AIPDF provides a straightforward function, it requires minimal setup, dependencies and configuration.
2. **Flexibility**: Extract data into Markdown, JSON, HTML, YAML, whatever... file format and schema.
3. **Power of AI**: Leverages state-of-the-art multi modal models (gpt, llama, ..).
4. **Customizable**: Tailor the extraction process to your specific needs with custom prompts.
5. **Efficient**: Utilizes parallel processing for faster extraction of multi-page PDFs.

## Requirements

- Python 3.7+

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
