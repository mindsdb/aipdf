# AIPDF: Minimalistic PDF to Markdown (and others), with GPT-like Multimodal Models

AIPDF is a stand-alone, minimalistic, yet powerful pure Python library that leverages multi-modal gen AI models (OpenAI, llama3 or compatible alternatives) to extract data from PDFs and convert it Markdown. 

## Installation

```bash
pip install aipdf
```

## Quick Start

```python
from aipdf import ocr

# Your API key
# This can also be via the environment variable AIPDF_API_KEY
api_key = 'your_api_key'

file = open('somepdf.pdf', 'rb')
markdown_pages = ocr(file, api_key)
```

By default, AIPDF attempts to determine which pages to send to the LLM based on their content and whether they can be processed using traditional text parsing. This is done to improve performance, and the behavior can be overridden by setting the `use_llm_for_all` parameter to `True`:

```python
markdown_pages = ocr(file, api_key, use_llm_for_all=True)
```

Every call to the LLM is made in parallel, so the processing time is significantly reduced. The above function will make these parallel calls using threading, however, it is also possible to make asynchronous calls instead by using the `ocr_async` function:

```python
from aipdf import ocr_async
import asyncio

# Your API key
# This can also be via the environment variable AIPDF_API_KEY
api_key = 'your_api_key'

file = open('somepdf.pdf', 'rb')

async def main():
    markdown_pages = await ocr_async(file, api_key)
    return markdown_pages

markdown_pages = asyncio.run(main())
```

The maximum number of concurrent requests made to the LLM can also be controlled via the `AIPDF_MAX_CONCURRENT_REQUESTS` environment variable. By default, there is no limit set.

##  Ollama

You can use with any ollama multi-modal models 

```python
ocr(pdf_file, api_key='ollama', model="llama3.2", base_url= 'http://localhost:11434/v1', prompt=...)
```
## Any file system

We chose that you pass a file object, because that way it is flexible for you to use this with any type of file system, s3, localfiles, urls etc

### From url
```python

pdf_file = io.BytesIO(requests.get('https://arxiv.org/pdf/2410.02467').content)

# extract
pages = ocr(pdf_file, api_key, prompt="extract tables, return each table in json")

```
### From S3

```python

s3 = boto3.client('s3', config=Config(signature_version='s3v4'),
                  aws_access_key_id=access_token,
                  aws_secret_access_key='', # Not needed for token-based auth
                  aws_session_token=access_token)


pdf_file = io.BytesIO(s3.get_object(Bucket=bucket_name, Key=object_key)['Body'].read())
# extract 
pages = ocr(pdf_file, api_key, prompt="extract charts data, turn it into tables that represent the variables in the chart")
```


## Why AIPDF?

1. **Simplicity**: AIPDF provides a straightforward function, it requires minimal setup, dependencies and configuration.
2. **Power of AI**: Leverages state-of-the-art multi modal models (gpt, llama, ..).
3. **Customizable**: Tailor the extraction process to your specific needs with custom prompts.
4. **Efficient**: Utilizes parallel processing for faster extraction of multi-page PDFs.

## Requirements

- Python 3.7+

We will keep this super clean, only 2 required libraries:

- openai library to talk to completion endpoints
- PyMuPDF library for traditional text parsing and image conversion

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any problems or have any questions, please open an issue on the GitHub repository.

---

AIPDF makes PDF data extraction simple, flexible, and powerful. Try it out and simplify your PDF processing workflow today!
