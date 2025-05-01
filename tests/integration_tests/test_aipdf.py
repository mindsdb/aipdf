import io
import logging
import os
import time
import unittest

from src.aipdf.ocr import ocr, ocr_async


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestOCRIntegration(unittest.TestCase):
    def setUp(self):
        # Path to the directory containing test PDF files
        self.files_dir = os.path.join(os.path.dirname(__file__), "files")

    def test_ocr_on_sample_pdfs(self):
        # Iterate through all PDF files in the files directory
        for file_name in os.listdir(self.files_dir):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.files_dir, file_name)
                with open(file_path, "rb") as pdf_file:
                    pdf_bytes = io.BytesIO(pdf_file.read())

                start_time = time.time()
                result = ocr(pdf_bytes)
                elapsed_time = time.time() - start_time
                logging.info(f"Processed {file_name} in {elapsed_time:.2f} seconds")

                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0, f"Result is empty for file: {file_name}")
                for page_content in result:
                    self.assertIsInstance(page_content, str)
                    self.assertGreater(len(page_content.strip()), 0, f"Page content is empty for file: {file_name}")


class TestOCRAsyncIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Path to the directory containing test PDF files
        self.files_dir = os.path.join(os.path.dirname(__file__), "files")

    async def test_ocr_async_on_sample_pdfs(self):
        # Iterate through all PDF files in the files directory
        for file_name in os.listdir(self.files_dir):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.files_dir, file_name)
                with open(file_path, "rb") as pdf_file:
                    pdf_bytes = io.BytesIO(pdf_file.read())

                start_time = time.time()
                result = await ocr_async(pdf_bytes)
                elapsed_time = time.time() - start_time
                logging.info(f"Processed {file_name} in {elapsed_time:.2f} seconds")

                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0, f"Result is empty for file: {file_name}")
                for page_content in result:
                    self.assertIsInstance(page_content, str)
                    self.assertGreater(len(page_content.strip()), 0, f"Page content is empty for file: {file_name}")


if __name__ == "__main__":
    unittest.main()