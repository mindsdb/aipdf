import unittest
from unittest.mock import MagicMock, patch
import io
from src.aipdf.ocr import (
    image_to_markdown,
    is_visual_page,
    page_to_image,
    page_to_markdown,
    ocr,
)


class TestPageToImage(unittest.TestCase):
    @patch("fitz.Page")
    def test_page_to_image(self, mock_page):
        # Mock page pixmap
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake image bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        result = page_to_image(mock_page)

        self.assertEqual(result, b"fake image bytes")
        mock_page.get_pixmap.assert_called_once()


class TestPageToMarkdown(unittest.TestCase):
    @patch("fitz.Page")
    def test_page_to_markdown(self, mock_page):
        # Mock page text blocks
        mock_page.get_text.return_value = [
            (0, 0, 100, 50, "Header"),
            (0, 60, 100, 100, "Body text"),
        ]

        result = page_to_markdown(mock_page)

        self.assertEqual(result, "Header\nBody text")
        mock_page.get_text.assert_called_once_with("blocks")


class TestIsVisualPage(unittest.TestCase):
    @patch("fitz.Page")
    def test_is_visual_page_with_images(self, mock_page):
        # Mock page with images
        mock_page.get_images.return_value = [("image1",)]

        result = is_visual_page(mock_page)

        self.assertTrue(result)

    @patch("fitz.Page")
    def test_is_visual_page_with_drawings(self, mock_page):
        # Mock page with drawings
        mock_page.rect.width = 100
        mock_page.rect.height = 100
        mock_page.get_images.return_value = []

        mock_rect = MagicMock()
        mock_rect.width = 100
        mock_rect.height = 100
        mock_drawing = {"rect": mock_rect}
        mock_page.get_drawings.return_value = [mock_drawing]
        mock_page.get_drawings.return_value = [mock_drawing]

        result = is_visual_page(mock_page)

        self.assertTrue(result)

    @patch("fitz.Page")
    def test_is_visual_page_with_no_visual_content(self, mock_page):
        # Mock page with no images or drawings
        mock_page.rect.width = 100
        mock_page.rect.height = 100
        mock_page.get_images.return_value = []
        mock_page.get_drawings.return_value = []
        mock_page.get_text.return_value = "Some text"

        result = is_visual_page(mock_page)

        self.assertFalse(result)


class TestImageToMarkdown(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_image_to_markdown_success(self, mock_openai):
        # Mock OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Markdown content"))]
        )

        file_object = io.BytesIO(b"fake image data")
        result = image_to_markdown(file_object, mock_client)

        self.assertEqual(result, "Markdown content")
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_image_to_markdown_failure(self, mock_openai):
        # Mock OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        file_object = io.BytesIO(b"fake image data")
        result = image_to_markdown(file_object, mock_client)

        self.assertIsNone(result)


class TestOCR(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()