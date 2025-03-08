import os
import torch
import base64
import urllib.request
import tempfile
import gradio as gr
import shutil
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# Initialize the model (globally to avoid reloading it on each request)
print("Initializing the model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")


def download_pdf(url):
    """Download a PDF from the specified URL"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        urllib.request.urlretrieve(url, temp_file.name)
        return temp_file.name
    except Exception as e:
        return None, f"Error during PDF download: {str(e)}"


def process_pdf_url(
    url,
    page_number=1,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a PDF from URL and generate output using olmOCR"""
    try:
        # Download the PDF
        pdf_path = download_pdf(url)
        if pdf_path is None:
            return "Error downloading the PDF", None

        # Process the PDF
        result, image = process_pdf_file(
            pdf_path,
            page_number,
            temperature,
            max_new_tokens,
            num_return_sequences,
            do_sample,
        )

        # Clean up temporary files
        os.unlink(pdf_path)

        return result, image

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_pdf_file(
    pdf_path,
    page_number=1,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a local PDF and generate output using olmOCR"""
    try:
        # Render the PDF page as an image
        image_base64 = render_pdf_to_base64png(
            pdf_path, page_number, target_longest_image_dim=1024
        )

        # Process the PDF with the generated base64
        return process_pdf_base64(
            image_base64,
            pdf_path,
            page_number,
            temperature,
            max_new_tokens,
            num_return_sequences,
            do_sample,
        )

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_file_upload(
    file,
    page_number=1,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a file (PDF or image) uploaded by the user"""
    try:
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.name)[1]
        )
        temp_file.close()
        shutil.copy(file.name, temp_file.name)

        # Determine if it's a PDF or an image
        file_extension = os.path.splitext(temp_file.name)[1].lower()

        if file_extension == ".pdf":
            # Process as PDF
            result, image = process_pdf_file(
                temp_file.name,
                page_number,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
        elif file_extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        ]:
            # Process as image
            result, image = process_image_file(
                temp_file.name,
                temperature,
                max_new_tokens,
                num_return_sequences,
                do_sample,
            )
        else:
            result = f"Unsupported file format: {file_extension}. Please use PDF or images (JPG, PNG, etc.)"
            image = None

        # Clean up temporary files
        os.unlink(temp_file.name)

        return result, image

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_image_file(
    image_path,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Process a local image and generate output using olmOCR"""
    try:
        # Open the image
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            image_base64 = base64.b64encode(img_data).decode("utf-8")

        # Use a generic anchor text for images
        anchor_text = "Image analysis."

        # Process the image with the generated base64
        return process_pdf_base64(
            image_base64,
            None,  # No PDF path
            1,  # Not applicable for images
            temperature,
            max_new_tokens,
            num_return_sequences,
            do_sample,
            anchor_text=anchor_text,
        )

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


def process_pdf_base64(
    image_base64,
    pdf_path=None,
    page_number=1,
    temperature=0.8,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=True,
    anchor_text=None,
):
    """Process an image in base64 format and generate output using olmOCR"""
    try:
        # If a PDF path was provided, get the anchor text
        if pdf_path and not anchor_text:
            anchor_text = get_anchor_text(
                pdf_path, page_number, pdf_engine="pdfreport", target_length=4000
            )
        elif not anchor_text:
            # If we don't have a PDF or a specified anchor text, use a generic anchor text
            anchor_text = "Document analysis."

        prompt = build_finetuning_prompt(anchor_text)

        # Build the complete prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        # Apply the chat template and processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

        # Display the image
        rendered_image = main_image.copy()

        inputs = processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for (key, value) in inputs.items()}

        # Generate the output
        output = model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
        )

        # Decode the output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )

        return text_output[0], rendered_image

    except Exception as e:
        import traceback

        return f"Error: {str(e)}\n{traceback.format_exc()}", None


# Create the Gradio interface
with gr.Blocks(title="olmOCR Document Analyzer") as demo:
    gr.Markdown("# olmOCR Document Analyzer")
    gr.Markdown("Analyze PDF documents and images using the olmOCR-7B model")

    with gr.Tabs() as tabs:
        with gr.TabItem("PDF URL"):
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="PDF URL",
                        placeholder="https://example.com/document.pdf",
                    )
                    page_number_url = gr.Number(
                        label="Page Number", value=1, minimum=1, step=1
                    )

                    with gr.Row():
                        with gr.Column():
                            temperature_url = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                            )
                            max_new_tokens_url = gr.Slider(
                                label="Max New Tokens",
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                            )

                        with gr.Column():
                            num_return_sequences_url = gr.Slider(
                                label="Number of Returned Sequences",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                            do_sample_url = gr.Checkbox(label="Do Sample", value=True)

                    submit_btn_url = gr.Button("Analyze PDF", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            image_output_url = gr.Image(label="PDF Page", type="pil")

                        with gr.Column():
                            text_output_url = gr.Textbox(label="Result", lines=10)

                submit_btn_url.click(
                    fn=process_pdf_url,
                    inputs=[
                        url_input,
                        page_number_url,
                        temperature_url,
                        max_new_tokens_url,
                        num_return_sequences_url,
                        do_sample_url,
                    ],
                    outputs=[text_output_url, image_output_url],
                )

                gr.Markdown("### Example URL")
                gr.Examples(
                    examples=[
                        ["https://molmo.allenai.org/paper.pdf", 1, 0.8, 50, 1, True],
                    ],
                    inputs=[
                        url_input,
                        page_number_url,
                        temperature_url,
                        max_new_tokens_url,
                        num_return_sequences_url,
                        do_sample_url,
                    ],
                )

        with gr.TabItem("Upload File"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload a file (PDF or image)",
                        file_types=[
                            ".pdf",
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".tiff",
                            ".tif",
                            ".webp",
                        ],
                    )
                    page_number_file = gr.Number(
                        label="Page Number (for PDFs only)", value=1, minimum=1, step=1
                    )

                    with gr.Row():
                        with gr.Column():
                            temperature_file = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                            )
                            max_new_tokens_file = gr.Slider(
                                label="Max New Tokens",
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                            )

                        with gr.Column():
                            num_return_sequences_file = gr.Slider(
                                label="Number of Returned Sequences",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                            do_sample_file = gr.Checkbox(label="Do Sample", value=True)

                    submit_btn_file = gr.Button("Analyze File", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            image_output_file = gr.Image(
                                label="Image/PDF Page", type="pil"
                            )

                        with gr.Column():
                            text_output_file = gr.Textbox(label="Result", lines=10)

                submit_btn_file.click(
                    fn=process_file_upload,
                    inputs=[
                        file_input,
                        page_number_file,
                        temperature_file,
                        max_new_tokens_file,
                        num_return_sequences_file,
                        do_sample_file,
                    ],
                    outputs=[text_output_file, image_output_file],
                )

        with gr.TabItem("Direct Base64"):
            with gr.Row():
                with gr.Column(scale=2):
                    base64_input = gr.Textbox(
                        label="Enter the base64 string of the image", lines=5
                    )

                    with gr.Row():
                        with gr.Column():
                            temperature_base64 = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                            )
                            max_new_tokens_base64 = gr.Slider(
                                label="Max New Tokens",
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                            )

                        with gr.Column():
                            num_return_sequences_base64 = gr.Slider(
                                label="Number of Returned Sequences",
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                            )
                            do_sample_base64 = gr.Checkbox(
                                label="Do Sample", value=True
                            )

                    submit_btn_base64 = gr.Button("Analyze Image", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            image_output_base64 = gr.Image(
                                label="Decoded Image", type="pil"
                            )

                        with gr.Column():
                            text_output_base64 = gr.Textbox(label="Result", lines=10)

                submit_btn_base64.click(
                    fn=lambda b64, t, m, n, d: process_pdf_base64(
                        b64, None, 1, t, m, n, d
                    ),
                    inputs=[
                        base64_input,
                        temperature_base64,
                        max_new_tokens_base64,
                        num_return_sequences_base64,
                        do_sample_base64,
                    ],
                    outputs=[text_output_base64, image_output_base64],
                )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
