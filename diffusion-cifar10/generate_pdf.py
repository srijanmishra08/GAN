#!/usr/bin/env python3
"""
Script to convert the technical report markdown to PDF.
Requires: pip install markdown2 pdfkit
"""

import markdown2
import pdfkit
import os

def markdown_to_pdf(markdown_file, pdf_file):
    """Convert markdown file to PDF."""
    
    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert to HTML
    html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
    
    # Add CSS styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Technical Report: High-Resolution Diffusion Model</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                margin: 2cm;
                font-size: 12pt;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }}
            h1 {{
                font-size: 18pt;
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                font-size: 16pt;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            h3 {{
                font-size: 14pt;
            }}
            h4 {{
                font-size: 12pt;
                font-weight: bold;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 0;
                padding-left: 20px;
                color: #7f8c8d;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .highlight {{
                background-color: #fff3cd;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    try:
        pdfkit.from_string(html_template, pdf_file)
        print(f"✅ PDF generated successfully: {pdf_file}")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("Note: You may need to install wkhtmltopdf:")
        print("  - macOS: brew install wkhtmltopdf")
        print("  - Ubuntu: sudo apt-get install wkhtmltopdf")
        print("  - Windows: Download from https://wkhtmltopdf.org/")

def main():
    """Main function."""
    markdown_file = "technical_report.md"
    pdf_file = "technical_report.pdf"
    
    if not os.path.exists(markdown_file):
        print(f"❌ Markdown file not found: {markdown_file}")
        return
    
    print("📄 Converting technical report to PDF...")
    markdown_to_pdf(markdown_file, pdf_file)

if __name__ == "__main__":
    main() 