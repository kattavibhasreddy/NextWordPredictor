import os
import bz2
from lxml import etree
import mwparserfromhell

# --- Configuration ---
WIKI_XML_FILE = 'simplewiki-latest-pages-articles.xml.bz2'
OUTPUT_TEXT_FILE = 'data.txt'
# Number of pages to process before printing a progress update
PROGRESS_INTERVAL = 1000

def extract_and_clean_text():
    """
    Decompresses and parses the Wikipedia XML dump, cleans the wikitext,
    and writes the plain text to an output file.
    This is a memory-efficient streaming parser.
    """
    print(f"--- Starting extraction from '{WIKI_XML_FILE}' ---")

    # Check if the input file exists
    if not os.path.exists(WIKI_XML_FILE):
        print(f"Error: Input file '{WIKI_XML_FILE}' not found.")
        print("Please make sure you have downloaded it and placed it in the project folder.")
        return

    page_count = 0
    # Open the output file for writing
    with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as outfile:
        # Use bz2.open to read the compressed file directly
        with bz2.open(WIKI_XML_FILE, 'rb') as bz2_file:
            # lxml.etree.iterparse allows us to process a huge XML file
            # piece by piece without loading it all into memory.
            # We are interested in the 'end' event for every 'page' tag.
            
            # <-- FIX: Updated to export-0.11
            context = etree.iterparse(bz2_file, events=('end',), tag='{http://www.mediawiki.org/xml/export-0.11/}page')

            for event, elem in context:
                # The XML has a "namespace", so we must include it to find tags.
                
                # <-- FIX: Updated to export-0.11
                ns = '{http://www.mediawiki.org/xml/export-0.11/}'
                
                # Find the text content within the <text> tag of each page
                text_content = elem.findtext(f'.//{ns}text')
                
                if text_content:
                    # Use mwparserfromhell to parse the raw wikitext
                    wikicode = mwparserfromhell.parse(text_content)
                    
                    # Use the strip_code() method to remove all markup
                    # (links, templates, bold/italics, etc.) and get plain text.
                    plain_text = wikicode.strip_code().strip()
                    
                    # Write the clean text to our output file, followed by a newline
                    outfile.write(plain_text + '\n')

                page_count += 1
                
                # Print a progress update every N pages
                if page_count % PROGRESS_INTERVAL == 0:
                    print(f" ...Processed {page_count} pages")

                # This is a crucial step for memory efficiency. It clears the processed
                # element from memory, preventing the XML tree from growing indefinitely.
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

    print(f"\n--- Extraction Complete! ---")
    print(f"Processed a total of {page_count} pages.")
    print(f"Clean text has been saved to '{OUTPUT_TEXT_FILE}'.")
    print("You are now ready to run 'train_model.py'.")

if __name__ == "__main__":
    extract_and_clean_text()