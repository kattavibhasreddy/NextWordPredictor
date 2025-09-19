import bz2
from lxml import etree

WIKI_XML_FILE = 'simplewiki-latest-pages-articles.xml.bz2'

print(f"Inspecting the root tag of '{WIKI_XML_FILE}' to find its namespace...")

try:
    with bz2.open(WIKI_XML_FILE, 'rb') as bz2_file:
        # Use iterparse to read just the start of the first element
        for event, elem in etree.iterparse(bz2_file, events=('start',)):
            # The first element is the root. Get its tag and print it.
            print(f"\nFound root tag: {elem.tag}")
            
            # We have what we need, so we can stop parsing.
            break
            
    print("\nInspection complete.")

except Exception as e:
    print(f"\nAn error occurred: {e}")