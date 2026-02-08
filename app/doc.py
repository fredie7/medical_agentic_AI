import PyPDF2
with open('Analytics_Engineer_Roadmap.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    number_of_pages = len(reader.pages)
    print(f'The PDF has {number_of_pages} pages.')
    text = ""
    for pages in reader.pages:
        text += pages.extract_text() + "\n"
    print(text)