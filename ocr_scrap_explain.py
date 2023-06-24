import time
import os
import pytesseract
from selenium import webdriver
from PIL import Image
from selenium.webdriver.chrome.service import Service
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate


service = Service("chromedriver.exe")
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

url = "https://pypi.org/project/pytesseract/"
driver.get(url)
time.sleep(3)
temp_file_path = "tmp.png"

driver.save_screenshot(temp_file_path)

# for windows only
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
ocr_data = pytesseract.image_to_string(Image.open("tmp.png"))
os.remove(temp_file_path)


template = """
You are a chatbot that is helpful.
Below you will find unstructured, incomplete data.
Try to answer questions to your best ability about the given data.

{ocr_data}
Human: {question}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["ocr_data", "question"],
    template=template
)


llm_chain = LLMChain(
    llm=OpenAI(openai_api_key=""),
    prompt=prompt,
    verbose=True
)

print(llm_chain.predict(ocr_data=ocr_data,
                        question="Can you tell what purpose does this project have?"))
