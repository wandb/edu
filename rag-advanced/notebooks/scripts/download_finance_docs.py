import concurrent.futures
import requests
import io
from PyPDF2 import PdfReader
from tqdm.notebook import tqdm
import time
import random
import json
import os
import weave
import pathlib

class PDFProcessor:
    def __init__(self):
        self.GITHUB_API = "https://api.github.com"
        self.REPO_OWNER = "docugami"
        self.REPO_NAME = "KG-RAG-datasets"
        self.DOCS_PATH = "sec-10-q/data/v1/docs"
        self.INITIAL_BACKOFF = 60
        self.MAX_BACKOFF = 3600
        self.MAX_RETRIES = 5
        self.GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
        self.data_dir = pathlib.Path("../data/finance_docs")

    def github_request(self, url):
        headers = {"Authorization": f"token {self.GITHUB_TOKEN}"} if self.GITHUB_TOKEN else {}
        backoff = self.INITIAL_BACKOFF
        
        for attempt in range(self.MAX_RETRIES):
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                print(f"Received 403 Forbidden. Response: {response.text}")
                print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
                
                if 'rate limit exceeded' in response.text.lower():
                    wait_time = min(backoff * (2 ** attempt) + random.uniform(0, 1), self.MAX_BACKOFF)
                    print(f"Rate limit exceeded. Attempt {attempt + 1}/{self.MAX_RETRIES}. Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"GitHub API request forbidden. Please check your token or permissions.")
            else:
                print(f"Unexpected status code: {response.status_code}. Response: {response.text}")
                response.raise_for_status()
        
        raise Exception(f"Failed to retrieve data after {self.MAX_RETRIES} attempts")

    def get_pdf_files(self):
        url = f"{self.GITHUB_API}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/contents/{self.DOCS_PATH}"
        response = self.github_request(url)
        
        contents = response.json()
        return [item for item in contents if item["name"].endswith('.pdf')]

    def download_pdf(self, pdf_file):
        pdf_url = pdf_file['download_url']
        pdf_name = pdf_file['name']
        local_path = self.data_dir / pdf_name
        
        if not local_path.exists():
            response = requests.get(pdf_url)
            if response.status_code == 200:
                local_path.write_bytes(response.content)
                return str(local_path)
            else:
                print(f"Failed to download PDF from {pdf_url}")
                return None
        return str(local_path)

    def process_pdf(self, pdf_file):
        local_path = self.download_pdf(pdf_file)
        if local_path:
            with open(local_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return {
                "content": text,
                "metadata": {
                    "source": pdf_file['name'],
                    "raw_tokens": len(text.split()),
                },
            }
        return None

    @weave.op()
    def load_pdf_documents(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = self.get_pdf_files()
        
        # Use all available CPU cores
        num_processes = os.cpu_count()
        
        data = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.process_pdf, pdf_file) for pdf_file in pdf_files]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files), desc="Processing PDF files"):
                result = future.result()
                if result is not None:
                    data.append(result)
        
        return data