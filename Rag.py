import os
import time
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class RagPipeLine():
    def __init__(self):
        self.model_name = os.getenv("model_name", "BAAI/bge-m3")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("COLLECTION_NAME", "Phone_store")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if not all([self.qdrant_url, self.qdrant_api_key, self.google_api_key]):
            raise ValueError("Missing required environment variables: QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY")

        self.embeddings = self.load_embeddings()
        self.retriever = self.load_retriever(embeddings=self.embeddings)
        self.pipe = self.load_model_pipeline(max_new_tokens=300)
        self.prompt = self.load_prompt_template()
        self.rag_pipeline = self.load_rag_pipeline(llm=self.pipe,
                                            retriever=self.retriever,
                                            prompt=self.prompt)
        print("RAG Pipeline initialized successfully.")

    def load_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return embeddings

    def load_retriever(self, embeddings):
        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, prefer_grpc=False, check_compatibility=False)
        db = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )
        return db.as_retriever(search_kwargs={"k": 15})

    def load_model_pipeline(self, max_new_tokens=300):
        pipe = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.1,
            top_p=0.8,
            max_output_tokens=max_new_tokens,
            google_api_key=self.google_api_key
        )
        return pipe

    def get__LLM__answer(self, question):
        response = self.pipe.invoke(question)
        return response

    def load_prompt_template(self):
        PROMPT_TEMPLATE = """
        Bạn là chuyên gia tư vấn điện thoại chuyên nghiệp với kinh nghiệm 10 năm tại Thegioididong/Cellphones.

        Dựa vào **THÔNG SỐ KỸ THUẬT CỤ THỂ** của từng mẫu điện thoại dưới đây và kiến thức chuyên môn về thị trường Việt Nam, hãy tư vấn chân thực, khách quan.

        {context}

        ---

        **KHÁCH HÀNG HỎI**: {question}

        **HUỚNG DẪN TƯ VẤN**:
        1. **PHÂN TÍCH ƯU/ NHƯỢC** từ specs (pin thực tế, hiệu năng gaming, camera thực chiến)
        2. **SO SÁNH** với đối thủ cùng tầm giá (iPhone vs Samsung vs Xiaomi)
        3. **PHÙ HỢP NHU CẦU**: gaming/streaming/chụp ảnh/pin trâu/dưới Xtr
        4. **GIÁ TRỊ TIỀN BẠC**: Đáng mua hay chờ sale?
        5. **KIẾN THỨC BỔ SUNG**: Benchmark thực tế, độ bền VN, chính sách bảo hành

        **TRẢ LỜI**:
        ✅ Ngắn gọn, thuyết phục như sales pro
        ✅ Bullet points rõ ràng  
        ✅ Kết luận "Nên mua/Không nên/Đợi model mới"
        ✅ Giá VND, so sánh local market

        Bắt đầu tư vấn 👇
        """
        input_variables = ["context", "question"]
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=input_variables)
        return prompt

    def load_rag_pipeline(self, llm, retriever, prompt):
        doc_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}",
        )
        rag_pipeline = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context",
                "document_prompt": doc_prompt,
            },
        )
        return rag_pipeline


    def rag_ask(self, question):
        max_retries = 3
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                # Use invoke instead of __call__ to avoid deprecation warning
                return self.rag_pipeline.invoke({"question": question})
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                
                print(f"Error in rag_ask: {e}")
                raise e
