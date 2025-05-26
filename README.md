  
**Tóm tắt Hoàn chỉnh Hệ thống RAG và Tương tác Agent cho Tài liệu QuantConnect**

**Ngày cập nhật:** 25 tháng 5 năm 2025

**Mục tiêu cốt lõi:** Xây dựng một hệ thống chạy local cho phép các agent AI chuyên biệt (Analyst, PM, Architect, PO/SM, Dev, Documentation) truy cập và sử dụng thông tin một cách chính xác từ các file tài liệu HTML "single-page" của QuantConnect (được tạo bởi SinglePageDocGenerator.py từ kho GitHub QuantConnect/Documentation/tree/master/single-page). Hệ thống này tập trung vào nội dung văn bản và mã nguồn (Python và C\#), bỏ qua hình ảnh, GIF, video để đơn giản hóa và giảm thiểu "ảo giác", nhằm hỗ trợ tối đa cho công việc của các agent, đặc biệt là các tác vụ liên quan đến định nghĩa và code trong "xây dựng hệ thống giao dịch".

**Các Thành phần Chính của Hệ thống:**

1. **RAG Service (Dịch vụ RAG Tài liệu QuantConnect):**

   * **Chức năng:** Là "trung tâm tri thức" chuyên sâu về các khía cạnh Python và C\# của QuantConnect được ghi lại trong tài liệu. Nhận câu hỏi, truy xuất các đoạn tài liệu HTML liên quan đã được xử lý, và dùng LLM (Google Gemini API) để sinh câu trả lời.  
   * **Nguồn dữ liệu chính:** Các file Quantconnect-\*.html (ví dụ: Quantconnect-Lean-Engine.html, Quantconnect-Writing-Algorithms.html) từ thư mục single-page của kho GitHub QuantConnect/Documentation.  
     * **Các file sẽ bỏ qua:** Quantconnect-Local-Platform.html, Quantconnect-Cloud-Platform.html.  
   * **Pipeline RAG (Xây dựng bằng LangChain hoặc LlamaIndex):**  
     * **Data Ingestion & Preprocessing (Thu thập và Tiền xử lý HTML):**  
       * **Thu thập:** Định kỳ clone/pull kho GitHub QuantConnect/Documentation.  
       * **Xác định file mục tiêu:** Tự động quét và xác định các file Quantconnect-\*.html trong thư mục single-page (ngoại trừ các file đã quyết định bỏ qua).  
       * **Phân tích HTML (HTML Parsing):**  
         * Sử dụng thư viện như BeautifulSoup hoặc lxml để phân tích cú pháp các file HTML "single-page" lớn. Cân nhắc kỹ thuật streaming parsing nếu cần thiết do dung lượng file lớn.  
       * **Trích xuất và Phân loại Element:**  
         * Trích xuất các "element" HTML có ý nghĩa: Tiêu đề (từ \<h1\> đến \<h6\>), Đoạn văn (\<p\>), Danh sách (\<ul\>, \<ol\>), **Bảng biểu (**\<table\>**)**, **Khối mã (Code Blocks)** (xác định ngôn ngữ từ class \<pre class="python"\> hoặc \<pre class="csharp"\>).  
       * **Làm sạch và Loại bỏ Nội dung không cần thiết:**  
         * Loại bỏ hoàn toàn các thẻ \<img\> (bao gồm ảnh bìa phía trên mục lục và tất cả ảnh tĩnh khác).  
         * Loại bỏ các thẻ liên quan đến GIF, video.  
         * Loại bỏ các thẻ \<script\>, \<style\>, header/footer/menu điều hướng chung không chứa nội dung cốt lõi.  
       * **Xử lý Mục lục (Table of Content \- ToC):**  
         * Phân tích cú pháp ToC để trích xuất cấu trúc tài liệu (tiêu đề mục, số thứ tự, cấp độ phân cấp từ class toc-h\*, và ID section liên kết href="\#section\_id").  
         * **Không đưa nội dung ToC vào làm chunk chính.** Thông tin từ ToC sẽ được dùng làm metadata cho các chunk nội dung.  
       * **Chunking (Phân đoạn):**  
         * Chia nhỏ nội dung HTML dựa trên cấu trúc logic đã xác định (ví dụ: mỗi section, subsection, hoặc nhóm nội dung dưới một tiêu đề, sử dụng thông tin từ ToC để định hướng).  
         * Áp dụng các công cụ như HTMLHeaderTextSplitter hoặc RecursiveCharacterTextSplitter của LangChain (có thể tùy chỉnh cho code dựa trên ngôn ngữ) cho nội dung text của các element/section đã trích xuất.  
         * Đảm bảo các chunk không quá lớn cho LLM và mô hình embedding.  
         * **Metadata cho chunk:** Gắn metadata chi tiết cho mỗi chunk: tên file HTML nguồn, tiêu đề mục/số thứ tự/cấp độ phân cấp (từ ToC), id của section, và ngôn ngữ lập trình ("python" hoặc "csharp") cho các chunk mã nguồn.  
     * **Embedding (Nhúng Vector):** Sử dụng mô hình embedding của Google (ví dụ: text-embedding-004, qua GoogleGenerativeAIEmbeddings của LangChain) để chuyển đổi mỗi chunk cuối cùng thành vector.  
     * **Vector Database (Cơ sở dữ liệu Vector):** Lưu trữ các vector embedding cùng với siêu dữ liệu chi tiết cho mỗi chunk. Lựa chọn: ChromaDB, FAISS, Weaviate, Pinecone (chạy local hoặc có phương án local).  
     * **Retrieval (Truy xuất):** Khi nhận câu hỏi, nhúng câu hỏi. Tìm kiếm trong Vector Database để lấy top-K chunks tài liệu HTML có nội dung tương đồng nhất. Tận dụng metadata (ví dụ: language cho code, tiêu đề mục) để lọc hoặc ưu tiên kết quả.  
     * **Generation (Sinh Phản hồi):** Gửi các chunks truy xuất được và câu hỏi gốc cho LLM (Google Gemini API, ví dụ: gemini-pro hoặc gemini-1.5-pro, qua ChatGoogleGenerativeAI của LangChain). Prompt được thiết kế để LLM trả lời dựa trên ngữ cảnh được cung cấp và trích dẫn nguồn từ siêu dữ liệu (tên file, tiêu đề mục/section ID).  
   * **Giao diện:** Cung cấp một API endpoint (ví dụ: http://localhost:PORT/query\_qc\_python\_docs) cho các agent tương tác.  
2. **"Bộ nhớ đệm Tri thức" (Knowledge Cache / Shared Knowledge Base):**

   * **Chức năng:** Lưu trữ các cặp câu hỏi-câu trả lời (và nguồn) đã được RAG Service xử lý.  
   * **Mục đích:** Giảm gọi API LLM, giảm độ trễ, tăng tính nhất quán.  
   * **Hoạt động:** Kiểm tra cache trước khi gọi RAG Service. Nếu có kết quả phù hợp và còn актуаль (còn hiệu lực/liên quan), sử dụng từ cache. Nếu không, gọi RAG Service và cập nhật kết quả mới vào cache. Cần có chiến lược cache invalidation, đặc biệt khi tài liệu nguồn có thể được cập nhật (dựa trên việc theo dõi thay đổi file trong kho GitHub).  
3. **Agent "RAG Interaction Orchestrator" (Điều phối Tương tác RAG):**

   * **Chức năng:** Agent/module trung gian chuyên trách.  
   * **Nhiệm vụ:**  
     * Nhận "nhu cầu thông tin" từ các agent AI chuyên biệt.  
     * **Logic "Hỏi":** Dùng LLM (Gemini) để chuyển "nhu cầu" thành câu hỏi RAG tối ưu, có thể bao gồm việc xác định ngôn ngữ lập trình ưu tiên (Python/C\#) nếu agent yêu cầu.  
     * Tương tác với "Bộ nhớ đệm Tri thức".  
     * Gọi RAG Service API nếu cần.  
     * **Logic "Xử lý Output":** Dùng LLM (Gemini) để phân tích, tóm tắt, trích xuất thông tin từ phản hồi của RAG Service (hoặc cache), cấu trúc hóa output cho agent yêu cầu. Nếu RAG trả về mã C\# cho một yêu cầu mà Agent có thể mong muốn Python (do tài liệu chỉ có C\# cho chủ đề đó), Orchestrator có thể làm rõ điều này với Agent.  
     * Trả thông tin đã xử lý cho agent yêu cầu.  
4. **Các Agent AI Chuyên biệt (Analyst, PM, Architect, PO/SM, Dev, Documentation):**

   * **Chức năng:** Thực hiện nhiệm vụ chuyên môn liên quan đến QuantConnect (Python, C\#), tập trung vào việc "xây dựng hệ thống giao dịch".  
   * **Tích hợp Logic "Tự nhận ra Điểm mù Kiến thức":** Module/Agent "Tự đánh giá" dùng LLM để phân tích nhiệm vụ và kiến thức hiện có, kích hoạt "nhu cầu thông tin" nếu cần tra cứu định nghĩa, ví dụ code, hoặc giải thích khái niệm.  
   * **Tương tác:** Gửi "nhu cầu thông tin" đến "RAG Interaction Orchestrator", nhận lại thông tin đã xử lý và sử dụng để hoàn thành công việc.

**Luồng Hoạt động Tổng thể (Tóm tắt):**

1. **Agent Chính có Nhiệm vụ.** (Ví dụ: Agent Analyst cần hiểu một API hoặc tìm ví dụ code)  
2. **Module "Tự đánh giá"** của Agent Chính xác định có "điểm mù kiến thức" liên quan đến tài liệu QuantConnect.  
3. Nếu có, Agent Chính gửi "nhu cầu thông tin" (ví dụ: "giải thích API X" hoặc "cho ví dụ Python của hàm Y") đến **"RAG Interaction Orchestrator"**.  
4. **"RAG Interaction Orchestrator"**:  
   * Kiểm tra **"Bộ nhớ đệm Tri thức"**.  
   * (Nếu cần) Dùng Gemini để **tạo câu hỏi RAG chi tiết** (có thể kèm theo yêu cầu lọc theo ngôn ngữ).  
   * Gửi câu hỏi đến **RAG Service**.  
5. **RAG Service** (sử dụng phân tích HTML cho input, Gemini cho generation) xử lý và trả kết quả (văn bản, đoạn mã kèm metadata ngôn ngữ và nguồn).  
6. **"RAG Interaction Orchestrator"** nhận kết quả:  
   * Dùng Gemini để **xử lý output** (tóm tắt, trích xuất, cấu trúc hóa, làm rõ nếu chỉ có mã C\# cho một yêu cầu Python).  
   * **Cập nhật "Bộ nhớ đệm Tri thức"**.  
   * Trả thông tin đã xử lý cho Agent Chính.  
7. **Agent Chính** sử dụng thông tin để hoàn thành nhiệm vụ.

**Công nghệ và Công cụ Chính:**

* **LLM APIs:** Google Gemini API.  
* **Framework Phát triển LLM:** LangChain (ưu tiên) hoặc LlamaIndex.  
* **Xử lý HTML:** BeautifulSoup, lxml.  
* **Mô hình Embedding:** Google (ví dụ: text-embedding-004).  
* **Vector Database:** ChromaDB, FAISS, Weaviate, Pinecone (có phương án chạy local).  
* **API Framework (cho RAG Service):** FastAPI (hoặc tương tự).  
* **Quản lý Nguồn:** Git (để theo dõi và cập nhật tài liệu từ QuantConnect/Documentation).

