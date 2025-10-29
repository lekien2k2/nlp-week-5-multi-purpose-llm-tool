## Thành viên nhóm

| Họ và tên                | MSSV    |
| ------------------------ | ------- |
| Lê Tấn Kiên              | 2591309 |
| Đỗ Quốc Khánh            | 2591307 |
| Nguyễn Trương Hoàng Hiếu | 2591304 |

# 📘 HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY ỨNG DỤNG MULTI-PURPOSE LLM TOOL

## 📋 MỤC LỤC

1. [Giới thiệu](#giới-thiệu)
2. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
3. [Các bước cài đặt](#các-bước-cài-đặt)
4. [Chạy ứng dụng](#chạy-ứng-dụng)
5. [Sử dụng ứng dụng](#sử-dụng-ứng-dụng)
6. [Cấu trúc dự án](#cấu-trúc-dự-án)
7. [Khắc phục sự cố](#khắc-phục-sự-cố)

---

## 📖 GIỚI THIỆU

**Multi-Purpose LLM Tool** là một ứng dụng web single-page sử dụng nhiều Large Language Model (LLM) từ các nhà cung cấp cloud để thực hiện các tác vụ xử lý văn bản và chat trực tiếp.

### Chức năng chính:

- 📝 **Summarize**: Tóm tắt văn bản
- 🇫🇷 **Translate to French**: Dịch sang tiếng Pháp
- 👶 **Explain Like I'm 5**: Giải thích đơn giản
- 🔑 **Extract Keywords**: Trích xuất từ khóa
- 💻 **Generate Python Code**: Tạo code Python
- 💬 **Chat**: Chat trực tiếp với AI (không qua prompt template)

### Tính năng nâng cao:

- 🔄 **So sánh nhiều model**: Chạy cùng một prompt trên nhiều LLM và so sánh kết quả cạnh nhau
- 📚 **Lịch sử**: Tự động lưu tất cả request/response để xem lại và quản lý
- ✏️ **Tùy chỉnh prompt**: Cho phép nhập prompt hoàn toàn tùy chỉnh thay vì dùng prompt mặc định
- 📥 **Export kết quả**: Xuất kết quả ra file JSON hoặc TXT
- 🇻🇳 **Hỗ trợ tiếng Việt**: Tự động phát hiện và yêu cầu AI trả lời bằng tiếng Việt
- 📝 **Markdown rendering**: Hiển thị kết quả với format đẹp (bold, italic, code, headings, lists, v.v.)

### Công nghệ sử dụng:

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (ES6)
- **LLM Providers**:
  - GPT (OpenAI)
  - Deepseek
  - Gemini (Google)
- **Storage**: JSON file (lưu lịch sử)
- **Markdown**: Custom JavaScript parser

---

## 💻 YÊU CẦU HỆ THỐNG

### Phần cứng tối thiểu:

- **CPU**: Intel Core i3 hoặc tương đương
- **RAM**: 4GB (khuyến nghị 8GB)
- **Ổ cứng**: 100MB trống
- **Kết nối**: **Internet bắt buộc** (sử dụng API cloud)

### Phần mềm:

- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux
- **Python**: 3.8 trở lên
- **Trình duyệt**: Chrome, Firefox, Edge, hoặc Safari
- **API Keys**: Cần có API keys từ các nhà cung cấp LLM

### API Keys cần thiết:

1. **OpenAI API Key** (cho GPT)
   - Đăng ký tại: https://platform.openai.com/api-keys
2. **Deepseek API Key**
   - Đăng ký tại: https://platform.deepseek.com
3. **Gemini API Key** (cho Google Gemini)
   - Đăng ký tại: https://aistudio.google.com/app/apikey

**Lưu ý**: Bạn chỉ cần API key của ít nhất 1 model để sử dụng ứng dụng.

---

## 🔧 CÁC BƯỚC CÀI ĐẶT

### BƯỚC 1: Cài đặt Python

1. Tải Python từ: https://www.python.org/downloads/
2. Chạy file cài đặt
3. ✅ **QUAN TRỌNG**: Tick vào "Add Python to PATH"
4. Click "Install Now"
5. Kiểm tra cài đặt:
   ```bash
   python --version
   ```

### BƯỚC 2: Clone hoặc tải dự án

Di chuyển đến thư mục bạn muốn đặt dự án:

```bash
cd your-project-folder
```

### BƯỚC 3: Tạo Virtual Environment

1. Tạo virtual environment:

   ```bash
   python -m venv venv
   ```

2. Kích hoạt virtual environment:

   **Windows (PowerShell):**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**

   ```cmd
   venv\Scripts\activate.bat
   ```

   **macOS/Linux:**

   ```bash
   source venv/bin/activate
   ```

   **Lưu ý** (Windows PowerShell): Nếu gặp lỗi "cannot be loaded because running scripts is disabled", chạy:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

   Sau đó thử lại.

3. Xác nhận đã kích hoạt (sẽ thấy `(venv)` ở đầu dòng lệnh):
   ```
   (venv) PS C:\your-project-folder>
   ```

### BƯỚC 4: Cài đặt Dependencies

Với virtual environment đã kích hoạt, chạy:

```bash
pip install -r requirements.txt
```

Các package sẽ được cài đặt:

- Flask==3.0.0
- python-dotenv==1.0.0
- requests==2.31.0
- google-generativeai==0.8.3
- protobuf>=3.20.2,<6.0.0

### BƯỚC 5: Lấy API Keys

#### OpenAI (GPT):

1. Đăng nhập/đăng ký tại: https://platform.openai.com
2. Vào API Keys: https://platform.openai.com/api-keys
3. Click "Create new secret key"
4. Copy API key (bắt đầu với `sk-`)

#### Deepseek:

1. Đăng nhập/đăng ký tại: https://platform.deepseek.com
2. Vào phần API Keys
3. Tạo API key mới
4. Copy API key

#### Gemini (Google):

1. Truy cập: https://aistudio.google.com/app/apikey
2. Đăng nhập với tài khoản Google
3. Click "Create API Key"
4. Copy API key

### BƯỚC 6: Cấu hình Environment Variables

1. Tạo file `.env` trong thư mục gốc của dự án (cùng cấp với thư mục `app/`)
2. Thêm các API keys vào file `.env`:

```env
# OpenAI API Key (cho GPT)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Deepseek API Key
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# Gemini API Key (Google)
GEMINI_API_KEY=your-gemini-api-key-here
```

**Lưu ý**:

- Thay thế các giá trị `your-*-api-key-here` bằng API keys thực tế của bạn
- Bạn chỉ cần thêm keys của các model bạn muốn sử dụng
- Không commit file `.env` lên Git (đã có trong `.gitignore`)

---

## 🚀 CHẠY ỨNG DỤNG

### Các bước chạy:

1. **Mở Terminal/PowerShell/Command Prompt**

2. **Di chuyển đến thư mục dự án**:

   ```bash
   cd your-project-folder
   ```

3. **Kích hoạt virtual environment** (nếu chưa kích hoạt):

   **Windows (PowerShell):**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**

   ```bash
   source venv/bin/activate
   ```

4. **Di chuyển vào thư mục app và khởi động Flask server**:

   **Cách 1: Chạy từ thư mục app/**

   ```bash
   cd app
   python app.py
   ```

   **Cách 2: Chạy từ thư mục gốc**

   ```bash
   python -m app.app
   ```

5. **Xác nhận server đang chạy**, bạn sẽ thấy:

   ```
   * Serving Flask app 'app'
   * Debug mode: on
   * Running on http://0.0.0.0:5000
   * Running on http://127.0.0.1:5000
   Press CTRL+C to quit
   ```

6. **Mở trình duyệt** và truy cập:
   ```
   http://127.0.0.1:5000
   ```
   hoặc
   ```
   http://localhost:5000
   ```

### Dừng ứng dụng:

Nhấn `Ctrl + C` trong terminal để dừng server.

---

## 📱 SỬ DỤNG ỨNG DỤNG

### Giao diện chính:

1. **Ô nhập liệu**: Nhập văn bản cần xử lý hoặc câu hỏi
2. **Chọn mô hình**: Dropdown để chọn model AI (GPT, Deepseek, hoặc Gemini)
3. **Chế độ so sánh**: Checkbox để chọn so sánh nhiều model cùng lúc
4. **Tùy chỉnh prompt**: Checkbox để nhập prompt hoàn toàn tùy chỉnh
5. **6 nút chức năng**:
   - 📝 Summarize - Tóm tắt văn bản
   - 🇫🇷 Translate to French - Dịch sang tiếng Pháp
   - 👶 Explain Like I'm 5 - Giải thích đơn giản
   - 🔑 Extract Keywords - Trích xuất từ khóa
   - 💻 Generate Python Code - Tạo code Python
   - 💬 Chat - Chat trực tiếp với AI
6. **Ô kết quả**: Hiển thị kết quả từ AI với markdown formatting
7. **Lịch sử**: Xem lại các request trước đó
8. **Export**: Xuất kết quả ra file

### Cách sử dụng cơ bản:

**Ví dụ 1: Chat với GPT**

1. Chọn model "GPT (OpenAI)" từ dropdown
2. Nhập câu hỏi: "What is machine learning?"
3. Click nút **"💬 Chat"**
4. Nhận câu trả lời trực tiếp từ AI

**Ví dụ 2: Tóm tắt văn bản với Deepseek**

1. Chọn model "Deepseek"
2. Nhập văn bản:
   ```
   Artificial intelligence is transforming healthcare. Machine learning
   algorithms can detect diseases from medical images with high accuracy.
   AI helps doctors make better treatment decisions.
   ```
3. Click nút **"📝 Tóm tắt"**
4. Nhận bản tóm tắt (tự động bằng tiếng Việt nếu input là tiếng Việt)

**Ví dụ 3: Tạo Python code**

1. Chọn model "Gemini (Google)"
2. Nhập mô tả: "Create a function that calculates factorial"
3. Click nút **"💻 Tạo code Python"**
4. Nhận code Python với markdown formatting đẹp

### Cách sử dụng tính năng Chat:

Nút **💬 Chat** cho phép bạn trò chuyện trực tiếp với AI:

- **Không thêm prompt template**: Câu hỏi của bạn được gửi trực tiếp đến model
- **Hỗ trợ đa ngôn ngữ**: Hỏi bằng tiếng Việt, tiếng Anh hoặc bất kỳ ngôn ngữ nào
- **Tự do hỏi**: Có thể hỏi bất cứ điều gì, không giới hạn bởi các task cụ thể

**Ví dụ sử dụng Chat:**

- "Giải thích về quantum computing"
- "Viết một bài thơ về mùa thu"
- "What are the benefits of renewable energy?"
- "Cách nấu phở bò như thế nào?"

### Cách sử dụng chế độ so sánh:

1. **Tick checkbox "Chế độ so sánh"**
2. **Chọn các model** bạn muốn so sánh (có thể chọn nhiều)
3. **Nhập văn bản** hoặc sử dụng prompt tùy chỉnh
4. **Click một trong các nút chức năng** (hoặc Chat)
5. **Xem kết quả** hiển thị cạnh nhau từ các model khác nhau
6. **So sánh** chất lượng và phong cách của từng model

**Ví dụ**: So sánh cách GPT, Deepseek và Gemini trả lời cùng một câu hỏi:

- Tick "Chế độ so sánh"
- Chọn cả 3 model: GPT, Deepseek, Gemini
- Nhập: "What is artificial intelligence?"
- Click "💬 Chat"
- Xem 3 câu trả lời khác nhau cạnh nhau với markdown formatting đẹp

### Tùy chỉnh prompt:

1. **Tick checkbox "Tùy chỉnh prompt"**
2. **Nhập prompt** của bạn vào textarea xuất hiện
3. Prompt này sẽ **thay thế hoàn toàn** prompt mặc định từ các nút chức năng
4. **Chọn model** và **click nút bất kỳ** (prompt tùy chỉnh sẽ được sử dụng)

**Ví dụ**:

- Tick "Tùy chỉnh prompt"
- Nhập: "Viết một bài thơ về mùa thu bằng tiếng Việt, 4 câu, thể thơ lục bát"
- Chọn model GPT
- Click nút bất kỳ
- Nhận bài thơ theo yêu cầu

### Xem lịch sử:

1. **Cuộn xuống** phần "📚 Lịch sử"
2. **Xem** các request trước đó với:
   - Model đã sử dụng
   - Task đã chọn
   - Thời gian
   - Input và output
3. **Xóa** từng mục bằng nút "Xóa"
4. **Làm mới** bằng nút "🔄 Làm mới"

### Export kết quả:

1. Sau khi có kết quả, **nút Export sẽ xuất hiện**
2. **Click "📥 Export JSON"** để xuất định dạng JSON
3. Hoặc **Click "📄 Export TXT"** để xuất định dạng text dễ đọc
4. File sẽ **tự động được download**

### Markdown Formatting:

Kết quả từ AI được tự động render markdown:

- **Bold**: `**text**` → **text**
- **Italic**: `*text*` → _text_
- **Code**: `` `code` `` → `code`
- **Code blocks**: `code` → Code trong khung đẹp
- **Headings**: `# Heading` → Heading lớn
- **Lists**: `- item` → Danh sách có bullet
- **Links**: `[text](url)` → Link có thể click

### Thời gian xử lý:

- **GPT**: Khoảng 2-5 giây
- **Deepseek**: Khoảng 2-5 giây
- **Gemini**: Khoảng 3-7 giây
- **Chế độ so sánh**: Tất cả model chạy song song, thời gian = model chậm nhất

### Chi phí:

- **GPT**: Có phí (theo pricing của OpenAI)
- **Deepseek**: Có phí (theo pricing của Deepseek)
- **Gemini**: Có free tier, sau đó có phí

---

## 📁 CẤU TRÚC DỰ ÁN

```
nlp-day5/
│
├── app/
│   ├── app.py                      # Main Flask application
│   │   ├── Route '/'              # Trang chính
│   │   ├── Route '/process'       # Xử lý request đơn lẻ
│   │   ├── Route '/compare'       # So sánh nhiều model
│   │   ├── Route '/history'       # Lấy lịch sử (GET)
│   │   ├── Route '/history/<id>'  # Xóa lịch sử (DELETE)
│   │   ├── Route '/export'        # Xuất kết quả
│   │   └── Route '/health'        # Health check
│   │
│   ├── models/                     # LLM Model implementations
│   │   ├── __init__.py
│   │   ├── openai_model.py        # GPT/OpenAI implementation
│   │   ├── deepseek_model.py      # Deepseek implementation
│   │   ├── gemini_model.py        # Gemini implementation
│   │   └── model_factory.py      # Factory pattern để route calls
│   │
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── language_utils.py      # Vietnamese detection & prompt processing
│   │   ├── prompt_utils.py        # Task prompt definitions
│   │   └── history_utils.py       # History management (save/load)
│   │
│   ├── templates/
│   │   └── index.html             # Giao diện web (HTML + CSS + JS)
│   │
│   └── history.json               # File lưu lịch sử (tự động tạo)
│
├── venv/                          # Virtual environment (tự tạo)
│   ├── Scripts/ (Windows) hoặc bin/ (macOS/Linux)
│   └── Lib/ hoặc lib/
│
├── .env                           # Cấu hình API keys (không commit)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # File này
```

### Giải thích các file chính:

**app/app.py**:

- Main Flask application
- Xử lý HTTP requests và routing
- Business logic cho các tính năng
- Import từ models và utils

**app/models/**:

- **openai_model.py**: Implementation cho GPT API
- **deepseek_model.py**: Implementation cho Deepseek API
- **gemini_model.py**: Implementation cho Gemini API (sử dụng Google AI SDK)
- **model_factory.py**: Factory pattern để route calls đến đúng model

**app/utils/**:

- **language_utils.py**: Phát hiện tiếng Việt và thêm language instruction
- **prompt_utils.py**: Định nghĩa các task prompts mặc định
- **history_utils.py**: Quản lý lịch sử (save, load, delete)

**app/templates/index.html**:

- Giao diện người dùng single-page
- HTML structure với responsive design
- CSS styling (gradient, animations, modern UI)
- JavaScript (AJAX requests, markdown parser, dynamic UI)

**requirements.txt**:

```
Flask==3.0.0          # Web framework
python-dotenv==1.0.0  # Load environment variables
requests==2.31.0      # HTTP library
google-generativeai==0.8.3  # Google Gemini SDK
protobuf>=3.20.2,<6.0.0      # Dependency cho Gemini
```

**.env** (ví dụ):

```
OPENAI_API_KEY=sk-your-openai-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
GEMINI_API_KEY=your-gemini-key-here
```

**app/history.json**:

- Tự động tạo khi có request đầu tiên
- Lưu tối đa 100 entries gần nhất
- Format JSON với metadata đầy đủ

---

## 🛠️ KHẮC PHỤC SỰ CỐ

### Lỗi 1: "python is not recognized"

**Nguyên nhân**: Python chưa được thêm vào PATH

**Giải pháp**:

1. Gỡ cài đặt Python
2. Cài lại và tick "Add Python to PATH"
3. Hoặc thêm thủ công vào Environment Variables
4. Restart terminal

### Lỗi 2: "ModuleNotFoundError: No module named 'flask'"

**Nguyên nhân**: Chưa cài đặt dependencies hoặc chưa kích hoạt virtual environment

**Giải pháp**:

```bash
# Kích hoạt virtual environment trước
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Cài đặt dependencies
pip install -r requirements.txt
```

### Lỗi 3: "Scripts cannot be loaded" (Windows PowerShell)

**Nguyên nhân**: PowerShell chặn chạy scripts

**Giải pháp**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Lỗi 4: "Error: OPENAI_API_KEY not configured"

**Nguyên nhân**: Chưa thêm API key vào file `.env`

**Giải pháp**:

1. Tạo file `.env` trong thư mục gốc dự án (cùng cấp với thư mục `app/`)
2. Thêm các API keys:
   ```
   OPENAI_API_KEY=sk-your-key-here
   DEEPSEEK_API_KEY=sk-your-key-here
   GEMINI_API_KEY=your-key-here
   ```
3. Khởi động lại Flask server

### Lỗi 5: "Error calling OpenAI API" hoặc các API khác

**Nguyên nhân**:

- API key không hợp lệ hoặc hết hạn
- Hết credit/quota
- Internet không kết nối
- API service đang down

**Giải pháp**:

1. Kiểm tra API key có đúng không
2. Kiểm tra tài khoản còn credit không
3. Kiểm tra kết nối internet
4. Thử lại sau vài phút

### Lỗi 6: "Port 5000 already in use"

**Nguyên nhân**: Port 5000 đang được sử dụng

**Giải pháp**:

Sửa file `app/app.py`, dòng cuối:

```python
app.run(debug=True, host="0.0.0.0", port=5001)
```

Sau đó truy cập: http://localhost:5001

### Lỗi 7: "Error: Invalid model provider"

**Nguyên nhân**: Chọn model không hợp lệ

**Giải pháp**:

- Chỉ chọn: GPT, Deepseek, hoặc Gemini
- Đảm bảo model đó có API key đã được cấu hình

### Lỗi 8: Model không trả về kết quả

**Nguyên nhân**:

- API key không hợp lệ
- Rate limit
- Lỗi từ phía nhà cung cấp

**Giải pháp**:

1. Kiểm tra lại API key
2. Kiểm tra rate limit trên dashboard của nhà cung cấp
3. Thử model khác
4. Xem console log để biết lỗi chi tiết

### Lỗi 9: "ModuleNotFoundError: No module named 'models'"

**Nguyên nhân**: Chạy từ sai thư mục

**Giải pháp**:

Chạy từ thư mục `app/`:

```bash
cd app
python app.py
```

Hoặc từ thư mục gốc:

```bash
python -m app.app
```

### Lỗi 10: Gemini không hoạt động

**Nguyên nhân**: Chưa cài đặt google-generativeai hoặc chưa restart server

**Giải pháp**:

1. Cài đặt lại:
   ```bash
   pip install google-generativeai
   ```
2. **Quan trọng**: Khởi động lại Flask server (Ctrl+C rồi chạy lại `python app.py`)

---

## 📊 KIỂM TRA HỆ THỐNG

### Checklist trước khi chạy:

- [ ] Python đã cài đặt (kiểm tra: `python --version`)
- [ ] Virtual environment đã tạo (folder `venv` tồn tại)
- [ ] Virtual environment đã kích hoạt (thấy `(venv)` ở terminal)
- [ ] Dependencies đã cài đặt (chạy: `pip list`)
- [ ] File `.env` đã tồn tại trong thư mục gốc
- [ ] Ít nhất 1 API key đã được thêm vào `.env`
- [ ] Internet đã kết nối

### Test cơ bản:

1. **Test Flask**:

   ```bash
   cd app
   python app.py
   ```

   Nếu thấy "Running on http://127.0.0.1:5000" → Flask hoạt động ✅

2. **Test Web Interface**:

   - Mở http://localhost:5000
   - Kiểm tra hiển thị giao diện
   - Xem phần "Mô hình khả dụng" ở footer

3. **Test API**:

   - Nhập text: "Hello world"
   - Chọn model có API key đã cấu hình
   - Click "💬 Chat"
   - Nếu có kết quả → Ứng dụng hoạt động hoàn hảo ✅

---

## 📈 DEMO NHANH (5 PHÚT)

Để demo nhanh cho thầy:

### 1. Mở Terminal:

```bash
cd app
.\venv\Scripts\Activate.ps1  # Windows (nếu chưa activate)
python app.py
```

### 2. Mở trình duyệt: http://localhost:5000

### 3. Demo 6 chức năng cơ bản:

**Chức năng 1 - Chat**:

```
Input: What is machine learning?
Output: [Câu trả lời chi tiết từ AI với markdown formatting]
```

**Chức năng 2 - Summarize**:

```
Input: Artificial intelligence is transforming healthcare. Machine learning
algorithms can detect diseases from medical images with high accuracy.
AI helps doctors make better treatment decisions.
Output: [Tóm tắt ngắn gọn]
```

**Chức năng 3 - Translate to French**:

```
Input: Hello! How are you today?
Output: [Bản dịch tiếng Pháp]
```

**Chức năng 4 - Explain Like I'm 5**:

```
Input: Explain how the internet works
Output: [Giải thích đơn giản như cho trẻ 5 tuổi]
```

**Chức năng 5 - Extract Keywords**:

```
Input: Machine learning engineer position. Required: Python,
TensorFlow, deep learning, data science. Location: San Francisco.
Output: [Danh sách keywords]
```

**Chức năng 6 - Generate Python Code**:

```
Input: Create a function that calculates factorial
Output: [Python code hoàn chỉnh với markdown formatting]
```

### 4. Demo tính năng nâng cao:

**So sánh model**:

- Tick "Chế độ so sánh"
- Chọn 2-3 model
- Nhập text bất kỳ
- Click một nút chức năng (hoặc Chat)
- Show: Kết quả từ nhiều model cạnh nhau với markdown đẹp

**Lịch sử**:

- Cuộn xuống phần "Lịch sử"
- Show: Các request trước đó với đầy đủ thông tin

**Export**:

- Sau khi có kết quả, click "Export JSON" hoặc "Export TXT"
- Show: File được download

**Tùy chỉnh prompt**:

- Tick "Tùy chỉnh prompt"
- Nhập prompt: "Viết một câu chuyện ngắn về robot"
- Chọn model và click nút bất kỳ
- Show: Kết quả từ prompt tùy chỉnh

**Hỗ trợ tiếng Việt**:

- Nhập tiếng Việt: "Trí tuệ nhân tạo là gì?"
- Click "📝 Tóm tắt" hoặc "💬 Chat"
- Show: Kết quả tự động bằng tiếng Việt

**Markdown formatting**:

- Xem kết quả có code blocks, headings, bold text được render đẹp
- So sánh với text thô (không còn dấu `**` nữa)

---

## 🎯 KẾT LUẬN

Ứng dụng Multi-Purpose LLM Tool đã được phát triển với các đặc điểm:

### ✅ Ưu điểm:

- **Đa model**: Hỗ trợ 3 LLM phổ biến (GPT, Deepseek, Gemini)
- **Chat trực tiếp**: Tính năng chat không qua prompt template
- **So sánh**: Có thể so sánh kết quả từ nhiều model cùng lúc
- **Lịch sử**: Tự động lưu tất cả request/response
- **Linh hoạt**: Cho phép tùy chỉnh prompt hoàn toàn
- **Export**: Xuất kết quả ra file để phân tích
- **Hỗ trợ tiếng Việt**: Tự động phát hiện và yêu cầu trả lời bằng tiếng Việt
- **Markdown rendering**: Hiển thị kết quả đẹp với format đúng
- **Giao diện**: Đẹp, hiện đại, responsive, dễ sử dụng
- **Code**: Rõ ràng, có cấu trúc tốt (models/, utils/), dễ bảo trì
- **Không cần cài đặt phức tạp**: Chỉ cần Python và API keys

### 📚 Thông tin kỹ thuật:

- **Framework**: Flask 3.0.0
- **LLM Providers**: OpenAI, Deepseek, Google
- **Language**: Python 3.8+
- **Frontend**: HTML5, CSS3, JavaScript (ES6)
- **API**: RESTful API
- **Storage**: JSON file
- **Concurrency**: ThreadPoolExecutor cho so sánh model
- **Markdown**: Custom JavaScript parser

### ⚠️ Lưu ý:

- Cần internet và API keys để sử dụng
- Có chi phí khi sử dụng (theo pricing của từng provider)
- Kiểm tra rate limits và quotas trước khi sử dụng nhiều

### 📝 Cấu trúc Code:

Ứng dụng được tổ chức theo mô hình modular:

- **Models**: Mỗi LLM provider trong file riêng, dễ thêm/sửa
- **Utils**: Các hàm tiện ích được tách riêng, dễ test và tái sử dụng
- **Routes**: Business logic tập trung trong app.py
- **Templates**: Giao diện tách biệt, dễ chỉnh sửa

---

**Chúc bạn demo thành công! 🎉**

_Nếu có câu hỏi, vui lòng tham khảo file README.md hoặc kiểm tra lỗi trong console_
