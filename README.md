#  K-Means & K-NN (Implementation From Scratch)

> **Môn học:** Trí tuệ Nhân tạo (Artificial Intelligence)  
> **Sinh viên thực hiện:** Phan Thanh Trí  
> **MSSV:** 2001230977  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Algorithm-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  Giới thiệu (Introduction)

Dự án này tập trung vào việc **cài đặt thủ công (from scratch)** hai thuật toán học máy cơ bản và quan trọng nhất mà **không sử dụng các class mô hình có sẵn** (như `KMeans` hay `KNeighborsClassifier`) trong thư viện `scikit-learn`.

Mục tiêu chính của đồ án là thể hiện sự hiểu biết sâu sắc về toán học, quy trình tính toán ma trận và logic hoạt động bên trong của các thuật toán:

1.  **K-Means Clustering (Phân cụm):** Thuật toán học không giám sát (Unsupervised Learning) dùng để gom nhóm dữ liệu dựa trên đặc trưng.
2.  **K-Nearest Neighbors (K-NN):** Thuật toán học có giám sát (Supervised Learning) dùng để phân loại dữ liệu mới dựa trên khoảng cách với các điểm dữ liệu cũ.

---

##  Tính năng nổi bật (Key Features)

### 1. Thuật toán cốt lõi (Core Algorithms)
* **Pure Implementation:** Sử dụng `numpy` và `scipy.spatial.distance` để tính toán ma trận hiệu năng cao.
* **Step-by-Step Visualization:**
    * *K-Means:* Hiển thị tọa độ tâm cụm thay đổi sau mỗi vòng lặp để theo dõi quá trình hội tụ.
    * *K-NN:* In ra chi tiết Index, Khoảng cách và Nhãn của $K$ láng giềng gần nhất.

### 2. Tối ưu hóa tham số (Auto-Tuning)
Chương trình tích hợp sẵn module tự động tìm tham số $K$ tốt nhất:
* **K-Means:** Sử dụng **Phương pháp Khuỷu tay (Elbow Method)** kết hợp thuật toán hình học (Maximum Distance) để máy tính tự động chốt giá trị $K$ tối ưu.
* **K-NN:** Sử dụng phương pháp đánh giá **Độ lỗi (Error Rate)** trên tập Test để chọn $K$ có độ chính xác cao nhất.

### 3. Xử lý dữ liệu linh hoạt
* **Random Generation:** Tự sinh dữ liệu giả lập (`make_blobs`) để demo nhanh.
* **CSV Upload Support:** Hỗ trợ tải file `.csv` từ máy tính (tương thích tốt với Google Colab).
* **Data Cleaning:** Tự động phát hiện và loại bỏ dòng tiêu đề (header) hoặc các dữ liệu lỗi.

---

##  Cấu trúc mã nguồn

Dự án được chia thành 3 Cell (Ô lệnh) chính để dễ quản lý trên Google Colab:

| Cell | Chức năng chính | Mô tả |
| :--- | :--- | :--- |
| **Cell 1** | `Libraries` & `Classes` | Chứa hàm `get_data_input`, Class `KMeansClustering` và Class `KNNClassifier`. |
| **Cell 2** | `Optimization` | Chứa các hàm tìm $K$ tự động: `find_best_k_auto` và `find_best_k_knn`. |
| **Cell 3** | `Main Execution` | Chương trình chính: Gọi menu, tải dữ liệu, chạy tối ưu hóa và hiển thị kết quả. |

---

##  Hướng dẫn sử dụng (How to Run)

Để chạy chương trình trên Google Colab hoặc Jupyter Notebook, vui lòng thực hiện tuần tự:

1.  **Bước 1:** Chạy **Cell 1** để nạp các Class và thư viện cần thiết.
2.  **Bước 2:** Chạy **Cell 2** để nạp các hàm toán học tìm $K$ tối ưu.
3.  **Bước 3:** Chạy **Cell 3** để bắt đầu chương trình chính.
4.  **Bước 4:** Chọn nguồn dữ liệu theo menu hướng dẫn trên màn hình:
    * Nhập `1`: Sử dụng dữ liệu ngẫu nhiên (Demo nhanh).
    * Nhập `2`: Upload file CSV từ máy tính.

---

##  Định dạng dữ liệu đầu vào (Input Format)

Nếu bạn chọn tính năng **Upload File**, file `.csv` cần tuân thủ định dạng sau (Chương trình sẽ tự động xử lý/bỏ qua dòng tiêu đề nếu có):

### 1. Cho K-Means (Chỉ cần Features)
File chỉ cần chứa các cột đặc trưng số học (tọa độ điểm).
```csv
1.5, 2.5
3.2, 4.1
8.5, 9.0
...
### 2. Cho K-NN (Features + Label)
File cần có các cột đặc trưng và **cột cuối cùng bắt buộc phải là Nhãn (Label/Class)**.
```csv
Feature_1, Feature_2, Label
1.5,       2.5,       0
8.5,       9.0,       1
...

 Minh họa kết quả (Screenshots)Sau khi chạy, chương trình sẽ hiển thị các biểu đồ trực quan:

 Phương pháp Khuỷu tay (Elbow Method):
 Biểu đồ đường thể hiện chỉ số Inertia giảm dần khi $K$ tăng.
 Chương trình sẽ đánh dấu ngôi sao đỏ tại điểm gãy khúc (Elbow Point) - đó là số cụm $K$ tối ưu được chọn.
<img width="744" height="432" alt="image" src="https://github.com/user-attachments/assets/3e1d4a0b-da74-434e-a9f1-5f1dcf93ae28" />

 Kết quả phân cụm / Phân loại:
 Hiển thị biểu đồ Scatter Plot với các điểm dữ liệu được tô màu theo cụm/lớp.
 Hiển thị vị trí các Tâm cụm (Centroids) (đối với K-Means) bằng dấu X màu đỏ nổi bật.
 <img width="626" height="433" alt="image" src="https://github.com/user-attachments/assets/67922cc1-c6d1-4b72-a829-605367b616a1" />

 Hiển thị điểm dữ liệu mới cần dự đoán (đối với K-NN) và kết quả phân loại.
<img width="719" height="448" alt="image" src="https://github.com/user-attachments/assets/041049a0-1327-4880-b716-af58b9172e0d" />
<img width="673" height="427" alt="image" src="https://github.com/user-attachments/assets/dea00084-4ef7-4081-a720-30ec3c175a79" />
