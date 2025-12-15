#  K-Means & K-NN 

> **Môn học:** Thực Hành Trí tuệ Nhân tạo (Artificial Intelligence)  
> **Sinh viên thực hiện:** Phan Thanh Trí  
> **MSSV:** 2001230977  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Algorithm-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  Giới thiệu (Introduction)

Đồ án này tập trung vào việc **cài đặt thủ công (from scratch)** hai thuật toán học máy cơ bản và quan trọng nhất mà **không sử dụng các class mô hình có sẵn** (như `KMeans` hay `KNeighborsClassifier`) trong thư viện `scikit-learn`.

Mục tiêu chính của đồ án là thể hiện sự hiểu biết sâu sắc về toán học, quy trình tính toán ma trận và logic hoạt động bên trong của các thuật toán:

1.  **K-Means Clustering (Phân cụm):** Thuật toán học không giám sát (Unsupervised Learning) dùng để gom nhóm dữ liệu dựa trên đặc trưng.
2.  **K-Nearest Neighbors (K-NN):** Thuật toán học có giám sát (Supervised Learning) dùng để phân loại dữ liệu mới dựa trên khoảng cách với các điểm dữ liệu cũ.

---
##  Cơ sở lý thuyết (Theoretical Basis)

Dự án được xây dựng dựa trên các nền tảng toán học đại số tuyến tính và thống kê cơ bản:

### 1. K-Means Clustering
Thuật toán K-Means hoạt động dựa trên việc tối ưu hóa hàm mục tiêu (Objective Function) để phân chia dữ liệu $X$ thành $K$ cụm $C_1, ..., C_k$.

* **Hàm mất mát (Loss Function / Inertia):**
    Mục tiêu là tìm các tâm cụm $\mu$ sao cho tổng bình phương khoảng cách từ các điểm đến tâm cụm của nó là nhỏ nhất:
    
    $$J = \sum_{j=1}^{K} \sum_{i=1}^{n} ||x_i^{(j)} - \mu_j||^2$$
    
    *Trong đó:*
    * $x_i^{(j)}$: Điểm dữ liệu thứ $i$ thuộc cụm $j$.
    * $\mu_j$: Tâm (centroid) của cụm $j$.
    * $||...||^2$: Bình phương khoảng cách Euclidean L2.

* **Quy trình lặp (Iterative Optimization):**
    1.  **Gán nhãn (Cluster Assignment):** Mỗi điểm $x^{(i)}$ được gán cho cụm $k$ có tâm $\mu_k$ gần nhất:
        $$c^{(i)} := \text{arg}\min_k ||x^{(i)} - \mu_k||^2$$
    2.  **Cập nhật tâm (Move Centroids):** Tính lại vị trí tâm mới bằng trung bình cộng tọa độ các điểm trong cụm:
        $$\mu_k := \frac{1}{|C_k|} \sum_{x \in C_k} x$$

### 2. K-Nearest Neighbors (K-NN)
K-NN là thuật toán dựa trên vùng lân cận (Instance-based learning). Việc phân loại dựa trên khoảng cách giữa điểm dữ liệu mới $x_{new}$ và tập dữ liệu huấn luyện $S$.

* **Khoảng cách Euclidean (Euclidean Distance):**
    Độ tương đồng giữa hai vector $A$ và $B$ trong không gian $n$ chiều:
    
    $$d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$

* **Cơ chế bầu chọn (Majority Voting):**
    Giả sử $N_k$ là tập hợp $K$ điểm láng giềng gần nhất của $x_{new}$. Nhãn dự đoán $\hat{y}$ được quyết định bởi lớp chiếm đa số:
    
    $$\hat{y} = \text{arg}\max_v \sum_{(x_i, y_i) \in N_k} I(y_i = v)$$
    
    *Trong đó:* $I(.)$ là hàm chỉ thị (Indicator function), trả về 1 nếu $y_i$ trùng với nhãn $v$, ngược lại là 0.
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
