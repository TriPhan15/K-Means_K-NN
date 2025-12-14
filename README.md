#  K-Means & K-NN (From Scratch) 

> **Môn học:** Trí tuệ Nhân tạo (Artificial Intelligence)  
> **Sinh viên:** [Phan Thanh Trí]  
> **MSSV:** [2001230977]
>
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Algorithm-orange)
![Status](https://img.shields.io/badge/Status-Educational-green)

##  Giới thiệu
Dự án này tập trung vào việc **cài đặt thủ công (from scratch)** hai thuật toán học máy cơ bản nhất mà không sử dụng các class mô hình có sẵn trong thư viện `scikit-learn`. Mục tiêu chính là hiểu rõ bản chất toán học và luồng hoạt động của thuật toán.

1.  **K-Means Clustering:** Thuật toán học không giám sát (Unsupervised Learning) dùng để phân cụm dữ liệu.
2.  **K-Nearest Neighbors (K-NN):** Thuật toán học có giám sát (Supervised Learning) dùng để phân loại dựa trên khoảng cách.

##  Tính năng nổi bật
* **Step-by-Step Visualization:** * **K-Means:** Hiển thị tọa độ tâm cụm cũ và mới sau mỗi vòng lặp để theo dõi quá trình hội tụ.
    * **K-NN:** In ra chi tiết khoảng cách và nhãn của K láng giềng gần nhất trước khi đưa ra quyết định bầu chọn.
* **Đa dạng nguồn dữ liệu:**
    * Tự sinh dữ liệu ngẫu nhiên (Random Data Generation).
    * Hỗ trợ tải lên file `.csv` từ máy tính (Tối ưu cho Google Colab).
* **Giao diện tương tác (CLI):** Menu điều khiển rõ ràng, tự động làm sạch màn hình giúp dễ quan sát kết quả.
* **Trực quan hóa:** Vẽ biểu đồ Scatter Plot hiển thị rõ ràng các cụm (Clusters) và biên giới phân loại.

##  Yêu cầu cài đặt
Đảm bảo bạn đã cài đặt Python và các thư viện cần thiết:

```bash
pip install numpy matplotlib pandas scipy scikit-learn

