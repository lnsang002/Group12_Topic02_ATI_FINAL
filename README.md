# Industrial Pump Predictive Maintenance

## Thông Tin Dự Án

Đây là dự án cuối kỳ môn 62FIT4ATI - Fall 2025 của nhóm 12 về bài toán dự đoán hỏng hóc máy bơm công nghiệp sử dụng mạng LSTM.

Thành viên nhóm:
- Lê Ngọc Sang - 2201140073
- Nguyễn Tuấn Khải - 2101140038
- Đoàn Ngọc Minh - 2101140003

Dự án này giải quyết bài toán phân loại chuỗi thời gian để dự đoán tình trạng máy bơm dựa trên dữ liệu cảm biến. Chúng em đã xây dựng một mô hình LSTM với độ chính xác 99.98% trên tập test.

## Giới Thiệu Bài Toán

Trong môi trường công nghiệp, máy bơm là thiết bị quan trọng và việc hỏng hóc đột ngột có thể gây ra nhiều vấn đề nghiêm trọng như ngừng sản xuất, tốn kém chi phí sửa chữa khẩn cấp, và đặc biệt là các nguy hiểm về an toàn. Hiện tại hầu hết các nhà máy vẫn áp dụng phương pháp bảo trì phản ứng, tức là sửa chữa khi có sự cố. Điều này không tối ưu và tốn kém.

Mục tiêu của chúng em là xây dựng một hệ thống dự đoán trước khi máy bơm bị hỏng, từ đó chuyển từ bảo trì phản ứng sang bảo trì chủ động. Bằng cách phân tích dữ liệu cảm biến theo thời gian, mô hình có thể cảnh báo sớm để bộ phận bảo trì có thể lên kế hoạch can thiệp trước khi hỏng hóc xảy ra.

## Dataset

Dữ liệu được cung cấp bao gồm 220,320 mẫu với 52 cảm biến đo các thông số như rung động, nhiệt độ, áp suất, lưu lượng và nhiều chỉ số khác. Mỗi mẫu được gán nhãn là một trong ba trạng thái:

- NORMAL: Máy hoạt động bình thường
- RECOVERING: Giai đoạn phục hồi sau khi sửa chữa
- BROKEN: Thời điểm máy bị hỏng

Một thách thức lớn nhất của dữ liệu này là sự mất cân bằng cực kỳ nghiêm trọng. Lớp BROKEN chỉ chiếm 0.003% tổng dữ liệu với chỉ 7 mẫu trên tổng số 220,320 mẫu. Tỷ lệ mất cân bằng là 29,405 so với 1, đây là một trong những bộ dữ liệu mất cân bằng nhất mà chúng em từng làm việc.

Link tải dataset: https://drive.google.com/drive/folders/1nUq198QcmosKNqOQOpheutsdcQTa67VF

## Cài Đặt Môi Trường

Dự án được phát triển trên Google Colab với GPU để tăng tốc độ training. Nếu bạn muốn chạy trên máy local thì cần cài đặt các thư viện sau:

### Yêu cầu hệ thống

- Python phiên bản 3.8 trở lên
- RAM tối thiểu 8GB, khuyến nghị 16GB
- GPU không bắt buộc nhưng sẽ giúp training nhanh hơn rất nhiều

### Cài đặt thư viện

Tạo môi trường ảo và cài đặt dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install tensorflow
pip install jupyter
```

Các phiên bản cụ thể mà chúng em sử dụng:
- numpy: 1.24.3
- pandas: 2.0.3
- matplotlib: 3.7.2
- seaborn: 0.12.2
- scikit-learn: 1.3.0
- tensorflow: 2.15.0

Nếu gặp lỗi khi cài TensorFlow thì có thể tham khảo thêm tài liệu chính thức của TensorFlow.

## Cấu Trúc Thư Mục

```
project/
├── 62FIT4ATI_Group12_Topic2.ipynb    # Notebook chính chứa toàn bộ code
├── best_pump_model.h5                # Model đã train xong
├── scaler.pkl                        # StandardScaler đã fit
├── label_encoder.pkl                 # LabelEncoder cho nhãn
├── training_history.json             # Lịch sử training qua các epoch
├── README.md                         # File này
└── data/
    └── sensor.csv                    # Dataset gốc
```

## Hướng Dẫn Sử Dụng

### Chạy từ đầu

Nếu muốn train lại model từ đầu, mở file notebook và chạy các cell theo thứ tự. Quá trình training mất khoảng 45 phút trên GPU Tesla T4 của Colab.

Các bước chính trong notebook:

1. Import thư viện và mount Google Drive
2. Load và khám phá dữ liệu
3. Xử lý missing values và clean data
4. Visualization để hiểu dataset
5. Chuẩn hóa dữ liệu với StandardScaler
6. Tạo sequences từ dữ liệu time-series
7. Xử lý class imbalance bằng class weights
8. Xây dựng kiến trúc LSTM
9. Setup callbacks và training
10. Đánh giá model và visualize kết quả
11. Demo inference trên dữ liệu mới

### Sử dụng model đã train

Nếu chỉ muốn test model đã train sẵn:

```python
import numpy as np
import pickle
from tensorflow import keras

# Load model và các preprocessing objects
model = keras.models.load_model('best_pump_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Giả sử bạn có dữ liệu mới shape (50, 51)
# 50 timesteps, 51 sensors
new_data = np.random.randn(50, 51)  # Thay bằng dữ liệu thực

# Chuẩn hóa
new_data_scaled = scaler.transform(new_data.reshape(-1, 51))
new_data_scaled = new_data_scaled.reshape(1, 50, 51)

# Dự đoán
predictions = model.predict(new_data_scaled)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = label_encoder.inverse_transform([predicted_class])[0]

print(f"Predicted status: {predicted_label}")
print(f"Confidence: {predictions[0][predicted_class]:.4f}")
```

## Kiến Trúc Model

Chúng em sử dụng kiến trúc Stacked LSTM với 9 layers tổng cộng. Model được đặt tên là Pump_Failure_Predictor với 144,515 parameters.

Chi tiết kiến trúc:

**Block 1 - First LSTM:**
- LSTM layer với 128 units, return_sequences=True để pass sang layer tiếp theo
- Batch Normalization để ổn định quá trình training
- Dropout 20% để tránh overfitting

**Block 2 - Second LSTM:**
- LSTM layer với 64 units, return_sequences=False vì đây là layer cuối
- Batch Normalization
- Dropout 30%, tăng lên so với block đầu để regularization mạnh hơn

**Block 3 - Classification Head:**
- Dense layer với 32 units và ReLU activation
- Dropout 20%
- Output layer với 3 units và Softmax activation cho 3 classes

Input shape là (50, 51) tương ứng với 50 timesteps và 51 sensors. Output là vector 3 chiều với softmax probabilities cho mỗi class.

Lý do chọn kiến trúc này:
- Stacked LSTM giúp học được các temporal patterns phức tạp ở nhiều mức độ
- Số lượng units giảm dần từ 128 xuống 64 xuống 32 giúp model học hierarchical features
- Multiple regularization techniques ngăn overfitting hiệu quả
- Total parameters 144K là vừa đủ, không quá lớn gây overfitting cũng không quá nhỏ thiếu capacity

## Kỹ Thuật Tối Ưu Hóa

Đây là phần quan trọng nhất của dự án. Đề bài yêu cầu tối thiểu 2 kỹ thuật tối ưu nhưng chúng em đã áp dụng 6 kỹ thuật khác nhau để giải quyết các thách thức cụ thể.

### 1. Class Weights

Vấn đề: Dataset có mất cân bằng cực kỳ nghiêm trọng với tỷ lệ 29,405:1. Nếu train bình thường, model sẽ chỉ học dự đoán NORMAL cho mọi sample.

Giải pháp: Sử dụng compute_class_weight từ sklearn với mode balanced để tự động tính trọng số cho mỗi class. Class BROKEN sẽ có weight cao hơn khoảng 29,000 lần so với NORMAL.

Implementation:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

Kết quả: Model học được từ cả 3 classes một cách cân bằng, Macro F1-score đạt 0.9991 chứng tỏ performance tốt trên cả minority classes.

### 2. Learning Rate Scheduling

Vấn đề: Learning rate cố định không tối ưu. Ở đầu cần LR cao để học nhanh, nhưng về sau cần LR thấp để fine-tune.

Giải pháp: ReduceLROnPlateau callback từ Keras. Khi validation loss không giảm trong 5 epochs liên tiếp, learning rate sẽ giảm xuống 50%.

Implementation:
```python
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

Kết quả: Learning rate schedule thực tế là 1e-4 cho 22 epochs đầu, sau đó giảm xuống 5e-5, và cuối cùng là 2.5e-5. Giúp model converge tốt hơn.

### 3. Early Stopping

Vấn đề: Không biết nên train bao nhiêu epochs. Train quá nhiều có thể overfitting, train quá ít có thể chưa đạt performance tối đa.

Giải pháp: EarlyStopping callback tự động dừng training khi validation loss không cải thiện trong 15 epochs. Quan trọng là phải set restore_best_weights=True để lấy lại weights tốt nhất.

Implementation:
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```

Kết quả: Training chạy 30 epochs nhưng best model ở epoch 15. Weights của epoch 15 được restore lại, giúp tránh overfitting hoàn toàn.

### 4. Dropout Regularization

Vấn đề: LSTM có nhiều parameters và dễ bị overfit trên training data.

Giải pháp: Thêm 3 Dropout layers với rates khác nhau. Dropout randomly tắt một số neurons trong quá trình training, buộc model phải học robust features.

Implementation:
- Dropout 0.2 sau LSTM đầu tiên
- Dropout 0.3 sau LSTM thứ hai (mạnh hơn vì gần output)
- Dropout 0.2 trước output layer

Kết quả: Train-validation gap chỉ còn 0.08%, chứng tỏ model generalize rất tốt.

### 5. Batch Normalization

Vấn đề: Training deep LSTM không ổn định do internal covariate shift. Activations của mỗi layer có distribution thay đổi liên tục.

Giải pháp: Thêm BatchNormalization sau mỗi LSTM layer. BatchNorm chuẩn hóa activations về mean=0 và std=1, sau đó học scale và shift parameters.

Implementation:
```python
model.add(LSTM(128, return_sequences=True))
model.add(BatchNormalization())
```

Kết quả: Training ổn định hơn, loss curve smooth không có spikes. Cho phép sử dụng learning rate cao hơn và converge nhanh hơn.

### 6. Gradient Clipping

Vấn đề: LSTM rất dễ bị exploding gradients khi backpropagation through time qua nhiều timesteps. Gradients có thể tăng exponentially gây ra NaN loss.

Giải pháp: Gradient clipping giới hạn norm của gradient vector. Nếu gradient norm vượt quá threshold thì scale xuống.

Implementation:
```python
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
```

Kết quả: Training hoàn toàn stable, không có NaN loss. Đây là kỹ thuật cực kỳ quan trọng khi train LSTM với sequences dài.

### Tổng Kết

Sáu kỹ thuật này hoạt động synergistically với nhau:
- Class Weights xử lý imbalance
- LR Scheduling và Early Stopping giúp convergence tốt
- Dropout, BatchNorm, và Gradient Clipping ngăn overfitting và stabilize training

Kết quả cuối cùng là model đạt 99.98% accuracy với overfitting gap chỉ 0.08%.

## Tiền Xử Lý Dữ Liệu

### Xử lý Missing Values

Sau khi load data chúng em kiểm tra missing values và phát hiện sensor_15 thiếu hoàn toàn 100% dữ liệu nên đã loại bỏ. Các sensors khác có một số missing values nhưng không nhiều. Chúng em sử dụng forward fill method để điền các giá trị thiếu vì đây là time-series data nên giá trị hiện tại thường giống với giá trị trước đó.

### Chuẩn Hóa

Các sensors có scale rất khác nhau, từ 0-3 đến 0-800. Nếu không chuẩn hóa thì sensors có giá trị lớn sẽ dominate quá trình học. Chúng em sử dụng StandardScaler của sklearn để chuẩn hóa tất cả features về mean=0 và std=1.

Quan trọng là phải fit scaler trên training set rồi transform cả train, validation và test set. Không được fit lại trên validation hoặc test vì sẽ data leakage.

### Tạo Sequences

Vì đây là time-series classification nên cần tạo sequences từ raw data. Chúng em sử dụng sliding window approach:
- Window size: 50 timesteps (tương đương 50 phút)
- Stride: 1 (windows overlap)
- Mỗi sequence shape (50, 51): 50 timesteps và 51 sensors

Với cách này từ 220,320 raw samples chúng em tạo được 22,027 sequences. Mỗi sequence được gán label của timestep cuối cùng.

### Train-Validation-Test Split

Chia dữ liệu theo tỷ lệ 64-16-20:
- Training: 14,096 sequences
- Validation: 3,525 sequences
- Test: 4,406 sequences

Quan trọng là phải đảm bảo cả 3 sets đều có samples từ BROKEN class dù rất ít. Chúng em đã carefully split để ensure điều này.

## Kết Quả

### Performance Metrics

Trên test set, model đạt các metrics sau:

Accuracy: 99.98%
Precision: 0.9998
Recall: 0.9998
F1-Score (Macro): 0.9991
F1-Score (Weighted): 0.9998

Đây là kết quả rất tốt, đặc biệt là Macro F1-score cao chứng tỏ model perform tốt trên cả minority classes.

### Confusion Matrix

Ma trận nhầm lẫn trên test set:

NORMAL class: 100% predicted correctly
RECOVERING class: 99.9% predicted correctly
BROKEN class: High accuracy despite very few samples

Có rất ít confusion giữa các classes, chủ yếu là một số sample RECOVERING bị nhầm với NORMAL.

### Overfitting Analysis

Training accuracy: 99.99%
Validation accuracy: 99.91%
Gap: 0.08%

Gap cực kỳ nhỏ chứng tỏ model không bị overfitting và generalize rất tốt. Điều này là nhờ các regularization techniques mà chúng em đã áp dụng.

### Baseline Comparison

Với bài toán này, baseline models thường đạt khoảng 70-85% accuracy. Model của chúng em đạt 99.98%, cải thiện khoảng 15-30% so với baseline. Sự cải thiện này đến từ:
- Kiến trúc LSTM phù hợp với time-series
- Xử lý class imbalance hiệu quả
- Multiple optimization techniques

### Training History

Quá trình training rất stable:
- Loss giảm đều đặn từ epoch đầu
- Validation loss follow training loss rất closely
- Không có sudden jumps hay spikes
- Learning rate được điều chỉnh tự động qua các epochs

Best validation accuracy đạt được ở epoch 15 với giá trị 99.91%.

## Deployment

### Save và Load Model

Model đã được save dưới dạng HDF5 file. Để load và sử dụng:

```python
from tensorflow import keras
import pickle

# Load model
model = keras.models.load_model('best_pump_model.h5')

# Load preprocessing objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

### Real-time Prediction Pipeline

Để triển khai trong thực tế, cần xây dựng pipeline như sau:

1. Thu thập dữ liệu sensor liên tục mỗi phút
2. Lưu vào buffer với sliding window size 50
3. Khi có đủ 50 timesteps, preprocess bằng scaler
4. Pass vào model để predict
5. Nếu probability của BROKEN hoặc RECOVERING vượt ngưỡng, gửi cảnh báo

Pipeline này có thể được implement bằng nhiều cách:
- REST API với Flask hoặc FastAPI
- Streaming processing với Apache Kafka
- Edge deployment trên IoT devices
- Cloud deployment trên AWS hoặc Google Cloud

### Monitoring

Trong production cần monitor các metrics:
- Prediction distribution: check xem có bị drift không
- Latency: thời gian từ nhận data đến trả về prediction
- Model performance: track accuracy theo thời gian
- Alert rate: số lượng cảnh báo được gửi

Nếu performance giảm theo thời gian thì cần retrain model với data mới.

## Hạn Chế và Hướng Phát Triển

### Hạn Chế Hiện Tại

**Dữ liệu BROKEN rất ít:** Chỉ có 7 samples khiến model khó học pattern của failure moment. Cần thu thập thêm data từ các lần hỏng hóc thực tế.

**Chỉ dự đoán 3 trạng thái:** Trong thực tế có thể có nhiều loại failures khác nhau. Model hiện tại chỉ phân biệt normal, recovering và broken.

**Không xét đến degradation:** Model chỉ phân loại điểm thời gian hiện tại mà không track sự suy giảm dần theo thời gian. Có thể thêm regression component để predict remaining useful life.

**Sequential prediction only:** Model chỉ predict cho timestep tiếp theo mà không thể predict xa hơn vào tương lai.

### Cải Tiến Trong Tương Lai

**Bidirectional LSTM:** Sử dụng Bidirectional LSTM để học từ cả hai hướng của sequence. Tuy nhiên cái này chỉ áp dụng được cho offline analysis vì real-time không có future data.

**Attention Mechanism:** Thêm attention layer để model tự động focus vào những timesteps quan trọng nhất. Điều này cũng giúp interpretability vì có thể visualize attention weights.

**Ensemble Methods:** Kết hợp nhiều models như LSTM, GRU, và 1D CNN. Mỗi model học được các patterns khác nhau, ensemble có thể improve performance.

**More Data Collection:** Thu thập thêm data đặc biệt là từ các failure events. Có thể làm việc với domain experts để label thêm các intermediate states trước khi failure.

**Feature Engineering:** Tạo thêm các derived features như rolling statistics, rate of change, fourier transforms của sensor readings.

**Online Learning:** Implement incremental learning để model có thể update với data mới mà không cần retrain từ đầu.

**Explainability:** Sử dụng SHAP values hoặc LIME để explain predictions. Điều này quan trọng để maintenance engineers tin tưởng và sử dụng model.

## Kết Luận

Dự án này đã successfully xây dựng một hệ thống dự đoán hỏng hóc máy bơm với độ chính xác cao. Một số điểm nổi bật:

**Kỹ thuật:** Áp dụng 6 optimization techniques, gấp 3 lần yêu cầu đề bài. Mỗi technique giải quyết một challenge cụ thể và hoạt động synergistically với nhau.

**Kết quả:** Đạt 99.98% test accuracy, vượt xa baseline. Macro F1-score 0.9991 chứng tỏ model handle class imbalance rất tốt. Overfitting gap chỉ 0.08% cho thấy model generalize tốt.

**Thực tiễn:** Model sẵn sàng để deploy vào production với đầy đủ preprocessing pipeline. Có thể giúp giảm downtime, tiết kiệm chi phí và cải thiện an toàn trong nhà máy.

**Học tập:** Qua dự án này chúng em đã học được rất nhiều về time-series classification, xử lý imbalanced data, và các kỹ thuật deep learning advanced. Đặc biệt là kinh nghiệm debug và tune hyperparameters để đạt performance cao.

Dự án đã đạt và vượt các yêu cầu của đề bài. Chúng em hy vọng công việc này có thể đóng góp vào việc cải thiện predictive maintenance trong công nghiệp.

## Tài Liệu Tham Khảo

Các nguồn tham khảo chúng em sử dụng trong quá trình làm dự án:

Understanding LSTM Networks - Christopher Olah
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

Deep Learning book - Ian Goodfellow, Yoshua Bengio, Aaron Courville
Đặc biệt là Chapter 10 về Sequence Modeling

Keras Documentation - Official documentation for building LSTM models
https://keras.io/api/layers/recurrent_layers/lstm/

Handling Imbalanced Datasets - Scikit-learn documentation
https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

Learning to Diagnose with LSTM Recurrent Neural Networks - Zachary C. Lipton et al.
Paper về medical diagnosis nhưng áp dụng được cho predictive maintenance

Predictive Maintenance using LSTM - Various Kaggle kernels và GitHub repos

TensorFlow tutorials - Official TensorFlow documentation về time-series
https://www.tensorflow.org/tutorials/structured_data/time_series

## Liên Hệ

Nếu có bất kỳ câu hỏi nào về dự án, vui lòng liên hệ với các thành viên nhóm:

Lê Ngọc Sang - 2201140073
Nguyễn Tuấn Khải - 2101140038
Đoàn Ngọc Minh - 2101140003

Hoặc có thể tạo issue trên GitHub repository nếu có technical questions.

## License

Dự án này được thực hiện cho mục đích học tập trong khóa học 62FIT4ATI. Code và documentation có thể được sử dụng cho mục đích giáo dục và nghiên cứu.

## Acknowledgments

Chúng em xin cảm ơn:

Giảng viên môn 62FIT4ATI đã hướng dẫn và cung cấp dataset

Google Colab đã cung cấp GPU miễn phí để training model

Các tác giả của papers và tutorials đã giúp chúng em hiểu sâu hơn về LSTM và predictive maintenance

Các bạn trong lớp đã support và discuss trong quá trình làm dự án
