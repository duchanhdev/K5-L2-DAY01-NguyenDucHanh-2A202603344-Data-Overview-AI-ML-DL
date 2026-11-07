# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Record này nói về cả bức ảnh (phân loại toàn ảnh - image classification), chứ không phải một vật thể riêng lẻ có gán tọa độ (như bài toán phát hiện đối tượng - object detection).
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Danh sách tên lớp đi kèm sẵn trong checkpoint / taxonomy của mô hình (cụ thể là tập dữ liệu chuẩn ImageNet-1K gồm 1.000 lớp cố định được nhúng sẵn trong trọng số của yolo11n-cls.pt)
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   class_id: Định dạng số hóa để lưu trữ, truy xuất cơ sở dữ liệu.
     class_name: Định dạng để cho con người đọc dễ dàng hơn.
     taxonomy_name: Đây là bộ mẫu để biết xem class_id đó có nguồn gốc ở đâu. Vì các taxonomy_name khác nhau có thể có cùng class_id nhưng chỉ đối tượng khác nhau
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline phải nêu rõ quy tắc chọn nhãn trong các trường hợp, có thể ưu tiên đối tượng chủ đạo: ưu tiên chọn lớp có vật thể chiếm diện tích lớn hoặc nó là tâm điểm thu hút.
6. Vì sao không được lấy `score` của model làm ground truth?

   Score chỉ là xác suất dự đoán chủ quan của mô hình ở thời điểm suy luận chứ không phải thực tế. Mô hình có thể sai hoặc tự tin thái quá.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Góc trên-trái ($x_{min}, y_{min}$): Nằm ở tọa độ $x = 385.33 pixel (tính từ mép trái ảnh) và $y = 69.24 pixel (tính từ mép trên ảnh xuống).
     Góc dưới-phải ($x_{max}, y_{max}$): Nằm ở tọa độ $x = 498.92 pixel và $y = 348.92 pixel.
     Kích thước hộp: Khung chứa vật thể có chiều rộng (bbox_width) là 113.58 pixel và chiều cao (bbox_height) là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Ngưỡng 0.20: Phát hiện 80 vật thể
     Ngưỡng 0.60: Phát hiện 25 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Khi hạ ngưỡng độ tin cậy, số lượng hộp dự đoán tăng lên kéo theo nhiều hộp nhiễu, hộp trùng lặp (nhiều bounding box cùng ôm một vật) hoặc hộp nhận diện sai. Reviewer sẽ phải tốn nhiều thời gian và công sức hơn để:
   + Kiểm tra, rà soát và xóa bỏ các hộp dự đoán sai (False Positives).
   + Thực hiện gộp/lọc bỏ các hộp bị trùng lặp đối tượng.
   + Bổ sung hoặc điều chỉnh tọa độ cho các hộp bị thiếu sót.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Quy tắc bám sát viền đối tượng (Tight Bounding Box):
   + Hộp giới hạn phải ôm khít toàn bộ ranh giới trực quan của vật thể (từ điểm cực trái, cực phải, cực trên đến cực dưới của đối tượng).
   + Không để khoảng trống dư thừa quá lớn xung quanh vật thể, đồng thời không được cắt xén phạm vi thân chính của vật thể cần gán nhãn.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Xác định vật thể bị che khuất hoặc cắt mép:
   + Trên ảnh kitchen, các vật dụng như các dãy chảo/nồi treo trên vách tường phía trên hoặc các vật thể nhỏ (như các bowl và đồ dùng trên bàn) thường bị che khuất một phần lẫn nhau hoặc bị cắt mép ở ranh giới khung hình.  
     Hình thức xử lý (Viết vào guideline hay Escalation):
   + Sẽ viết thành quy tắc chuẩn trực tiếp trong guideline chứ không đưa lên hỏi Lab Coach (escalation).
     Vì sao:
   + Hiện tượng vật thể bị che khuất (occlusion) hay cắt mép (truncation) là tình huống cực kỳ phổ biến và mang tính hệ thống trong mọi dự án gán nhãn dữ liệu thị giác máy tính.
   + Thay vì mất thời gian hỏi lại cho từng trường hợp nhỏ lẻ (gây nghẽn dòng công việc), dự án cần chuẩn hóa sẵn các tiêu chí trong guideline (ví dụ: quy định vật thể bị che khuất trên 50% thì bỏ qua hay vẫn gán nhãn, cách ước lượng khung bounding box cho phần bị khuất) để người gán nhãn có căn cứ tự xử lý đồng bộ và nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0],[443.0, 72.0],[442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Bounding Box (Khung bao): Chỉ cung cấp hình chữ nhật giới hạn vùng chứa tổng quát của đối tượng (dựa trên tọa độ góc trên-trái và dưới-phải), kéo theo cả các vùng nền xung quanh và không thể hiện được hình dáng thực tế của vật thể.
     Polygon (Đa giác/Mask): Cung cấp đường viền chi tiết, ôm sát theo hình dạng, góc cạnh, đường cong hoặc các phần lồi lõm của vật thể ở cấp độ pixel, giúp phân định rõ ràng đâu là đối tượng và đâu là nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   instance_id dùng để định danh duy nhất (unique identifier) cho từng cá thể riêng biệt trong một bức ảnh.
     Class ID chỉ định danh loại danh mục, nghĩa là nó cho biết đối tượng là gì chứ không phân biệt các cá thể cùng loại với nhau. Cần instance_id để định danh từng đối tượng, kể cả chúng cùng loại.
     Tracking ID được dùng để theo dõi một đối tượng xuyên suốt qua nhiều khung hình (frames) liên tiếp trong video. Trong ảnh tĩnh dùng instance_id
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Đường biên mask phải bám sát ranh giới thực tế giữa vật thể và nền. Tại các điểm chuyển đổi màu sắc hoặc biên mờ, người gán nhãn cần đặt điểm polygon ở tâm của dải pixel chuyển tiếp. Với các phần bị che khuất, tuân thủ nguyên tắc nội suy theo hình học chuẩn của vật thể dựa trên tài liệu hướng dẫn.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Có 1 person đã bị che ngoài bức ảnh nên chỉ có 1 bàn tay trong hình.
     Sử dụng Guideline (Tài liệu hướng dẫn gán nhãn)
     Các trường hợp chồng lấn (overlap), biên mờ hoặc vật thể chạm nhau là những tình huống chuẩn hóa đã được định nghĩa cách xử lý trong tài liệu dự án (ví dụ: ưu tiên vật che khuất phía trước hoặc quy tắc cắt lớp pixel). Chỉ khi gặp trường hợp ngoại lệ hoàn toàn mới cần Escalation (leo thang báo cáo cấp trên).

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                     | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                     | Annotator làm gì?                                                                     | Reviewer xem gì?                                                                                  |
| --------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Nhãn dạng text/ID toàn cục cho cả ảnh (Ví dụ: `"kitchen"`).       |                                                                        | Gắn nhãn danh mục tổng quát phù hợp nhất với nội dung chính của bức ảnh.              | Kiểm tra xem nhãn có khớp với ngữ cảnh tổng thể của ảnh không; nếu sai, yêu cầu **rework**.       |
| Phát hiện vật thể     | Tọa độ khung chữ nhật `[xmin, ymin, xmax, ymax]` kèm tên class.   | Vật thể bị cắt cụt cạnh khung hình hoặc chồng lấn dày đặc lên nhau.    | Vẽ bounding box bao trọn toàn bộ vật thể, đảm bảo không bỏ sót.                       | Kiểm tra độ khít của khung chữ nhật, tránh tình trạng box quá lớn hoặc quá lỏng.                  |
| Instance segmentation | Tập hợp tọa độ đa giác (`polygon_xy`) kèm `instance_id` và class. | Biên vật thể mờ,hoặc bị che khuất một phần. Như chén ly che khuất nhau | Chấm các điểm polygon bám sát đường viền pixel thực tế và quản lý đúng `instance_id`. | Kiểm tra đường biên polygon có bị lẹm ra ngoài nền hoặc bỏ sót chi tiết, ID có bị nhầm lẫn không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Tuân thủ tuyệt đối tính bảo mật của tập dữ liệu lab, không sao chép, chia sẻ, hay sử dụng dữ liệu vào mục đích khác ngoài phạm vi học tập và nghiên cứu của bài lab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng lại và báo cáo ngay cho Giảng viên hướng dẫn hoặc Quản lý dự án (Project Manager).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
