from detecto import core, utils
import matplotlib.pyplot as plt
import cv2
import numpy as np

fname = 'test.jpg'

#Load mô hình
model = core.Model.load('id_card_4_corner.pth',['top_left', 'top_right', 'bottom_left', 'bottom_right'])

#Hàm gộp các box bị trùng nhau
def non_max_suppression_fast(boxes, labels, overlapThresh):
    #Nếu không có box nào thì trả về danh sách rỗng
    if len(boxes) == 0:
        return []

    #Chuyển các giá trị nguyên về thực
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    #Khởi tạo danh sách index được chọn  
    pick = []

    #Tạo mảng chứa các toạ độ x,y
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    #Tính diện tích các box
    area = (x2 - x1) * (y2 - y1)
    
    #Tạo mảng index được sắp xếp từ bé đến lớn của y2
    idxs = np.argsort(y2)

    #Lặp cho đến khi idxs rỗng
    while len(idxs) > 0:
        #Chọn từ cuối idxs
        last = len(idxs) - 1
        i = idxs[last]
        #Thêm vào danh sách được chọn
        pick.append(i)

        #Tìm (x,y) lớn nhất cho điểm bắt đầu của box, nhỏ nhất cho điểm kết thúc
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        #Tính kích thước của hộp giới hạn
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        #Tính tỉ lệ trùng lặp
        overlap = (w * h) / area[idxs[:last]]

        #Xoá các index có tỷ lệ trùng lớn hơn ngưỡng
        idxs = np.delete(idxs, np.concatenate(([last],
         np.where(overlap > overlapThresh)[0])))


    
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    #Gán lại nhãn cho box
    m = final_boxes[:,0]
    n = final_boxes[:,1]
    h = (max(m)+min(m))/2
    v = (max(n)+min(n))/2
    r,c = final_boxes.shape
    for i in range(r):
        if (final_boxes[i,0]<h):
                if (final_boxes[i,1]<v):
                        final_labels[i]='top_left'
                else:
                        final_labels[i]='bottom_left'
        else:
                if(final_boxes[i,1]<v):
                        final_labels[i]='top_right'
                else:
                        final_labels[i]='bottom_right'

    return final_boxes, final_labels

#Hàm lấy toạ độ tâm các box
def get_coordinates(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2

#Hàm cắt ảnh
def perspective_transoform(image, source_points):
    dest_points = np.float32([[0,0], [500,0], [500,300], [0,300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    output = cv2.warpPerspective(image, M, (500, 300))
    
    return output


#Đọc file và sử dụng mô hình dự đoán
image = utils.read_image(fname)
labels, boxes, scores = model.predict(image)
#Gộp box trùng
final_boxes, final_labels = non_max_suppression_fast(boxes.numpy(), labels, 0.15)

final_points = list(map(get_coordinates, final_boxes))
label_boxes = dict(zip(final_labels, final_points))
points = np.float32([
    label_boxes['top_left'], label_boxes['top_right'], label_boxes['bottom_right'], label_boxes['bottom_left']
])

output = perspective_transoform(image, points)
plt.imshow(output)
plt.show()