from ultralytics import YOLO

model = YOLO(r'models\best.pt')

result = model.predict(r'Input\Input.mp4', save = True)
print(result[0])
print('============')
for box in result[0].boxes:
    print(box)
    