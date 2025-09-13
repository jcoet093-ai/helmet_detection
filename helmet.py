import cv2

from ultralytics import YOLO

model=YOLO("helmet (2).pt")

img=cv2.imread("helmet_img.jpg")
results=model(img)
count=0


for r in results:
    for box in r.boxes:
        cls=int(box.cls[0])
        label=model.names[cls]
        conf=float(box.conf[0])

        if conf>0.5:
            count += 1
            x,y,w,h = map(int,box.xyxy[0])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,f"{label}{count}:{conf:.2f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)



cv2.imshow("results",img )
cv2.waitKey(0)
cv2.destroyAllWindows()

