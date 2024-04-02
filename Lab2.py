import torch, cv2, json, time, requests
import numpy as np


#Duration between each detection
detected_start = None
detected_duration = 2  # seconds

#LINE TOKEN
LINE_NOTIFY_TOKEN = 'BHb48WFbLe54GOTYYMfCqk16W509Sl8AiUj4HvRUbQt'

#Capture
vid = cv2.VideoCapture(0)

#Load Model
model = torch.hub.load('.', 'custom', path='best.pt', source='local', device='mps') 

#Create Detection Box
def plot_boxes(result_dict, frame):
    for ob in result_dict:
        rec_start = (int(ob['xmin']), int(ob['ymin']))
        rec_end = (int(ob['xmax']), int(ob['ymax']))
        color = (255, 0, 0)
        thickness = 3

        cv2.rectangle(frame, rec_start, rec_end, color, thickness)
        cv2.putText(frame, "%s %0.2f" % (ob['name'], ob['confidence']), rec_start, cv2.FONT_HERSHEY_DUPLEX, 2, color, thickness)
    return frame

#Detection
while(True):
    ret, frame = vid.read()
    results = model(frame)
    result_jsons = results.pandas().xyxy[0].to_json(orient="records")
    result_dict = json.loads(result_jsons)
    # print(result_dict)
    frame2 = plot_boxes(result_dict, frame)
    cv2.imshow('YOLO', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #Filter the desired detection name
    ob_detected = False
    for ob in result_dict:
        if ob['name'] == 'person' or ob['name'] == 'cell phone':
            ob_detected = True
            break

    if ob_detected:
        if detected_start is None:
            detected_start = time.time()
        else:
            current_time = time.time()
            if current_time - detected_start >= detected_duration:
                message = "Detected"
                image_path = "temp_detected_image.jpg"  
                cv2.imwrite(image_path, frame)  

                # Send message and image via Line Notify
                headers = {"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"}
                payload = {"message": message}
                files = {'imageFile': open(image_path, 'rb')}
                response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=payload, files=files)
                if response.status_code == 200:
                    print("detected message sent successfully!")
                else:
                    print("Error sending detected message.")
                
                detected_start = None  # Reset the detection time

    else:
        detected_start = None

vid.release()
cv2.destroyAllWindows()