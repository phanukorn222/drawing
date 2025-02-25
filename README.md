# Hand Drawing with Mediapipe

โปรเจคนี้ใช้ **Mediapipe** จาก Google และ **OpenCV** ในการสร้างแอปพลิเคชันที่สามารถวาดรูปด้วยมือ โดยจะใช้การตรวจจับการเคลื่อนไหวของนิ้วมือ (hand landmarks) ผ่านกล้องเว็บแคมและสามารถบันทึกภาพที่วาดได้หรือเคลียร์กระดานวาดได้ด้วยการชูนิ้วมือในพื้นที่ที่กำหนด

## ฟีเจอร์

- วาดรูปบนกระดานโดยใช้การชูนิ้วชี้
- หยุดการวาดเมื่อชูนิ้วชี้และนิ้วกลาง
- สามารถบันทึกภาพที่วาดได้โดยคลิกในโซน "Save"
- สามารถเคลียร์กระดานวาดได้โดยคลิกในโซน "Clear"

## การติดตั้ง

โปรเจคนี้ต้องการ **Python** และไลบรารีบางตัวที่สามารถติดตั้งได้โดยใช้คำสั่ง:

### ติดตั้งไลบรารีที่จำเป็น

```bash
pip install opencv-python mediapipe numpy
