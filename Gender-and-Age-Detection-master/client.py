import cv2
import aiohttp
import asyncio
import numpy as np
from threading import Thread
from queue import Queue, Empty

SERVER_URL = "http://127.0.0.1:8000/detect/"
FRAME_QUEUE_SIZE = 2  # Keep only latest frames

class AsyncClient:
    def __init__(self):
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.latest_results = {}
        self.running = True

    async def process_frame(self, frame: np.ndarray):
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    SERVER_URL,
                    data={"image": img_encoded.tobytes()},
                    timeout=0.5  # Short timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.latest_results = data.get("results", [])
            except Exception as e:
                pass  # Silently drop timed out requests

    def worker(self):
        asyncio.run(self._worker())

    async def _worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get_nowait()
                await self.process_frame(frame)
            except Empty:
                await asyncio.sleep(0.01)

    def start(self):
        Thread(target=self.worker, daemon=True).start()

    def stop(self):
        self.running = False

def main():
    client = AsyncClient()
    client.start()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Put frame in queue (discard old frames)
            if client.frame_queue.full():
                client.frame_queue.get_nowait()
            client.frame_queue.put(frame.copy())

            # Display latest results
            display_frame = frame.copy()
            for result in client.latest_results:
                x1, y1, x2, y2 = result.get("box", [0,0,0,0])
                label = f"{result.get('gender', '')}, {result.get('age', '')}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(display_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.imshow("Real-time Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        client.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()