# PyAV
# https://stackoverflow.com/questions/65004533/pyav-how-to-display-multiple-video-streams-to-the-screen-at-the-same-time

import av
import cv2
import numpy as np
from multiprocessing import Process

def process_video(video_stream_url, index):
    container = av.open(video_stream_url)
    video_stream = next(s for s in container.streams if s.type == 'video')

    for packet in container.demux(video_stream):
        for frame in packet.decode():
            bgr_frame = np.array(frame.to_image())
            bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Display Stream Number "+str(index), cv2.WINDOW_NORMAL)
            cv2.imshow("Display Stream Number "+str(index), bgr_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

def main():
    video_stream_urls = [
        "rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/101",
        "rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/201",
        "rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/301",
        "rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/401"
        # Add more RTSP stream URLs as needed
    ]

    processes = []

    for i, url in enumerate(video_stream_urls):
        p = Process(target=process_video, args=(url,i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()