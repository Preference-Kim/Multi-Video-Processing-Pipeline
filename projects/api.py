import os
import asyncio
from uuid import UUID, uuid4
from typing import List, Dict
import ctypes
import yaml
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from http import HTTPStatus

import numpy as np
import cv2
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
#from pydantic import BaseModel



# Process setup
"""mp.cpu_count()=64"""
cv2.setNumThreads(32) # disable opencv multithreading to avoid system being overloaded
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
# torch.backends.cudnn.benchmark = True  # faster for fixed-size inference


mp = multiprocessing.get_context("spawn")


with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


SOURCES: List | None = CONFIG.get('sources')
IN_HEIGHT: int = 1080
IN_WIDTH: int = 1920
IN_DEPTH: int = 3

SHARED_MEMORY = []
REFERENCE_ARRAY = np.zeros((IN_HEIGHT, IN_WIDTH, IN_DEPTH), dtype=np.uint8).nbytes
for idx in range(len(SOURCES)):
    try:
        SHARED_MEMORY.append(
            shared_memory.SharedMemory(name='shm'+str(idx), create=True, size=REFERENCE_ARRAY)
        )
    except FileExistsError:
        SHARED_MEMORY.append(
            shared_memory.SharedMemory(name='shm'+str(idx), create=False, size=REFERENCE_ARRAY)
        )
del REFERENCE_ARRAY

EVT_RUNNING = mp.Value(ctypes.c_bool, True)

"""
ㅆㅃ
"""


class Job(BaseModel):
    uid: UUID = Field(default_factory=uuid4)
    status: str = "in_progress"
    result: int = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ThreadPoolExecutor()
    yield
    app.state.executor.shutdown()

app = FastAPI(lifespan=lifespan) # $ uvicorn projects.api:app --reload
jobs: Dict[UUID, Job] = {}


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(app.state.executor, fn, *args)  # wait and return result


async def start_task(uid: UUID, param: int) -> None:
    jobs[uid].result = await run_in_process(imread_on_mem, param)
    jobs[uid].status = "complete"


@app.get("/new_task/{param}", status_code=HTTPStatus.ACCEPTED) # expected to use post, but not working -- https://stackoverflow.com/questions/71853957/fastapi-post-method-ends-up-giving-method-not-allowed
async def task_handler(param: int, background_tasks: BackgroundTasks):
    new_task = Job()
    jobs[new_task.uid] = new_task
    background_tasks.add_task(start_task, new_task.uid, param)
    return new_task


@app.get("/status/{uid}")
async def status_handler(uid: UUID):
    return jobs[uid]


def imread_on_mem(stream_id: int=0):
    cap = cv2.VideoCapture(SOURCES[stream_id], cv2.CAP_FFMPEG, params=[cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    EVT_RUNNING.value = True

    while EVT_RUNNING.value:
        cap.grab()
        ret, frame = cap.retrieve()
        shared_bytes = np.ndarray(frame.shape, dtype=frame.dtype, buffer=SHARED_MEMORY[stream_id].buf)
        shared_bytes[:] = frame

    cap.release()
    SHARED_MEMORY[stream_id].unlink()
    SHARED_MEMORY[stream_id].close()


"""
When building APIs, you normally use these specific HTTP methods to perform a specific action.
Normally you use:
    POST: to create data.
    GET: to read data.
    PUT: to update data.
    DELETE: to delete data.
"""


@app.get("/monitoring/{stream_id}") #TODO: 주소로부터 이미지 읽기(Reader)
def monitor(stream_id: int):
    EVT_RUNNING.value = True
    s = 'shm'+str(stream_id)
    print(f"TRYING TO OPEN SHM {s}")
    shared_mem_r = shared_memory.SharedMemory(name='shm'+str(stream_id))
    # shared_bytes = np.frombuffer(shared_mem_r.buf, dtype=np.uint8)
    return StreamingResponse(get_stream(shared_mem_r), media_type="multipart/x-mixed-replace; boundary=frame")

def get_stream(shared_mem_r: shared_memory.SharedMemory):    
    while EVT_RUNNING.value:
        _ = cv2.waitKey(int(1000/30))
        shared_arr = np.ndarray((IN_HEIGHT, IN_WIDTH, IN_DEPTH), dtype=np.uint8, buffer=shared_mem_r.buf)
        ret, encoded_buf = cv2.imencode('.jpg', shared_arr)
        frame = encoded_buf.tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n'
    
@app.get("/close/")
def terminate():
    EVT_RUNNING.value = False
    return {"message": "Completed the execution"}