#!/usr/bin/env python3
import cv2
import time
from time import sleep
import numpy as np
import os
import threading
from threading import Thread
from pre_process import mask, mask_red
from message import Interface
from aiohttp import web
import asyncio
from functools import partial
#from usearch.index  import search, MetricKind
import faiss

FAISS_GPU_RES = faiss.StandardGpuResources()

class VideoEOFException(Exception):
    pass


class VideoStreamWidget:
    def __init__(self, src, is_video=False):
        self.lock = threading.Lock()
        self.status = False
        self.frame = None
        self.capture = cv2.VideoCapture(src)
        if self.capture.isOpened():
            print("Camera opened")
        else:
            print("Camera not opened")

        self.is_video = is_video
        self.thread = None

    def start(self):
        if self.is_video:
            return

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        if self.is_video:
            return

        while True:
            if self.capture.isOpened():
                status_temp, frame_temp = self.capture.read()
                with self.lock:
                    self.status, self.frame = status_temp, frame_temp

    def get_frame(self):
        if self.is_video:
            self.status, self.frame = self.capture.read()

            if not self.status:
                raise VideoEOFException()

            return self.frame.copy()


        with self.lock:
            if not self.status:
                return None

            if self.frame is None:
                return None

            return self.frame.copy()




class Image:
    def __init__(self, rgb, sift_type, tools):
        self.rgb = rgb

        self.keypoint_channel, self.mask = mask(rgb)

        self.keypoints = None
        self.descriptors = None
        
        opencv_kp_lib = ["orb", "surf", 'sift_opencv']

        if sift_type in opencv_kp_lib:
            self.compute_keypoints_opencv(tools[sift_type])
        elif sift_type == "sift_ocl" : 
            self.compute_keypoints_sift_ocl(tools[sift_type])
        else : 
            raise Exception("Wrong keypoints processor")

    def compute_keypoints_sift_ocl(self, sift_ocl):
        sift_results = sift_ocl(self.keypoint_channel)
        self.keypoints = np.hstack((sift_results.x.reshape(-1, 1), sift_results.y.reshape(-1, 1)))
        self.descriptors = sift_results.desc

        self.keypoints = self.keypoints[self.mask[sift_results.y.astype(int), sift_results.x.astype(int)]]
        self.descriptors = self.descriptors[self.mask[sift_results.y.astype(int), sift_results.x.astype(int)]]

        #self.faiss_index = Index(ndim = self.descriptors.shape[1], metric = 'l2sq')  # TODO metric param
        #self.faiss_index.add(self.descriptors)

    def compute_keypoints_opencv(self, tool):
        keypoints, self.descriptors = tool.detectAndCompute(self.keypoint_channel, None)
        
        keypoints_ = []
        for kp in keypoints :
            keypoints_.append(kp.pt)
        keypoints_ = np.array(keypoints_)

        self.keypoints = keypoints_[self.mask[keypoints_[:, 1].astype(int), keypoints_[:, 0].astype(int)]]
        self.descriptors = self.descriptors[self.mask[keypoints_[:, 1].astype(int), keypoints_[:, 0].astype(int)]]


    def match_with(self, other):
        if len(self.descriptors) < len(other.descriptors):
            smallest = self
            biggest = other
            flip = False
        else:
            smallest = other
            biggest = self
            flip = True

        indices = np.empty((smallest.descriptors.shape[0], 3), dtype=np.int64)
        indices[:, 0] = np.arange(smallest.descriptors.shape[0])
       
       
       # Faiss
        
        index = faiss.IndexFlatL2(biggest.descriptors.shape[1])
        index.add(biggest.descriptors)

        index_gpu = faiss.index_cpu_to_gpu(FAISS_GPU_RES, 0, index)

        distances, indices[:, 1:] = index_gpu.search(smallest.descriptors, 2)


       # Usearch
       # matches = search(biggest.descriptors.astype(np.float32), smallest.descriptors.astype(np.float32), 2, MetricKind.L2sq, exact=True)  # TODO batches
       # 
       # indices[:, 1:] = matches.keys
       # distances = matches.distances

        good_mask = distances[:, 0] < 0.5 * distances[:, 1]

        if flip:
            return indices[good_mask, :2][..., ::-1]
        else:
            return indices[good_mask, :2]
    


class OpticalFlowProcessor:
    def __init__(self, video_stream, sift_type, plot_bool):
        self.video_stream = video_stream
        self.lock = threading.Lock()
        self.sift_type = sift_type

        self.interface = Interface(1106)

        self.reference = None
        self.new_reference_requested = True

        self.tools = {}
        if sift_type == "orb":
            self.tools[sift_type] = cv2.ORB_create()
        elif sift_type == 'sift_opencv':
            self.tools[sift_type] = cv2.SIFT_create()
        
        self.plot_bool = plot_bool
        self.config = { 
                'norm_val' : 100, 
                'radius_red' : 5,
                'saturation' : 200, 
                'dead_zone' : 5,
                'V_scale' : 100,}
        self.V = np.zeros(3)


    def start(self):
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while True:
            try:
                frame = self.video_stream.get_frame()
            except VideoEOFException:
                print('Video Excpetion')
                break

            if frame is None:
                sleep(0.1)
                print('Frame is None')
                continue
            with self.lock:
                if self.new_reference_requested or (self.reference is None):
                    self.update_reference(frame.copy())
                    self.new_reference_requested = False
           
            self.process(self.reference, Image(frame.copy(), self.sift_type, self.tools))
            
            #################### Normalement géré par ardupilot mtn
            #self.V[1] = -self.V[1]
            #self.V = -self.V
            #######################
            V_scale = self.config["V_scale"]
            self.V /= V_scale

            message = (self.V, self.quality)
            self.interface.send(message, "vel_qual")

    def process(self, reference, new_frame) : 
        
        # Calculation of indexes of matching points
        matches = reference.match_with(new_frame)
        
        if not matches.shape[0]:
            return np.zeros(2)
        
        # Extraction of good key points
        matched_kp_ref = reference.keypoints[matches[:, 0]]
        matched_kp_new_frame = new_frame.keypoints[matches[:, 1]]
        
        
        #Z with red
        radius = self.config["radius_red"]

        red_channel, _ = mask_red(new_frame.rgb)
        red_channel_ref, _ = mask_red(reference.rgb)
        
        X = matched_kp_new_frame[:, 0].astype(int)
        Y = matched_kp_new_frame[:, 1].astype(int)
        keypoint_mask = np.zeros(red_channel.shape)
        
        for i in range(X.shape[0]):
            keypoint_mask[Y[i] - radius: Y[i] + radius, X[i] - radius: X[i] + radius] = 1
            
        red = red_channel * keypoint_mask


        X = matched_kp_ref[:, 0].astype(int)
        Y = matched_kp_ref[:, 1].astype(int)        
        keypoint_mask_ref = np.zeros(red_channel.shape)
        
        for i in range(X.shape[0]):
            keypoint_mask_ref[Y[i] - radius: Y[i] + radius, X[i] - radius: X[i] + radius] = 1
        
        red_ref = red_channel_ref * keypoint_mask_ref

        diff = red - red_ref
     
        diff = diff[diff != 0]
        if len(diff) == 0 : 
            diff = 0
        else : 
            pass
        red_val = np.mean(diff)
        
        # Calculation of the speed
        V = np.array([  np.median(matched_kp_new_frame[:, 0] - matched_kp_ref[:, 0]), 
                        np.median(matched_kp_new_frame[:, 1] - matched_kp_ref[:, 1]),
                        red_val])
        # Filter
        dead_zone = self.config["dead_zone"]
        sat = self.config["saturation"] 
        
        if np.linalg.norm(V) < dead_zone:
            V = np.zeros(V.shape)
        if np.abs(V[0]) > sat: 
            if V[0] > 0:
                V[0] = sat
            else:
                V[0] = -sat
        if np.abs(V[1]) > sat:
            if V[1] > 0:
                V[1] = sat
            else:
                V[1] = -sat
        self.V = V

        quality = matches.shape[0]

        norm_val = self.config["norm_val"]
        self.quality = min(255, int(255*quality/norm_val)) 
        
        
        if self.plot_bool == True :  
            ######################### PLOT #################################
            
            image, mask_ = new_frame.keypoint_channel, new_frame.mask
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            image = cv2.addWeighted(image, 0.8, cv2.cvtColor(mask_.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR), 0.2, 0)
            image = cv2.addWeighted(image, 0.5, cv2.cvtColor(keypoint_mask.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR), 0.5, 0)
            for point in new_frame.keypoints :
                pos = point.astype(int)
                cv2.circle(image, pos, 2, (0, 0, 255), -1)
            for point in matched_kp_new_frame:
                pos = point.astype(int)
                cv2.circle(image, pos, 2, (0, 255, 255), 1)
            try:
                image = cv2.arrowedLine(image, (np.array(image.shape)[1::-1]//2).astype(int), (np.array(image.shape)[1::-1]//2 + V[:2]).astype(int), (0, 255, 0), 8)
            except:
                print((np.array(image.shape)[1::-1]//2 + V).astype(int))
                return np.zeros(2)
            image = cv2.putText(image, str(np.round(V[2], 2)), np.array(image.shape)[1::-1]//2, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_4) 
            
            cv2.imshow("video", image)
            cv2.waitKey(1)

    def update_reference(self, new_frame):
        
        if self.sift_type == "sift_ocl" : 
            if 'ocl' not in self.tools :

                # set to 1 to see the compilation going on
                os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"
                from silx.image import sift
                
                self.tools['sift'] = sift.SiftPlan((new_frame.shape[0] // 2, new_frame.shape[1] // 2), new_frame.dtype, devicetype="GPU")
        
        self.reference = Image(new_frame.copy(), self.sift_type, self.tools)
        
        print("-------------------")
        print("Reference updated")
        print("-------------------")

    def request_reference_update(self):
        with self.lock:
            self.new_reference_requested = True

    def request_config_update(self, id_, val) : 
        with self.lock:
            self.config[id_] = val


async def route_request_reference_update(optical_flow_process, request):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, optical_flow_process.request_reference_update)

    return web.Response()


async def button_webpage(post_url, request):
    template = """
     <!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
</head>
<body>

<button id='refresh_button'>
    Refresh reference
</button>

<script>
    const refresh_button = document.querySelector('#refresh_button');

    refresh_button.onclick = e => {
        console.log("sending request")
        console.log(axios)
        axios.post("REPLACE_ME")
            .then(response => {console.log("okay")})
            .catch(err => {console.log("okay")});
    };
</script>

</body>
</html>
    """.replace("REPLACE_ME", post_url)

    return web.Response(text=template, content_type="text/html")



async def route_config_update(optical_flow_process, request):
    loop = asyncio.get_event_loop()
    val = await request.json()
    await loop.run_in_executor(None, partial(optical_flow_process.request_config_update, val['id'], int(val['val'])))

    return web.Response()


async def config(post_url, config, request) :
    Name = {"norm_val" : "Normalisation des matchs : 255/val*len_match",
            "radius_red": "Rayon de la zone autours des matchs d'analyse du rouge pour la profondeur",
            "saturation": "Saturation du déplacement sur le plan (xOz) : Coordonée du robot",
            "dead_zone": "Dead zone du déplacement sur le plan (xOz) : Coordonée du robot",
            "V_scale": "Facteur de division sur la vitesse /!\\ Penser à changer les PIDs"}
    template = """
     <!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
</head>
<body> """


    template += "<div id= 'button_container' style = 'display : flex; flex-direction : column;'>"
    for i in config.keys() :
        template += "<div>" 
        template += f"<input type = 'number' onchange = 'send_val(this.id, this.value)' id = {i} value = {config[i]}>"
        template += f"<label for = {i} style = 'color : white;'> {Name[i]} </label>"
        template += "</div>"
    template += "</div>"

    template += """
<script>
    function send_val(id, value){
        let i = parseInt(value)
        axios.post("REPLACE_ME", json = {id: id, val: i.toString()})
            .then(response => {console.log(id, i)})
            .catch(err => {console.log(err)});
    };
</script>

</body>
</html>
    """.replace("REPLACE_ME", post_url)


    return web.Response(text=template, content_type="text/html")



def main():
    vs = VideoStreamWidget("/home/crc/OF/cpp/video_test.mp4", is_video=False)
    ofp = OpticalFlowProcessor(vs, "sift_opencv", plot_bool=True)
    
    app = web.Application()
    
    app.add_routes([web.post('/update_reference', partial(route_request_reference_update, ofp))])
    app.add_routes([web.get('/', partial(button_webpage, "http://localhost:8080/update_reference"))])

    app.add_routes([web.post('/update_config', partial(route_config_update, ofp))])
    app.add_routes([web.get('/config', partial(config, "http://localhost:8080/update_config", ofp.config))])

    # TODO don't forget to setup CORS
    # TODO serve a tiny webpage with a button too

    vs.start()
    ofp.start()
    web.run_app(app)


if __name__ == '__main__':
    main()
