import uvicorn
import signal
import datetime
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import asyncio
import logging
import webbrowser
import threading

class Applet:
    def __init__(self, name, component):
        self.name = name
        self.component = component
        self.app = FastAPI()
        self.web = None
        self.ev_closed = threading.Event()

        @self.app.post('/close')
        async def close():
            self.close()

    def close(self):
        print(f"closing {self.name}")
        if self.web:
            print(f"deactivating {self.name}")
            self.web.deactivate_applet(self)
        print(f"setting ev")
        self.ev_closed.set()
        print(f"done")

    def wait(self):
        return self.ev_closed.wait()

    def run(self, workspace):
        workspace.web.activate_applet(self)
        self.wait()

class WebInterface: 
    class LogHandler(logging.Handler):
        def __init__(self, name, web):
            super().__init__()
            self.name = name 
            self.web = web
            self.formatter = logging.Formatter('%(message)s')
        def emit(self, record):
            self.web.broadcast({
                'command': 'log',
                'name': self.name,
                'level': record.levelname,
                'message': self.format(record)
            })

    def __init__(self, logger, port=8989, debug=False):
        self.log = logger
        self.port = port
        self.debug = debug
        self.applet = None
        self.app = FastAPI()
        self.loop = None
        self.log_handlers = {}
        self.log_active = set()
        self.ev_ready = threading.Event()
        self.log_history = []
        self.current_progress = None
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.websockets = []
        @self.app.get("/test")
        def test():
            return { "status": "success" }

        # websocket at /ws
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            self.log.info("websocket connected")
            await websocket.accept()
            self.websockets.append(websocket)
            self.on_connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    self.on_recv(websocket, data)
            except Exception:
                self.websockets.remove(websocket)

        # create subrouter at /applet
        self.applet_router = FastAPI()
        self.app.mount("/applet", self.applet_router)

        StaticFiles.is_not_modified = lambda self, *args, **kwargs: False
        static_files_path = os.path.join(os.path.dirname(__file__), 'web', 'static')
        self.app.mount("/", StaticFiles(directory=static_files_path, html=True), name="static")

        @self.app.on_event("startup")
        async def startup_event():
            self.log.info("startup")
            self.loop = asyncio.get_running_loop()
            self.ev_ready.set()

    def activate_applet(self, applet):
        '''Register an applet with the web interface'''
        self.applet = applet
        self.applet_router.routes.clear()
        self.applet_router.mount(f"/{applet.name}", applet.app, name=applet.name)
        self.applet.web = self
        self.activate()

    def connect_log(self, name, logger):
        '''Connect a logger to the web interface'''
        #self.log.info(f"connect_log: {logger.name}")
        self.broadcast({
            'command': 'connect_log', 
            'name': logger.name
        })
        self.log_active.add(name)
        self.log_handlers[name] = self.log_handlers.get(name) or self.LogHandler(name, self)
        logger.addHandler(self.log_handlers[name])

    def disconnect_log(self, name, logger):
        '''Disconnect a logger from the web interface'''
        #self.log.info(f"disconnect_log: {logger.name}")
        logger.removeHandler(self.log_handlers[name])
        self.log_active.remove(name)
        self.broadcast({
            'command': 'disconnect_log', 
            'name': logger.name
        })

    def activate(self):
        self.broadcast({
            'command': 'activate', 
            'applet': self.applet.name, 
            'component': self.applet.component, 
            'path': f"/applet/{self.applet.name}"
        })

    def deactivate_applet(self, applet): 
        if self.applet == applet:
            # Clear any existing progress first
            self.broadcast({'command': 'progress', 'clear': True})
            self.applet = None
            self.applet_router.routes.clear()
            self.broadcast({'command': 'deactivate'})
        else:
            self.log.error(f"deactivate_applet: applet {applet} is not active")

    def on_connect(self, websocket):
        # Send history first
        if self.log_history:
            asyncio.create_task(websocket.send_text(json.dumps({
                'command': 'history',
                'logs': self.log_history[-1000:],
                'progress': self.current_progress
            })))
        
        if self.applet:
            self.activate()
        for name in self.log_active:
            self.broadcast({
                'command': 'connect_log', 
                'name': name
            })

    def on_recv(self, websocket, data):
        '''Called when data is received from a websocket'''
        self.log.info(f"received: {data} from {websocket}")

    def broadcast(self, data: dict):
        '''Broadcast data to all connected websockets'''
        # Update history if needed
        if data['command'] == 'log':
            self.log_history.append({
                'id': len(self.log_history),
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'message': data['message']
            })
            if len(self.log_history) > 1000:
                self.log_history.pop(0)
        elif data['command'] == 'progress':
            if 'clear' in data and data['clear']:
                self.current_progress = None
            else:
                self.current_progress = data
        
        def do_broadcast():
            for ws in self.websockets:
                asyncio.create_task(ws.send_text(json.dumps(data)))
        self.loop.call_soon_threadsafe(do_broadcast)

    def stop(self):
        """Stop the web interface cleanly"""
        if hasattr(self, 'server'):
            self.server.should_exit = True
            if hasattr(self, 'thread'):
                self.thread.join()
            # Close all websockets
            if self.loop:
                for ws in self.websockets[:]:
                    self.loop.call_soon_threadsafe(lambda: asyncio.create_task(ws.close()))
                self.websockets.clear()

    def start(self):
        # Store original SIGINT handler
        self.original_sigint = signal.getsignal(signal.SIGINT)
        
        def sigint_handler(signum, frame):
            self.log.info("Received SIGINT, shutting down...")
            self.stop()
            # Restore original SIGINT handler and re-raise the signal
            signal.signal(signal.SIGINT, self.original_sigint)
            os.kill(os.getpid(), signal.SIGINT)

        # Set our custom SIGINT handler
        signal.signal(signal.SIGINT, sigint_handler)

        self.config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level='debug' if self.debug else 'critical')
        self.server = uvicorn.Server(self.config)

        self.log.info("-" * 80)
        self.log.info("")
        self.log.info(f"Web interface is running at http://0.0.0.0:{self.port}")
        self.log.info("")
        self.log.info("Open it up, and follow the instructions on the web page!") 
        self.log.info("")
        self.log.info("-" * 80)

        # Run web interface in the background
        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()

        self.ev_ready.wait()

        try:
            webbrowser.open(f"http://localhost:{self.port}")
        except:
            pass

