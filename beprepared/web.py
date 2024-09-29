from beprepared.workspace import Workspace

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import webbrowser

class WebInterface:
    def __init__(self, name='Web Interface', static_files_path=None, port=8989, debug=False):
        self.static_files_path = static_files_path
        self.port = port
        self.name = name
        self.debug = debug
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.post("/api/exit")
        async def exit_server():
            self.server.should_exit = True
            return {"status": "exiting"}

    def run(self):
        if self.static_files_path:
            self.app.mount("/", StaticFiles(directory=self.static_files_path, html=True), name="static")
        self.config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level='debug' if self.debug else 'critical')
        self.server = uvicorn.Server(self.config)


        Workspace.current.log.info("-" * 80)
        Workspace.current.log.info("")
        Workspace.current.log.info(f"{self.name} is running at http://localhost:{self.port}")
        Workspace.current.log.info("")
        Workspace.current.log.info("Open it up, and follow the instructions on the web page!") 
        Workspace.current.log.info("")
        Workspace.current.log.info("-" * 80)

        try:
            webbrowser.open(f"http://localhost:{self.port}")
        except:
            pass

        self.server.run()



