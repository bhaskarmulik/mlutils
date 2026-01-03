import mlflow
from pyngrok import ngrok
from typing import Optional
from IPython.core import getipython


class ngrok_tunnel:

    @staticmethod
    def setup_for_notebook(
        ngrok_auth : str 
    ) -> str | None:
        
        #Check if auth is provided
        if ngrok_auth != "":
            ngrok.set_auth_token(ngrok_auth)
        else:
            ValueError("No auth token provided")

        
        ngrok.kill()
        try:
            ipython = getipython.get_ipython()
            if ipython is not None:
                ipython.system_raw("mlflow ui --port 5000 &")
            else:
                raise NameError("Not in IPython environment")
        except NameError:
            import subprocess
            subprocess.Popen(["mlflow", "ui", "--port", "5000"])
        # Open an HTTPs tunnel on port 5000 for http://localhost:5000
        ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True, host_header = 'rewrite')
        print("MLflow Tracking UI:", ngrok_tunnel.public_url)

        return ngrok_tunnel.public_url