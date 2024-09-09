from huggingface_hub import snapshot_download
import urllib3, socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 200000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 200000000)
    ])
snapshot_download(repo_id="liuhaotian/llava-llama-2-13b-chat-lightning-preview",local_dir="C:\CodesFall24\Visual-Adversarial-Examples-Jailbreak-Large-Language-Models\ckpts")

