@echo off
setlocal

set "QLIB_ROOT=C:\Users\WANGKANG\Desktop\qlib"
set "JUPYTER_TOKEN="

"C:\Users\WANGKANG\miniconda3\envs\qlib\Scripts\jupyter-server.exe" ^
  --ServerApp.root_dir="%QLIB_ROOT%" ^
  --IdentityProvider.token="%JUPYTER_TOKEN%" ^
  --ServerApp.ip=127.0.0.1 ^
  --port=8888 ^
  --no-browser ^
  --ServerApp.websocket_ping_interval=60 ^
  --ServerApp.websocket_ping_timeout=120 ^
  --ZMQChannelsWebsocketConnection.limit_rate=False ^
  --ZMQChannelsWebsocketConnection.iopub_data_rate_limit=2147483647 ^
  --ZMQChannelsWebsocketConnection.iopub_msg_rate_limit=2147483647 ^
  --MappingKernelManager.buffer_offline_messages=True

pause


@REM 增加 WebSocket 的心跳发送间隔，避免高负载时误判断线
@REM --ServerApp.websocket_ping_interval=60
@REM --ServerApp.websocket_ping_timeout=120
@REM 彻底关闭对内核向前端发送数据量和消息频率的限制
@REM --ZMQChannelsWebsocketConnection.limit_rate=False
@REM --ZMQChannelsWebsocketConnection.iopub_data_rate_limit=2147483647
@REM --ZMQChannelsWebsocketConnection.iopub_msg_rate_limit=2147483647
@REM 确保前端意外断开时，服务器在内存中缓冲消息，等重连后回放
@REM --MappingKernelManager.buffer_offline_messages=True
