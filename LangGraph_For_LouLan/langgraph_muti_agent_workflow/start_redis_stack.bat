@echo off
echo 正在启动 Redis Stack...
docker run -d -p 6379:6379 --name redis-stack redis/redis-stack:latest
echo Redis Stack 已启动！
echo 等待服务就绪...
timeout /t 5 /nobreak >nul
echo 完成！现在可以运行 CoupletLoader.py 了
