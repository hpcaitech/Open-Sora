# 构建过程常见问题
## Failed to build apex / subprocess.CalledProcessError
检查构建过程日志在编译过程中是否出现 `Killed` 字样，若有则说明内存不足以构建该镜像。降低构建过程CPU数量可减小并发编译内存消耗。对于 Windows Docker Desktop 用户尝试编辑`.wslconfig`文件，加入以下字段。
```
memory=16GB # 越接近你的机器最大内存越好
processors=4 # 适当减小该数值
```
## SSL Error
对于处于互联网审查区域的用户，尝试换源，或者使用审查绕过方案连接互联网。
## 构建时间过长
在i7-10700@16GB的机器上耗费了约53min完成构建。若构建时间过长，尝试换源，或者使用审查绕过方案连接互联网。