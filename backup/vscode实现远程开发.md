##### 原理：

VSCode 的 Remote - SSH 功能本质上是利用 SSH 协议，在本地机器（客户端）和远程云服务器之间建立安全连接，然后在远程服务器上运行一个 VSCode Server 实例。本地 VSCode（前端界面）通过 SSH 通道与远程的 VSCode Server（后端）通信，从而实现远程开发体验。

用户在本地 VSCode 中操作时，感觉像在本地开发，但实际的计算和文件操作发生在远程服务器上。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/原理.png)



本例配置：本地电脑为win11，云服务器为Ubuntu

准备条件：本地安装好vscode

第一步：安装远程开发插件（remote-SSH）

第二步：配置相关文件

进入配置文件

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/image-20250311180136763.png)

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/image-20250311181110102.png)

文件内容

```config
Host debian			# 别名
  HostName yourip	# 实际服务器ip
  User root			# 登陆的用户名
  IdentityFile ~/.ssh/id_rsa	# 用于 SSH 认证的私钥文件路径
  Port 22			# 端口

Host ubuntu
  User root
  HostName yourip
  IdentityFile ~/.ssh/id_rsa
  Port 22
```

配置之后，连接即可。第一次连接需要初始化，可能会花费较长的时间。

注：在操作途中，需要密码。

打开文件夹，打开的是服务器的文件夹。



##### SSH免密登录

打开win的终端

输入

```
 ssh-keygen -t rsa 
```

生成秘钥对
![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/image-20250311180242242.png)

- 打开生成的秘钥保存路径，拷贝 `id_rsa.pub` 内容，添加到到云服务器的 `~/.ssh/authorized_keys` 文件后面。

重新连接，不需要密码