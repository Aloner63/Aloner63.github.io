使用 git clone 下载 Github 等网站的仓库时，可能会遇到类似 "Recv failure: Connection was reset" 或 "Failed to connect to http://github.com port 443 after 21114 ms: Couldn't connect to server" 的报错。即使打开了[全局代理](https://zhida.zhihu.com/search?content_id=232166536&content_type=Article&match_order=1&q=%E5%85%A8%E5%B1%80%E4%BB%A3%E7%90%86&zhida_source=entity)，也会报错。
此时，需要为 Git 单独配置代理
可以使用
```
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```
进行设置，代理端口为7890
也可以进入文件内手动修改
```
git config --global --edit

如果没有的话，就添加
[http]
	proxy = http://127.0.0.1:7890
[https]
	proxy = http://127.0.0.1:7890
```
