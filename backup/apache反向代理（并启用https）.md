

```
2024/11/15 更新

可以使用cloudfare启用https(较为简单，更新后的DNS可能需要1天到两天更新。chenhaohan.top实际用了20分钟左右)
```



#### apache反向代理(并启用https)



安装apache

```
sudo apt update
sudo apt install apache2
```

确保 Apache 服务器已经启用了反向代理相关的模块

```
a2enmod proxy
a2enmod proxy_http
```

编辑 Apache 的虚拟主机配置文件。通常这些文件位于 `/etc/apache2/sites-available/` 目录下。如果你还没有为你的网站创建一个配置文件，可以新建一个，例如 5244.conf

```
sudo nano /etc/apache2/sites-available/5244.conf
```

假设你要将访问 `www.example.com` 的请求反向代理到 `localhost:1234`

```
<VirtualHost *:80>
    ServerName www.chenhaohan.help

    # 启用反向代理
    ProxyPreserveHost On
    ProxyPass / http://localhost:5244/
    ProxyPassReverse / http://localhost:5244/

    # 日志配置
    ErrorLog ${APACHE_LOG_DIR}/www.chenhaohan.help_error.log
    CustomLog ${APACHE_LOG_DIR}/www.chenhaohan.help_access.log combined
RewriteEngine on
RewriteCond %{SERVER_NAME} =www.chenhaohan.help
RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]
</VirtualHost>


```

`ServerName example.com`：指定要处理的域名。

`ProxyPreserveHost On`：保持原始的 `Host` 请求头。

`ProxyPass` 和 `ProxyPassReverse` 指定将请求从 `/` 转发到 `localhost:8080`，这是你想代理的服务端口。

日志配置方便排查问题。



保存配置文件后，使用以下命令启用该站点配置:

```
sudo a2ensite 5244.conf
```

如果你在编辑时用的是默认站点 (`000-default.conf`)，则不需要此步骤.



在应用配置前，先检查 Apache 配置文件是否有错误：

```
sudo apache2ctl configtest
```

如果没有发现错误，Apache 会返回 `Syntax OK`。



最后，重启 Apache 以使更改生效：

```
sudo systemctl restart apache2
```





##### 启用https

具体步骤为：

**安装 Certbot 和 Apache 插件**

**获取 SSL 证书**

**自动配置 Apache 以支持 HTTPS**

**测试 HTTPS 配置**

**设置自动续期**



 1.安装 Certbot 和 Apache 插件

```
sudo apt update
sudo apt install certbot python3-certbot-apache
```

`certbot` 是主要的工具，它与 Let’s Encrypt 通信并获取 SSL 证书。

`python3-certbot-apache` 是 Certbot 的 Apache 插件，用于自动配置 Apache 来启用 SSL。



2.获取SSl证书

获取 SSL 证书时，Certbot 会与 Let’s Encrypt 服务器通信，并生成证书文件。要获取证书并为 Apache 服务器自动配置 HTTPS，运行以下命令：

```
sudo certbot --apache
```

这将启动 Certbot，并引导你完成以下步骤：

1. **输入你的电子邮件地址**： Certbot 需要一个电子邮件地址，用于通知你证书到期或安全问题。
2. **同意服务条款**： 你需要同意 Let’s Encrypt 的服务条款才能继续。
3. **选择要启用 HTTPS 的域名**： Certbot 会自动检测到配置在 Apache 上的域名。如果你的虚拟主机文件中配置了多个域名，Certbot 会显示所有域名，并让你选择想要为哪些域名启用 HTTPS。
4. **自动重定向 HTTP 到 HTTPS**： Certbot 会询问你是否要将所有 HTTP 请求重定向到 HTTPS。建议选择 `2`，启用自动重定向，这样所有通过 `http://` 的请求都会被重定向到 `https://`，确保所有通信都是加密的。

出现下面的选择时，选择 2 选项

```
Please choose whether or not to redirect HTTP traffic to HTTPS, removing HTTP access.

1: No redirect - Make no further changes to the webserver configuration.
2: Redirect - Make all requests redirect to secure HTTPS access. Choose this for new sites, or if you're confident your site works on HTTPS. You can undo this change by editing your web server's configuration.

Select the appropriate number [1-2] then [enter] (press 'c' to cancel): 2
```

Certbot 会自动生成 SSL 证书，更新 Apache 配置文件，并重新加载 Apache 服务器，使新的配置生效。



3.自动配置 Apache 以支持 HTTPS

Certbot 会自动修改 Apache 的虚拟主机配置文件，添加用于 HTTPS 的 SSL 相关配置。通常，这些更改包括：

- 启用 `443` 端口用于 HTTPS。
- 添加 `SSLEngine on` 以启用 SSL。
- 指定证书文件位置

```
<VirtualHost *:443>
    ServerName example.com

    SSLEngine on
    SSLCertificateFile /etc/letsencrypt/live/example.com/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/example.com/privkey.pem
    Include /etc/letsencrypt/options-ssl-apache.conf
</VirtualHost>
```

`SSLCertificateFile` 指定生成的证书。

`SSLCertificateKeyFile` 指定私钥。

`Include /etc/letsencrypt/options-ssl-apache.conf` 用于加载推荐的 SSL 设置。



4.测试https配置

检测https是否生效

```
sudo systemctl restart apache2
```

然后你可以通过浏览器访问你的域名，例如 `https://example.com`，查看 HTTPS 是否生效。浏览器地址栏应该显示安全的锁形图标，表示站点已加密。

也可以通过以下命令验证证书的有效性：

```
sudo certbot certificates
```

这个命令会显示当前系统上已生成的证书列表和到期日期。



5.设置自动续期

Let’s Encrypt 的证书有效期为 90 天，为了确保证书不会过期，你可以使用 Certbot 来自动续期。Certbot 包含一个系统定时任务（cron job），每天检查证书的有效性，并在即将过期时自动续期。

```
sudo certbot renew --dry-run
```

如果一切正常，你将看到类似于 `Congratulations, all renewals succeeded!` 的信息。这意味着 Certbot 已成功模拟了证书续期。



注意：**HTTP 到 HTTPS 自动重定向**：选择了自动重定向选项后，Certbot 会在 Apache 的配置中添加 `RewriteRule`，确保所有的 HTTP 请求重定向到 HTTPS。例如

```
<VirtualHost *:80>
    ServerName example.com
    RewriteEngine On
    RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [L,QSA,R=permanent]
</VirtualHost>
```

**Let's Encrypt 证书是免费的**，并且是广泛受信任的证书颁发机构（CA）。由于证书的有效期较短，因此自动续期功能非常重要。