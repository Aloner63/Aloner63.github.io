第一步：获取vps服务器。VPS服务器需要选择国外的，首选国际知名的vultr，速度不错、稳定且性价比高，按小时计费，能够随时开通和删除服务器，新服务器即是新ip。在[vultr](https://www.vultr.com/)网站上，购买你觉得合适得vps（一般来说，个人用得话，就选低配置就行）
注册并邮件激活账号，充值后即可购买服务器。充值方式是支付宝或paypal，使用paypal有银行卡（包括信用卡）即可。paypal注册地址：https://www.paypal.com/ （paypal是国际知名的第三方支付服务商，注册一下账号，绑定银行卡即可购买国外商品）
注意：2.5美元的套餐只提供ipv6，一般的电脑用不了，所以建议选择3.5美元以上的套餐
vultr实际上是折算成小时来计费的，比如服务器是5美元1个月，那么每小时收费为5/30/24=0.0069美元 会自动从账号中扣费，只要保证账号有钱即可。如果你部署的服务器实测后速度不理想，你可以把它删掉（destroy），重新换个地区的服务器来部署，方便且实用。因为新的服务器就是新的ip，所以当ip被墙时这个方法很有用。当ip被墙时，为了保证新开的服务器ip和原先的ip不一样，先开新服务器，开好后再删除旧服务器即可。在账号的Account——Make a payment选项里可以看到账户余额。

- 充值
Account下的Make a Payment下的支付宝（Alipay），选择充值金额，充值即可。
<img width="926" alt="充值" src="https://github.com/user-attachments/assets/bff67e2b-d191-4c13-8268-121a9ddf0492">

- 购买vps
点击Deploy New Server
<img width="251" alt="购买vps" src="https://github.com/user-attachments/assets/3fa70ae0-85c2-4140-bd20-9120dea7bfaf">

购买最便宜的类型
<img width="1007" alt="屏幕截图 2024-08-22 121722" src="https://github.com/user-attachments/assets/31917620-ce3a-4fa7-967f-2d1430a997c6">

选择服务器位置。不同的服务器位置速度会有所不同，有的服务器的最低价格会不同，一般纽约等位置的价格最低，可根据自己的需求来选择。推荐洛杉矶服务器，延迟较低且比较稳定。
<img width="1003" alt="屏幕截图 2024-08-22 122122" src="https://github.com/user-attachments/assets/dfb17c75-d0f9-480c-908a-c4ecbd8c1198">

点击图中的系统名字，会弹出具体系统版本，推荐Debain11。当然其他也可以
<img width="997" alt="屏幕截图 2024-08-22 122602" src="https://github.com/user-attachments/assets/ce44729d-2e71-49dd-88bd-9ed0def925e4">

选择服务器套餐。根据自己的需求来选择，如果服务器位置定了，套餐不影响速度，影响流量和配置，一般用的人数少，选择低配置就够了。便宜的套餐，点击Regular Cloud Compute，选择一个合适的套餐，提示升级选择No Thanks。
<img width="1008" alt="屏幕截图 2024-08-22 122803" src="https://github.com/user-attachments/assets/8931451c-f7c8-4053-931b-76dc5e93284c">

关闭自动备份Auto Backups，这个是收费的。点击它，在右侧的I understand the risk前面选择勾，然后点击Disable Auto Backups即可关闭自动备份。（只是作为梯子使用，不需要自动备份）
<img width="989" alt="屏幕截图 2024-08-22 122918" src="https://github.com/user-attachments/assets/237baca8-4be0-4049-8d5c-05e174e224f0">

上述步骤完成以后，即可进行购买。等6~10分钟就差不多了。一般5美元一下就够自己，以及朋友们快乐。

第二步：购买域名
在著名的[namesilo](https://www.namesilo.com)网站上购买域名
进入主页面，搜索自己意向的域名。
一般购买一年最划算的，到期后再重新购买。选一个最便宜的后缀名即可
<img width="617" alt="屏幕截图 2024-08-22 145645" src="https://github.com/user-attachments/assets/407b43c1-ce85-4ad3-8a39-009847f5f0f7">

进入购物车，去掉不必要的服务以后，价钱应该降到最低了。这时候点击checkout，结算。可以用paypal，也可以用支付宝。
<img width="630" alt="屏幕截图 2024-08-22 150627" src="https://github.com/user-attachments/assets/81ce04b3-bb80-48b0-83f9-307e30716a33">

购买完成后，配置域名解析。配置两个A类型，如下图
<img width="616" alt="屏幕截图 2024-08-22 150913" src="https://github.com/user-attachments/assets/e24e1cdf-f0d3-4702-992e-e1383c7c925f">
配置好之后，等待20分钟左右。IP的DNS要全球解析，这个过程需要花费一段时间。检测是否配置好解析，打开本地的CMD，直接ping域名，如果出现了你的vps  ip就证明已经配置成功，如果没有ping成功，慢慢等待。

目前为止，你已经拥有了自己的vps服务器和一个域名，并且他们两者已经进行绑定了。

第三步：使用ProxySU一键部署

由于手动配置较麻烦，这里我们选择使用大神的工具一键配置。
[ProxySU最新版](https://github.com/proxysu/ProxySU/releases)
下载本地，打开
<img width="741" alt="屏幕截图 2024-08-22 151536" src="https://github.com/user-attachments/assets/a78f9074-576d-4930-a402-30fbc5f0d759">

点击右上角的添加主机，选择合适的类型添加，推荐Hysteria。（其他几个也使用过，容易被墙，个人观点）
<img width="735" alt="屏幕截图 2024-08-22 151610" src="https://github.com/user-attachments/assets/1afd2dd1-27dd-4ee7-a8f5-87198a7206a5">

下面以添加Hysteria为例（其他几种大同小异）
此处密码为vps的用户名和密码。
只需填红框部分即可，端口可改可不改。
<img width="741" alt="屏幕截图 2024-08-22 152155" src="https://github.com/user-attachments/assets/8e3bf0b5-468a-4790-818a-b7d89826ad0b">

点击一键安装，等待安装完成
<img width="591" alt="屏幕截图 2024-08-22 152507" src="https://github.com/user-attachments/assets/48d5d862-2ddb-4e70-8b97-9185cafcf259">
<img width="591" alt="屏幕截图 2024-08-22 152620" src="https://github.com/user-attachments/assets/2276281d-373c-4447-b243-865efade57fd">

安装完成后，回到ProxySU主页面
<img width="741" alt="屏幕截图 2024-08-22 152722" src="https://github.com/user-attachments/assets/6b670313-fd70-447b-a1f4-30791324f70a">

点击配置，将全部文本内容储存在config.json中。

第四步：使用
[hysteria更新地址](https://github.com/apernet/hysteria/releases)进入下载，自己操作系统的版本。

下载好以后，将它和刚才的config.json放到同一个文件夹下。
<img width="685" alt="屏幕截图 2024-08-22 153437" src="https://github.com/user-attachments/assets/2e5a3c90-b3f2-4b2e-b532-0fe1a57ddb38">

打开hysteria-windows-amd64.exe会出现一个命令行窗口，这时候就可以再浏览器配置相应的端口进行使用了（端口配置，要和config.json中的端口一样。如果不想使用config.json中的端口，可以自行修改，只要不和本机正在使用的端口冲突就行）

至此，可以快乐学习知识了。

补充：关于手机使用Hysteria的问题
[Nekobox](https://github.com/MatsuriDayo/NekoBoxForAndroid)功能极强的网络代理app。
下载到手机，直接右上角，添加。可以复制config.json中的文字，选择直接从剪切板添加，也可以自己手动输入。输入完成，点击下面的小飞机符号，可以进行连接，如果不成功，请检查配置是否有误。


服务器选择。
以下是一些提供价格相对便宜的国外云服务器商，适合不同需求和预算的用户：

### 1. **Vultr**
   - **特点**：提供按小时或按月计费，数据中心分布全球，支持一键部署多种操作系统和应用程序。
   - **起价**：$2.50/月（512MB RAM，1 CPU，10GB SSD）
   - **适用场景**：个人网站、小型应用、测试环境。

   网站：[https://www.vultr.com](https://www.vultr.com)

### 2. **Linode**
   - **特点**：历史悠久，稳定性好，全球多个数据中心，提供强大的 API 和简单易用的管理界面。
   - **起价**：$5/月（1GB RAM，1 CPU，25GB SSD）
   - **适用场景**：开发者项目、中小型网站、应用托管。

   网站：[https://www.linode.com](https://www.linode.com)

### 3. **DigitalOcean**
   - **特点**：用户界面友好，提供一键部署功能，支持按小时或按月计费，适合初学者使用。
   - **起价**：$5/月（1GB RAM，1 CPU，25GB SSD）
   - **适用场景**：个人项目、小型应用、初创企业。

   网站：[https://www.digitalocean.com](https://www.digitalocean.com)

### 4. **Hetzner**
   - **特点**：德国知名的低价云服务提供商，性价比非常高，适合需要较多资源但预算有限的用户。
   - **起价**：€4.15/月（2GB RAM，1 CPU，20GB SSD）
   - **适用场景**：中型网站、开发环境、游戏服务器。

   网站：[https://www.hetzner.com](https://www.hetzner.com)

### 5. **UpCloud**
   - **特点**：提供超快的服务器（使用 MaxIOPS 技术），数据中心遍布欧美和亚太地区，支持按小时计费。
   - **起价**：$5/月（1GB RAM，1 CPU，25GB SSD）
   - **适用场景**：高性能需求的应用、数据库托管、生产环境。

   网站：[https://www.upcloud.com](https://www.upcloud.com)

### 6. **Hostwinds**
   - **特点**：提供经济实惠的 VPS 和云服务器，支持灵活的按月或按小时计费，支持无限带宽。
   - **起价**：$4.99/月（1GB RAM，1 CPU，30GB SSD）
   - **适用场景**：小型企业网站、博客、个人项目。

   网站：[https://www.hostwinds.com](https://www.hostwinds.com)

### 7. **Scaleway**
   - **特点**：法国的云服务提供商，提供便宜的虚拟云服务器，支持灵活的计费方式和自动扩展。
   - **起价**：€3.00/月（1GB RAM，1 CPU，20GB SSD）
   - **适用场景**：轻量级应用、个人开发项目、临时测试环境。

   网站：[https://www.scaleway.com](https://www.scaleway.com)

### 8. **Contabo**
   - **特点**：提供非常实惠的大容量 VPS 和云服务器，支持高配置，性价比极高。
   - **起价**：$3.99/月（4GB RAM，2 CPU，50GB SSD）
   - **适用场景**：大流量网站、视频流、企业级应用。

   网站：[https://www.contabo.com](https://www.contabo.com)

---

这些云服务器商各有特点，选择时可以根据你的需求（如性能、数据中心位置、计费方式等）以及预算来挑选合适的服务。
