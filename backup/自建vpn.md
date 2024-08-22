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

