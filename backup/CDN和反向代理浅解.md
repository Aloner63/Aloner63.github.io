#### CDN

CDN的全称为“Content Delivery Network”，及，内容分发网络。

它是建立并覆盖在承载网之上，由分布在不同区域的边缘节点服务器群组成的分布式网络。

CDN应用广泛，支持多种行业、多种场景内容加速，例如：图片小文件、大文件下载、视音频点播、直播流媒体、全站加速、安全加速。

CDN工作原理

CDN的工作原理就是将源站的资源缓存到位于全国各地的CDN节点上，用户请求资源时，就近返回节点上缓存的资源，而不需要每个用户的请求都回您的源站获取，避免网络拥塞、分担源站压力，保证用户访问资源的速度和体验
![图片](https://github.com/user-attachments/assets/689fd867-a8a4-4774-a1c5-ba40bbb9b2aa)


![图片](https://github.com/user-attachments/assets/ac497753-91a5-4c95-8f81-bbd30101d250)

借用阿里云官网的例子，来简单介绍CDN的工作原理。假设通过CDN加速的域名为`www.a.com`，接入CDN网络，开始使用加速服务后，当终端用户（北京）发起HTTP请求时，处理流程如下：

1. 当终端用户（北京）向`www.a.com`下的指定资源发起请求时，首先向LDNS（本地DNS）发起域名解析请求。
2. LDNS检查缓存中是否有`www.a.com`的IP地址记录。如果有，则直接返回给终端用户；如果没有，则向授权DNS查询。
3. 当授权DNS解析`www.a.com`时，返回域名CNAME [www.a.tbcdn.com](https://github.com/Aloner63/Aloner63.github.io/issues/www.a.tbcdn.com)对应IP地址。
4. 域名解析请求发送至阿里云DNS调度系统，并为请求分配最佳节点IP地址。
5. LDNS获取DNS返回的解析IP地址。
6. 用户获取解析IP地址。
7. 用户向获取的IP地址发起对该资源的访问请求。

- 如果该IP地址对应的节点已缓存该资源，则会将数据直接返回给用户，例如，图中步骤7和8，请求结束。
- 如果该IP地址对应的节点未缓存该资源，则节点向源站发起对该资源的请求。获取资源后，结合用户自定义配置的缓存策略，将资源缓存至节点，例如，图中的北京节点，并返回给用户，请求结束。

从这个例子可以了解到：

1. CDN的加速资源是跟域名绑定的。
2. 通过域名访问资源，首先是通过DNS分查找离用户最近的CDN节点（边缘服务器）的IP
3. 通过IP访问实际资源时，如果CDN上并没有缓存资源，则会到源站请求资源，并缓存到CDN节点上，这样，用户下一次访问时，该CDN节点就会有对应资源的缓存了。

简单讲，CDN就是通过将站点内容发布至遍布全球的海量加速节点，使其用户可就近获取所需内容。

CDN主要解决这么些问题:

- 物理距离远，多次网络转发，延时高不稳定;
- 所在运营商不同，需运营商之间转发绕行;
- 网络带宽处理能力有限，海量请求时，响应速度与可用性降低。



##### 为什么要用CDN？

如果你在经营一家网站，那你应该知道几点因素是你制胜的关键：

- 内容有吸引力
- 访问速度快
- 支持频繁的用户互动
- 可以在各处浏览无障碍

另外，你的网站必须能在复杂的网络环境下运行，考虑到全球的用户访问体验。你的网站也会随着使用越来越多的对象（如图片、帧、CSS及APIs）和形形色色的动作（分享、跟踪）而系统逐渐庞大。所以，系统变慢带来用户的流失。

Google及其它网站的研究表明，一个网站每慢一秒钟，就会丢失许多访客，甚至这些访客永远不会再次光顾这些网站。可以想像，如果网站是你的盈利渠道或是品牌窗口，那么网站速度慢将是一个致命的打击。

这就是你使用CDN的第一个也是最重要的原因：**为了加速网站的访问**



##### CDN与传统网站的区别

CDN主要功能是在不同的地点缓存内容，通过负载均衡技术，将用户的请求定向到最合适的缓存服务器上去获取内容，比如说，是北京的用户，我们让他访问北京的节点，深圳的用户，我们让他访问深圳的节点。通过就近访问，加速用户对网站的访问。解决Internet网络拥堵状况，提高用户访问网络的响应速度。


#### 反向代理

幕后的“交通警察”

**反向代理** 就像一个位于服务器和客户端之间的“交通警察”，它接收来自客户端的请求，然后根据一定的规则将请求转发给内部网络上的服务器。对于客户端来说，它只知道反向代理的IP地址，并不知道在它背后的服务器集群的存在。

##### 反向代理的工作原理

1. **接收请求：** 客户端向反向代理服务器发送请求。
2. **转发请求：** 反向代理服务器根据配置的规则，将请求转发给相应的后端服务器。
3. **接收响应：** 后端服务器处理请求后，将响应返回给反向代理服务器。
4. **返回响应：** 反向代理服务器将收到的响应再转发给客户端。

### 反向代理的作用

- **负载均衡：** 将大量的请求分发到多个后端服务器上，提高系统的并发处理能力。
- **缓存静态内容：** 缓存静态资源（如图片、CSS、JS文件），减少后端服务器的压力，提高响应速度。
- **隐藏后端服务器：** 客户端只能看到反向代理服务器的IP地址，保护了后端服务器的真实IP。
- **增强安全性：** 可以对请求进行过滤和防护，防止攻击。
- **实现虚拟主机：** 一个反向代理服务器可以为多个域名提供服务。

##### 反向代理的常见应用场景

- **大型网站：** 分散访问压力，提高网站的可用性。
- **CDN（内容分发网络）：** 将静态内容缓存到离用户最近的服务器上，加速访问。
- **API网关：** 统一管理和保护API接口。
- **微服务架构：** 将请求路由到不同的微服务。

##### 常用的反向代理软件

- **Nginx：** 高性能、轻量级，适合高并发场景。
- **Apache HTTP Server：** 功能强大，模块丰富，适合传统Web应用。
- **HAProxy：** 专注于TCP和HTTP负载均衡。
- **Varnish：** 高性能HTTP加速器。

##### 形象比喻

- 反向代理就像一个酒店的前台

  ：

  - 客人（客户端）向前台（反向代理）提出入住要求。
  - 前台根据客人的需求，安排客人入住不同的房间（后端服务器）。
  - 客人只知道前台，不知道房间的具体位置。



#### 反向代理和CDN的异同

##### **工作原理的不同**

- **CDN（内容分发网络）**：
  CDN的主要目的是通过**缓存静态内容**（如图片、CSS、JS、视频等）到多个分布在全球或局部的节点服务器，让用户可以从**最近的服务器**获取这些内容，从而减少网络延迟、提高访问速度。CDN的节点服务器位于全球多个地点，用户的请求会被路由到**最近的CDN节点**，而不是直接访问网站的原始服务器。

  **工作机制：**

  - 网站的静态资源会被复制并分发到各地的CDN服务器。
  - 当用户访问网站时，CDN会根据用户所在的地理位置，将其请求路由到最近的节点，从缓存中获取内容并返回给用户。
  - 如果CDN节点上没有缓存的资源，才会去原始服务器请求并缓存。

**反向代理**：
反向代理服务器是位于**用户和原始服务器之间**的中间服务器。它主要的任务是**代替原始服务器接收用户请求**，然后将请求转发给原始服务器处理，原始服务器返回数据后，反向代理再将数据返回给用户。反向代理一般放置在与原始服务器比较接近的地方（通常是同一个数据中心），它不存储数据，只是作为请求的中转站。

**工作机制：**

- 当用户访问一个网站时，用户的请求首先到达反向代理服务器。
- 反向代理服务器根据请求的内容，将请求转发给适当的原始服务器。
- 原始服务器处理完请求后，将响应结果返回给反向代理，反向代理再返回给用户。



#####  **主要用途的不同**

- **CDN的主要用途**：
  - **加速静态内容的传输**：CDN通过将静态资源（如图片、视频等）缓存到多个分布式节点，可以让用户从最近的节点获取资源，从而加快访问速度。
  - **减轻原始服务器的负担**：CDN缓存静态资源后，大量的用户请求都能从CDN获取，减少了对原始服务器的直接访问，降低服务器的压力。
  - **全球加速**：CDN可以让不同地理位置的用户都能快速访问内容，解决跨国、跨地区访问慢的问题。
- **反向代理的主要用途**：
  - **隐藏原始服务器的真实IP**：通过反向代理，用户只会看到代理服务器的IP地址，原始服务器的IP可以被隐藏，增加安全性。
  - **负载均衡**：反向代理可以将用户的请求分配到不同的原始服务器上，确保每台服务器的负载是均衡的，提高系统的整体性能。
  - **安全防护**：反向代理可以充当一个防护屏障，通过限制或过滤恶意请求，保护后端的原始服务器免受攻击。
  - **SSL卸载**：反向代理服务器可以处理SSL加密/解密的任务，减轻原始服务器的负担。

##### **缓存机制的不同**

- **CDN**：
  CDN的核心机制是缓存。CDN节点会将静态内容缓存起来，当用户访问这些内容时，可以直接从缓存中获取，而不需要每次都请求原始服务器。这是CDN加速的关键所在。
- **反向代理**：
  反向代理服务器一般**不主动缓存**数据，它的主要作用是**中转请求**。但在某些情况下，反向代理也可以配置缓存功能，尤其是对某些静态资源或重复的动态请求进行缓存，以进一步减轻原始服务器的压力。

##### **部署位置的不同**

- **CDN**：
  CDN的节点服务器通常**分布在全球或区域的多个地方**，越靠近用户的节点越有利于加速。因此，CDN是一个全球分布的网络。
- **反向代理**：
  反向代理服务器通常**位于原始服务器的前端**，可以在同一个数据中心或距离原始服务器较近的地方部署。它的目的是作为一个网关来接收所有的用户请求，并根据需要将这些请求转发给后端的原始服务器。
