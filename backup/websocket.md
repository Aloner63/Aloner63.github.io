<html><body>
<!--StartFragment--><p>WebSocket 是一种网络通信协议，旨在通过持久的、全双工（双向）通信连接实现实时数据交换。它在 Web 浏览器和服务器之间提供了一条持久的、双向的通信通道，使得数据能够实时地从服务器推送到客户端，而不需要客户端每次都发起请求。</p><h3>WebSocket 的特点</h3><ol><li><p><strong>持久连接</strong>：
WebSocket 建立连接后，客户端和服务器之间的连接保持长时间的打开状态，直到其中一方主动关闭连接。与传统的 HTTP 请求/响应模式不同，WebSocket 不需要每次通信都重新建立连接，从而减少了通信延迟和资源消耗。</p></li><li><p><strong>双向通信</strong>：
WebSocket 是双向的，这意味着数据不仅可以从服务器发送到客户端，还可以从客户端发送到服务器。这对于实时应用（如即时聊天、在线游戏、股票交易等）非常重要。</p></li><li><p><strong>实时性</strong>：
WebSocket 实现了低延迟的实时通信，当服务器或客户端有数据更新时，可以即时地推送到对方，无需等待请求。</p></li><li><p><strong>轻量</strong>：
在 WebSocket 协议中，头部信息非常小，通信开销比 HTTP 更低。建立连接后，客户端和服务器之间的数据交换不再需要额外的 HTTP 请求头，从而节省带宽。</p></li><li><p><strong>基于标准</strong>：
WebSocket 是由 IETF（互联网工程任务组）标准化的，定义在 <a rel="noopener" target="_new"><span>RFC</span><span> 6455</span></a> 中。</p></li></ol><h3>WebSocket 与传统 HTTP 的区别</h3>
特性 | HTTP | WebSocket
-- | -- | --
连接方式 | 每次请求都需要建立新的连接 | 建立一次连接后保持长时间开放
通信方向 | 客户端向服务器发起请求 | 双向通信，客户端和服务器都可以主动发送数据
数据交换方式 | 客户端发起请求，服务器响应 | 客户端和服务器可以随时互相发送消息
性能 | 每次请求都需要额外的头部信息 | 建立连接后通信开销更小，延迟更低

<h3>WebSocket 的应用场景</h3><ol><li><p><strong>即时通讯</strong>：
WebSocket 非常适合用于即时消息传递系统，如聊天应用（比如微信、Slack、Facebook Messenger 等），因为它允许即时的双向通信。</p></li><li><p><strong>在线游戏</strong>：
许多多人在线游戏使用 WebSocket 来保持客户端与服务器之间的实时连接，以便实时传输游戏数据（如玩家位置、得分、动作等）。</p></li><li><p><strong>股票和金融交易</strong>：
对于股票行情、外汇交易、期货市场等领域的实时数据推送，WebSocket 能够提供快速的实时更新，让交易员实时获得最新的市场信息。</p></li><li><p><strong>实时监控和报警系统</strong>：
WebSocket 适用于需要实时监控的应用，如实时传感器数据监控、网站分析数据更新、服务运行状态监控等。</p></li><li><p><strong>协作工具</strong>：
用于实时协作应用（如 Google Docs 或在线白板），让多个用户可以同步编辑或查看内容。</p></li></ol><h3>WebSocket 工作原理</h3><p>WebSocket 协议的工作过程如下：</p><ol><li><p><strong>握手（Handshake）</strong>：
客户端通过发起 HTTP 请求向服务器请求建立 WebSocket 连接。这个请求是一个特殊的 HTTP 请求，包含了 WebSocket 协议的标识符（<code>Upgrade</code> 和 <code>Connection</code> 头部信息）。如果服务器支持 WebSocket，它会返回一个 101 状态码，表示协议升级为 WebSocket，之后 WebSocket 连接建立成功。</p><ul><li><p>

请求示例：

```
GET /chat HTTP/1.1
Host: example.com
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Key: x3JJHMbDL1EzLkh9K8bM9K3KF6gW7mow
Sec-WebSocket-Version: 13
```
响应示例：
```
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: x3JJHMbDL1EzLkh9K8bM9K3KF6gW7mow
```
数据传输： 一旦 WebSocket 连接建立，客户端和服务器之间就可以开始通过 WebSocket 协议进行数据交换。数据通过 WebSocket 帧（frame）传输，支持文本、二进制数据等多种格式。

关闭连接： 当通信结束时，任意一方（客户端或服务器）可以发起关闭连接的请求。WebSocket 协议提供了一个优雅的关闭过程，确保数据不会丢失。

### WebSocket 与其他技术的对比

| 技术           | WebSocket                      | HTTP                                   | Server-Sent Events (SSE)             |
| -------------- | ------------------------------ | -------------------------------------- | ------------------------------------ |
| **连接方式**   | 持久连接，双向通信             | 短连接，单向请求/响应                  | 持久连接，单向通信（服务器到客户端） |
| **实时性**     | 高实时性                       | 延迟较高（每次请求都需要重新建立连接） | 比 HTTP 更实时，但仅限单向推送       |
| **客户端支持** | 广泛支持（现代浏览器和客户端） | 几乎所有浏览器都支持                   | 主流浏览器支持较好                   |
| **应用场景**   | 双向实时通信，聊天，游戏等     | Web 页面的请求和响应，文件下载等       | 实时数据推送，通知，直播等           |


**既然websocket那么强大了，为什么还有有http？**

**WebSocket 的优势：**

- **全双工通信：** 客户端和服务器可以同时发送和接收数据，实现真正的实时通信。
- **持久连接：** 一旦建立连接，就可以保持较长时间的连接，避免频繁的连接建立和断开。
- **低延迟：** 相比 HTTP 协议，WebSocket 的延迟更低，更适合实时应用。
- **高效：** WebSocket 协议头很小，数据传输效率高。

**HTTP 的优势：**

- **成熟稳定：** HTTP 协议经过多年的发展，已经非常成熟稳定，有广泛的兼容性。
- **简单易用：** HTTP 协议相对简单，容易理解和实现。
- **广泛支持：** 所有浏览器都支持 HTTP 协议，而 WebSocket 的支持程度相对较低。

**浏览器为什么不只使用 WebSocket？**

- **兼容性问题：** 虽然 WebSocket 已经得到了广泛支持，但不同浏览器对 WebSocket 的实现可能存在差异，需要考虑兼容性问题。
- **HTTP 的优势：** 在一些场景下，HTTP 的简单性和成熟性仍然具有优势，例如传统的请求-响应模式。
- **WebSocket 的复杂性：** WebSocket 的实现相对复杂，需要考虑连接管理、数据帧格式、错误处理等问题。

**浏览器如何结合 HTTP 和 WebSocket？**

- **HTTP 用于建立连接：** 浏览器通常使用 HTTP 协议来建立 WebSocket 连接，通过 HTTP 的 Upgrade 机制将连接升级为 WebSocket 连接。
- **WebSocket 用于实时通信：** 一旦建立 WebSocket 连接，就可以进行实时通信，实现诸如聊天、实时数据更新等功能。
- **HTTP 用于传输静态资源：** 浏览器仍然使用 HTTP 协议来获取 HTML、CSS、JavaScript 等静态资源。

**总结**

WebSocket 和 HTTP 各有优缺点，浏览器将两者结合使用，可以更好地满足不同的应用需求。在选择使用哪种协议时，需要根据具体的应用场景进行权衡。

**什么时候使用 WebSocket？**

- 需要实时双向通信的应用，例如在线聊天、实时数据可视化等。
- 需要低延迟、高性能的通信。
- 需要建立长连接，避免频繁的连接建立和断开。

**什么时候使用 HTTP？**

- 请求-响应模式的应用。
- 传输静态资源。
- 对兼容性要求较高，或者需要支持老旧浏览器。

**总之，WebSocket 和 HTTP 是互补得，而不是对立的。在实际开发中，应该根据具体需求选择合适的协议，或者将两者结合起来使用。**