I2C 是一种串行通信协议。

多用于：

- 单片机与外围设备之间的通信
- 电路板上多个集成电路（IC）之间的通信
- 不同电路板之间的近距离通信

基本特点：

- 只需要两根线：SDA（串行数据线）：用于传输数据。 SCL（串行时钟线）：用于同步通信
- 支持多主机和多从机
- 每个设备都有唯一的地址
- 通信速率可调（标准模式100kbps，快速模式400kbps，高速模式3.4Mbps）

基本概念：

1. **主机(Master)**：发起通信的设备
2. **从机(Slave)**：响应主机请求的设备
3. **地址**：每个从机都有唯一的7位或10位地址
![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/fe191640f3d748f982888852d6cdd49f.png)




注意，按照实际设计中经验大概是不超过 8 个器件。

这是由 I2C 地址决定：8 位地址，减去 1 位广播地址，是 7 位地址，2 7 = 128 2^7=1282 
7
 =128，但是地址 0x00 不用，那就是 127 个地址， 所以理论上可以挂 127 个从器件。但是，I2C 协议没有规定总线上设备最大数目，但是规定了总线电容不能超过 400pF。管脚都是有输入电容的，PCB 上也会有寄生电容，所以会有一个限制。实际设计中经验值大概是不超过 8 个器件。

总线之所以规定电容大小是因为，`I2C` 的 OD 要求外部有电阻上拉，电阻和总线电容产生了一个 RC 延时效应，电容越大信号的边沿就越缓，有可能带来信号质量风险。传输速度越快，信号的窗口就越小，上升沿下降沿时间要求更短更陡峭，所以 RC 乘积必须更小。

##### 起始位

如下图，就是 `I2C` 通信起始标志，通过这个起始位就可以告诉 `I2C` 从机，主机要开始进行 `I2C` 通信了。在 SCL 为高电平的时候，SDA 出现下降沿就表示为起始位：

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/c3cd8923426143478448754849264f4d.png)

##### 停止位

如下图，就是停止 `I2C` 通信的标志位，和起始位的功能相反。在 SCL 位高电平的时候，SDA 出现上升沿就表示为停止位：

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/03e520bafe8c4e1db48b5dafee30a9b1.png)

##### 数据传输

如下图，`I2C` 总线在数据传输的时候要保证在 SCL 高电平期间，SDA 上的数据稳定，即 SDA 上的数据变化只能在 SCL 低电平期间发生：

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/0a4d21a135fb44ddba63b02d7ffd0849.png)

##### 应答信号

当 I2C 主机发送完 8 位数据以后会将 SDA 设置为输入状态，等待 I2C 从机应答，也就是等到 I2C 从机告诉主机它接收到数据了。应答信号是由从机发出的，主机需要提供应答信号所需的时钟，主机发送完 8 位数据以后紧跟着的一个时钟信号就是给应答信号使用的。从机通过将 SDA 拉低来表示发出应答信号，表示通信成功，否则表示通信失败。

##### I2C设备地址格式

I2C 设备的地址为 8 位，但是时序操作时最后一位不属于地址，而是 R/W 状态位。所以有用的是前 7 位，使用时地址整体右移一位处理即可。

除此之位，一个设备地址的前四位是固定的，是厂家用来表示设备类型的：

- 比如接口为 I2C 的温度传感器类设备地址前四位一般为 1001 即 9X；
- EEPROM 存储器地址前四位一般为 1010 即 AX；
- oled屏地址前四位一般为 0111 即 7X 等。

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/f6d13e623f7348c98d82c855cd68a6ca.png)

下面结合图例，将前面所提到的信息整合一下：

起始信号
当 SCL 为高电平期间，SDA 由高到低的跳变，起始信号是一种电平跳变时序信号，而不是一个电平信号。该信号由主机发出，在起始信号产生后，总线就处于被占用状态，准备数据传输。

停止信号
当 SCL 为高电平期间，SDA 由低到高的跳变；停止信号也是一种电平跳变时序信号，而不是一个电平信号。该信号由主机发出，在停止信号发出后，总线就处于空闲状态。

应答信号
发送器每发送一个字节，就在时钟脉冲 9 期间释放数据线，由接收器反馈一个应答信号。应答信号为低电平时，规定为有效应答位（ACK 简称应答位），表示接收器已经成功地接收了该字节；应答信号为高电平时，规定为非应答位（NACK），一般表示接收器接收该字节没有成功。

数据有效性
IIC 总线进行数据传送时，时钟信号为高电平期间，数据线上的数据必须保持稳定，只有在时钟线上的信号为低电平期间，数据线上的高电平或低电平状态才允许变化。数据在 SCL 的上升沿到来之前就需准备好。并在下降沿到来之前必须稳定。

数据传输
在 IIC 总线上传送的每一位数据都有一个时钟脉冲相对应（或同步控制），即在 SCL 串行时钟的配合下，在 SDA 上逐位地串行传送每一位数据。数据位的传输是边沿触发。

空闲状态
IIC 总线的 SDA 和 SCL 两条信号线同时处于高电平时，规定为总线的空闲状态。此时各个器件的输出级场效应管均处在截止状态，即释放总线，由两条信号线各自的上拉电阻把电平拉高。

##### I2C写时序

要在 I2C 总线上写入，主机将在总线上发送：一个启动开始标志、从机地址、最后一位（R/W位）设置为 0，这表示写入。

从设备发送 ACK 响应确认后，主设备将发送其希望写入的寄存器的寄存器地址。从设备将再次确认，让主设备知道它已准备就绪。在此之后，主机将开始向从机发送寄存器数据，直到主机发送了它需要的所有数据（有时这只是一个字节），并且主机将以停止条件终止传输。

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/338204c22a62447f8d0e2495c849ed45.png)

具体步骤为：

```
开始信号。
发送 I2C 设备地址，每个 I2C 器件都有一个设备地址，通过发送具体的设备地址来决定访问哪个 I2C 器件。这是一个 8 位的数据，其中高 7 位是设备地址，最后 1 位是读写位（为 1 的话表示这是一个读操作，为 0 的话表示这是一个写操作）。
读写控制位，因为是向 I2C 从设备发送数据，因此是写信号 0。
从机发送的 ACK 应答信号。
重新发送开始信号。
发送要写入数据的寄存器地址。
从机发送的 ACK 应答信号。
发送要写入寄存器的数据。
从机发送的 ACK 应答信号。
停止信号。
```

##### I2C读时序

主机为了读取从设备的数据，主机必须首先指出希望从从设备的哪个寄存器读取数据。这是由主机写入从设备的“写操作”类似的方式开始传输，通过发送 R/W 位等于 0 的地址（表示写入），然后是它希望从中读取的寄存器地址来完成的。

一旦从设备确认该寄存器地址，主机将再次发送启动条件，然后发送从设备地址，R/W 位设置为 1（表示读取）。这一次，从设备将确认读取请求，主机释放 SDA 总线，但将继续向从设备提供时钟。在这部分事务中，主机将成为主“接收器”，将成为从“发射器”。

主机将继续发送时钟脉冲 SCL，但会释放 SDA，以便从设备可以传输数据。在数据的每个字节结束时，主机将向从设备发送 ACK，让从设备知道它已准备好接收更多数据。一旦主机接收到预期的字节数，它将发送一个 NACK，向从设备发送信号以停止通信并释放总线。之后，主机将设置停止条件。

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/230b0726c33d47dea239026854945834.png)

`I2C` 单字节读时序比写时序要复杂一点，读时序分为四个步骤，第一步是发送设备地址，第二步是发送要读取的寄存器地址，第三步重新发送设备地址，最后一步就是 `I2C` 从器件输出要读取的寄存器值，我们具体来看一下这步。

```
主机发送起始信号。
主机发送要读取的 I2C 从设备地址。
读写控制位，因为是向 I2C 从设备发送数据，因此是写信号 0。
从机发送的 ACK 应答信号。
重新发送 START 信号。
主机发送要读取的寄存器地址。
从机发送的 ACK 应答信号。
重新发送 START 信号。
重新发送要读取的 I2C 从设备地址。
读写控制位，这里是读信号 1，表示接下来是从 I2C 从设备里面读取数据。
从机发送的 ACK 应答信号。
从 I2C 器件里面读取到的数据。
主机发出 NACK 信号，表示读取完成，不需要从机再发送 ACK 信号了。
主机发出 STOP 信号，停止 I2C 通信。
```

![img](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/ab98eff9a1cf446abfca5509443b7d07.png)

##### 时钟同步和仲裁

###### 时钟同步

在 I2C 总线上传送信息时的时钟同步信号是由挂接在 SCL 线上的所有器件的 逻辑“与” 完成的。即如果有多个主机同时产生时钟，那么只有所有主机都发送高电平时，SCL 上才表现为高电平，否则 SCL 都表现为低电平。

SCL 线上由高电平到低电平的跳变将影响到这些器件，一旦某个器件的时钟信号下跳为低电平，将使 SCL 线一直保持低电平，使 SCL 线上的所有器件开始低电平期。此时，低电平周期短的器件的时钟由低至高的跳变并不能影响 SCL 线的状态，于是这些器件将进入高电平等待的状态。当所有器件的时钟信号都上跳为高电平时，低电平期结束，SCL 线被释放返回高电平，即所有的器件都同时开始它们的高电平期。其后，第一个结束高电平期的器件又将 SCL 线拉成低电平。这样就在 SCL 线上产生一个同步时钟。

可见，时钟低电平时间由时钟低电平期最长的器件确定，而时钟高电平时间由时钟高电平期最短的器件确定。下面是对它的通俗解释：

```
想象一个团队在玩一个游戏：大家一起控制一盏灯的开关，但只有当所有人都同意“开灯”或“关灯”时，灯的状态才会改变。

    每个人手里都有一个开关（对应 I²C 的开漏输出器件）。
    灯的状态是“关”还是“开”，由所有人的开关状态综合决定：
        只要有人按下开关，灯就会灭（低电平）。
        只有当所有人都松开开关，灯才会亮（高电平）。
```

###### 时钟仲裁

总线仲裁与时钟同步类似，当所有主机在 SDA 上都写 1 时，SDA 的数据才是 1，只要有一个主机写 0，那此时 SDA 上的数据就是 0。

一个主机每发送一个 bit 数据，在 SCL 为高电平时，就检查 SDA 的电平是否和发送的数据一致，如果不一致，这个主机便知道自己输掉了仲裁，然后停止向 SDA 写数据。也就是说，如果主机一致检查到总线上数据和自己发送的数据一致，则继续传输，这样在仲裁过程中就保证了赢得仲裁的主机不会丢失数据。

输掉仲裁的主机在检测到自己输了之后也就不再产生时钟脉冲，并且要在总线空闲时才能重新传输。

仲裁的过程可能要经过多个 bit 的发送和检查，实际上两个主机如果发送的时序和数据完全一样，则两个主机都能正常完成整个数据传输。

待补充。。。