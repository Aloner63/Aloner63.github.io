
[TOC]

## 11.FreeRTOS任务相关的其他API函数

### 一、FreeRTOS任务相关的其他API函数介绍

#### 1、FreeRTOS任务相关API函数介绍(部分常用的)

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BB%BB%E5%8A%A1%E7%9B%B8%E5%85%B3%E7%9A%84%E5%85%B6%E4%BB%96API%E5%87%BD%E6%95%B0.png)

### 二、任务状态查询API函数

#### 1、获取任务优先级函数

答：

```C
UBaseType_t  uxTaskPriorityGet(  const TaskHandle_t xTask  )
```

此函数用于获取指定任务的任务优先级，使用该函数需要将宏 INCLUDE_uxTaskPriorityGet 置1。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E4%BB%BB%E5%8A%A1%E4%BC%98%E5%85%88%E7%BA%A7%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E4%BB%BB%E5%8A%A1%E4%BC%98%E5%85%88%E7%BA%A7%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 2、修改任务优先级函数

答：

```c
void vTaskPrioritySet( TaskHandle_t xTask , UBaseType_t uxNewPriority )
```

此函数用于改变某个任务的任务优先级，使用该函数需将宏 INCLUDE_vTaskPrioritySet 为 1 。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BF%AE%E6%94%B9%E4%BB%BB%E5%8A%A1%E4%BC%98%E5%85%88%E7%BA%A7%E5%87%BD%E6%95%B0.png)

#### 3、获取系统任务数量函数

答：

```c
UBaseType_t   uxTaskGetNumberOfTasks( void )
```

此函数用于获取系统中任务的任务数量。

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E7%B3%BB%E7%BB%9F%E4%BB%BB%E5%8A%A1%E6%95%B0%E9%87%8F%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 4、获取系统中所有任务状态信息函数

答：

```C
UBaseType_t   uxTaskGetSystemState(   TaskStatus_t * const pxTaskStatusArray,
                                      const UBaseType_t uxArraySize,
                                      configRUN_TIME_COUNTER_TYPE * const pulTotalRunTime   )
```

此函数用于获取系统中所有任务的任务状态信息，使用该函数需将宏 configUSE_TRACE_FACILITY 置 1。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E7%B3%BB%E7%BB%9F%E4%B8%AD%E6%89%80%E6%9C%89%E4%BB%BB%E5%8A%A1%E7%8A%B6%E6%80%81%E4%BF%A1%E6%81%AF%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E7%B3%BB%E7%BB%9F%E4%B8%AD%E6%89%80%E6%9C%89%E4%BB%BB%E5%8A%A1%E7%8A%B6%E6%80%81%E4%BF%A1%E6%81%AF%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

参数成员pxTaskStatusArray的结构体：

```C
typedef struct xTASK_STATUS
{
    TaskHandle_t                   xHandle;                     /* 任务句柄 */ 
    const char *                   pcTaskName;                  /* 任务名 */
    UBaseType_t                    xTaskNumber;                 /* 任务编号 */
    eTaskStatee                    CurrentState;                /* 任务状态 */
    UBaseType_t                    uxCurrentPriority;           /* 任务优先级 */
    UBaseType_t                    uxBasePriority;              /* 任务原始优先级*/
    configRUN_TIME_COUNTER_TYPE    ulRunTimeCounter;            /* 任务运行时间*/
    StackType_t *                  pxStackBase;                 /* 任务栈基地址 */
    configSTACK_DEPTH_TYPE         usStackHighWaterMark;        /* 任务栈历史剩余最小值 */
} TaskStatus_t;
```

#### 5、获取系统中单个任务状态信息函数

答：

```c
void vTaskGetInfo(  TaskHandle_t     xTask,
                    TaskStatus_t *   pxTaskStatus,
                    BaseType_t       xGetFreeStackSpace,
                    eTaskState       eState  )  
```

此函数用于获取指定的单个任务的状态信息，使用该函数需将宏 configUSE_TRACE_FACILITY 置 1 

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E7%B3%BB%E7%BB%9F%E4%B8%AD%E5%8D%95%E4%B8%AA%E4%BB%BB%E5%8A%A1%E7%8A%B6%E6%80%81%E4%BF%A1%E6%81%AF%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

参数成员eState的结构体：

```C
typedef enum
{
    eRunning = 0,       /* 运行态 */
    eReady,             /* 就绪态 */
    eBlocked,           /* 阻塞态 */
    eSuspended,         /* 挂起态 */
    eDeleted,           /* 任务被删除 */
    eInvalid            /* 无效 */ 
} eTaskState;
```

#### 6、获取当前任务的任务句柄函数

答：

```c
TaskHandle_t    xTaskGetCurrentTaskHandle( void )
```

此函数用于获取当前任务的任务句柄， 使用该函数需将宏 INCLUDE_xTaskGetCurrentTaskHandle 置 1。

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E5%BD%93%E5%89%8D%E4%BB%BB%E5%8A%A1%E7%9A%84%E4%BB%BB%E5%8A%A1%E5%8F%A5%E6%9F%84%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

#### 7、通过任务名获取任务句柄函数

答：

```c
TaskHandle_t xTaskGetHandle(const char * pcNameToQuery); 
```

此函数用于通过任务名获取任务句柄 ， 使用该函数需将宏 INCLUDE_xTaskGetHandle 置 1。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%80%9A%E8%BF%87%E4%BB%BB%E5%8A%A1%E5%90%8D%E8%8E%B7%E5%8F%96%E4%BB%BB%E5%8A%A1%E5%8F%A5%E6%9F%84%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%80%9A%E8%BF%87%E4%BB%BB%E5%8A%A1%E5%90%8D%E8%8E%B7%E5%8F%96%E4%BB%BB%E5%8A%A1%E5%8F%A5%E6%9F%84%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 8、获取指定任务的任务堆栈历史最小剩余函数

```c
UBaseType_t    uxTaskGetStackHighWaterMark( TaskHandle_t  xTask )
```

此函数用于获取指定任务的任务栈历史最小剩余堆栈；使用该函数需将宏 INCLUDE_uxTaskGetStackHighWaterMark 置 1。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E6%8C%87%E5%AE%9A%E4%BB%BB%E5%8A%A1%E7%9A%84%E4%BB%BB%E5%8A%A1%E5%A0%86%E6%A0%88%E5%8E%86%E5%8F%B2%E6%9C%80%E5%B0%8F%E5%89%A9%E4%BD%99%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E6%8C%87%E5%AE%9A%E4%BB%BB%E5%8A%A1%E7%9A%84%E4%BB%BB%E5%8A%A1%E5%A0%86%E6%A0%88%E5%8E%86%E5%8F%B2%E6%9C%80%E5%B0%8F%E5%89%A9%E4%BD%99%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 9、查询指定任务运行状态函数

答：

```c
eTaskState    eTaskGetState(TaskHandle_t xTask)
```

此函数用于查询某个任务的运行状态，使用此函数需将宏 INCLUDE_eTaskGetState 置1 

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%9F%A5%E8%AF%A2%E6%8C%87%E5%AE%9A%E4%BB%BB%E5%8A%A1%E8%BF%90%E8%A1%8C%E7%8A%B6%E6%80%81%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%9F%A5%E8%AF%A2%E6%8C%87%E5%AE%9A%E4%BB%BB%E5%8A%A1%E8%BF%90%E8%A1%8C%E7%8A%B6%E6%80%81%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

参数成员xTask的结构体：

```c
typedef enum
{
    eRunning = 0,	/* 运行态 */
    eReady,         /* 就绪态 */
    eBlocked,       /* 阻塞态 */
    eSuspended,     /* 挂起态 */
    eDeleted,       /* 任务被删除 */
    eInvalid        /* 无效 */ 
} eTaskState;
```

#### 10、以“表格”的形式获取系统中任务信息函数

答：

```c
void   vTaskList(char * pcWriteBuffer)
```

此函数用于以“表格”的形式获取系统中任务的信息 ；

使用此函数需将宏 configUSE_TRACE_FACILITY 和configUSE_STATS_FORMATTING_FUNCTIONS 置1 。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BB%A5%E2%80%9C%E8%A1%A8%E6%A0%BC%E2%80%9D%E7%9A%84%E5%BD%A2%E5%BC%8F%E8%8E%B7%E5%8F%96%E7%B3%BB%E7%BB%9F%E4%B8%AD%E4%BB%BB%E5%8A%A1%E4%BF%A1%E6%81%AF%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

表格内容：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%A1%A8%E6%A0%BC%E5%86%85%E5%AE%B9.png)

- Name：创建任务的时候给任务分配的名字。
- State：任务的壮态信息， B 是阻塞态， R 是就绪态， S 是挂起态， D 是删除态。
- Priority：任务优先级。
- Stack： 任务堆栈的“高水位线”，就是堆栈历史最小剩余大小。
- Num：任务编号，这个编号是唯一的，当多个任务使用同一个任务名的时候可以通过此编号来做区分。

### 三、任务时间统计API函数

#### 1、任务时间统计函数

答：

```c
void    vTaskGetRunTimeStats( char * pcWriteBuffer ) 
```

此函数用于统计任务的运行时间信息，使用此函数需将宏 configGENERATE_RUN_TIME_STAT 、configUSE_STATS_FORMATTING_FUNCTIONS 置1。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BB%BB%E5%8A%A1%E6%97%B6%E9%97%B4%E7%BB%9F%E8%AE%A1API%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

时间统计表格：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%97%B6%E9%97%B4%E7%BB%9F%E8%AE%A1%E8%A1%A8%E6%A0%BC.png)

- Task：任务名称。
- Abs Time：任务实际运行的总时间(绝对时间)。
- %Time：占总处理时间的百分比。

#### 2、时间统计API函数使用流程

答：

1. 将宏 configGENERATE_RUN_TIME_STATS 置1 。
2. 将宏 configUSE_STATS_FORMATTING_FUNCTIONS 置1 。
3. 当将此宏 configGENERATE_RUN_TIME_STAT 置1之后，还需要实现2个宏定义：
   1. `portCONFIGURE_TIMER_FOR_RUNTIME_STATE()`：用于初始化用于配置任务运行时间统计的时基定时器；
      注意：这个时基定时器的计时精度需高于系统时钟节拍精度的10至100倍！
   2. `portGET_RUN_TIME_COUNTER_VALUE()`：用于获取该功能时基硬件定时器计数的计数值 。

## 12.FreeRTOS时间管理

### 一、延时函数介绍

#### 1、FreeRTOS的延时函数

答：FreeRTOS有两种延时函数：相对延时函数 和 绝对延时函数。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%BB%B6%E6%97%B6%E5%87%BD%E6%95%B0.png)

- **相对延时**：指每次延时都是从执行函数vTaskDelay()开始，直到延时指定的时间结束。
- **绝对延时**：指将整个任务的运行周期看成一个整体，适用于需要按照一定频率运行的任务。

注意：一般来说，绝对延时中的主体任务运行所需时间必须比绝对延时时间小。

#### 2、相对延时和绝对延时的区别

答：

**相对延时**：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E7%9B%B8%E5%AF%B9%E5%BB%B6%E6%97%B6.png)

**绝对延时**：

<img src="https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E7%BB%9D%E5%AF%B9%E5%BB%B6%E6%97%B6.png" style="zoom:150%;" />

### 二、延时函数解析

#### 1、相对延时函数内部解析

答：

1. 判断延时时间是否大于0，大于0才有效。
2. 挂起调度器。
3. 将当前正在运行的任务从就绪列表移除，添加到阻塞列表prvAddCurrentTaskToDelayedList( )。
   1. 将该任务从就绪列表中移除。
   2. 如果使能挂起操作，并且延时时间为0XFFFF FFFF，并且xCanBlockIndefinitely等于pdTRUE，就代表此时是一直等，相当于挂起，所以添加到挂起列表。
   3. 如果延时时间小于0XFFFF FFF。
      - 记录阻塞超时时间，并记录在列表项值里（通过该值确定插入阻塞列表的位置）。
      - 如果阻塞超时时间溢出，将该任务状态列表项添加到溢出阻塞列表。
      - 如果没溢出，则将该任务状态列表项添加到阻塞列表，并判断阻塞超时时间是否小于下一个阻塞超时时间，是的话就更新当前这个时间为下一个阻塞超时时间  
4. 恢复任务调度器。
5. 进行一次任务切换。

#### 2、延时函数的流程

答：

- 正在运行的任务。
- 调用延时函数。
- 此时将该任务移除就绪列表，并添加到阻塞列表中。
- 滴答中断里边进行计时。
- 判断阻塞时间是否到达，如果到达将从阻塞列表移除，添加到就绪列表。

## 13.FreeRTOS消息队列

### 一、队列简介

#### 1、FreeRTOS中的消息队列是什么

答：消息队列是任务到任务、任务到中断、中断到任务数据交流的一种机制(消息传递)。

#### 2、消息队列和全局变量的区别

答：消息队列作用有点类似于全局变量，但消息队列在RTOS中比全局变量更安全可靠。

假设有一个全局变量a=0，现在有两个任务都要写这个变量a。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97%E5%92%8C%E5%85%A8%E5%B1%80%E5%8F%98%E9%87%8F.png)

上图中任务1和任务2在RTOS中相互争取修改a的值，a的值容易受损错乱。

全局变量的弊端：数据无保护，导致数据不安全，当多个任务同时对该变量操作时，数据易受损。

#### 3、使用队列的情况

答：使用队列的情况如下：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97%E4%BD%BF%E7%94%A8%E6%83%85%E5%86%B5.png)

读写队列做好了保护，防止多任务或中断同时访问产生冲突。我们只需直接调用API函数即可，简单易用。

注意：FreeRTOS基于队列，实现了多种功能，其中包括队列集、互斥信号量、计数信号量、二值信号量、递归互斥信号量，因此很有必要深入了解FreeRTOS的队列。

#### 4、队列项目和队列长度

答：在队列中可以存储数量有限、大小固定的数据。队列中的每个数据就叫做 “队列项目” ，队列能够存储 “队列项目” 的最大数量称为队列的长度。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%A1%B9%E7%9B%AE%E5%92%8C%E9%98%9F%E5%88%97%E9%95%BF%E5%BA%A6.png)

在创建队列时，就要指定队列长度以及队列项目的大小！

#### 5、FreeRTOS队列特点

答：

1. **数据入队出队方式** ： 队列通常采用 “先进先出(FIFO)” 的数据存储缓冲机制，即先入队的数据会先从队列中被读取，FreeRTOS中也可以配置为 “后进先出(LIFO)” 方式。
2. **数据传递方式** ： FreeRTOS中队列采用实际值传递，即将数据拷贝到队列中进行传递，FreeRTOS采用拷贝数据传递，也可以传递指针，所以在传递较大的数据的时候采用指针传递。
3. **多任务访问** ： 队列不属于某个任务，任何任务和中断都可以向队列写入/读取消息。
4. **出队、入队阻塞** ： 当任务向一个队列发送/读取消息时，可以指定一个阻塞时间，假设此时当队列已满无法入队。

#### 6、消息队列阻塞时间设置

答：

- 若阻塞时间为0                                 ：直接返回不会等待。
- 若阻塞时间为0~port_MAX_DELAY ：等待设定阻塞时间，若在该时间内无法入队/出队，超时后直接返回不再等待。
- 若阻塞时间为port_MAX_DELAY     ：死等，一直等到可以入队/出队为止。

#### 7、入队/出队阻塞过程

答：

入队阻塞：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97%E5%85%A5%E9%98%9F%E9%98%BB%E5%A1%9E.png)

队列满了，此时写不进去数据：

1. 将该任务的状态列表项挂载在pxDelayedTaskList；
2. 将该任务的事件列表项挂载在xTasksWaitingToSend；

出队阻塞：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97%E5%87%BA%E9%98%9F%E9%98%BB%E5%A1%9E.png)

队列为空，此时读取不了数据：

1. 将该任务的状态列表挂载在pxDelayedTaskList；
2. 将该任务的事件列表项挂载在xTasksWaitingToReceive；

#### 8、当多个任务写入消息给一个 “满队列” 时，这些任务都会进入阻塞状态，也就是说有多个任务在等待同一个队列的空间。那当队列有空间时，哪个任务会进入就绪态？

答：

1. 优先级最高的任务
2. 如果大家的优先级相同，那等待时间最久的任务进入就绪态。

#### 9、队列创建、写入和读出过程

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97%E5%88%9B%E5%BB%BA%E3%80%81%E8%AF%BB%E5%86%99%E8%BF%87%E7%A8%8B.png)

### 二、队列结构体介绍

#### 1、队列结构体

答：

```C
typedef struct QueueDefinition 
{
    int8_t * pcHead;                       /* 存储区域的起始地址 */
    int8_t * pcWriteTo;                    /* 下一个写入的位置 */
    union
    {
        QueuePointers_t     xQueue;
        SemaphoreData_t  xSemaphore; 
    } u ;
    List_t xTasksWaitingToSend;             /* 等待发送列表 */
    List_t xTasksWaitingToReceive;          /* 等待接收列表 */
    volatile UBaseType_t uxMessagesWaiting; /* 非空闲队列项目的数量 */
    UBaseType_t uxLength；                  /* 队列长度 */
    UBaseType_t uxItemSize;                 /* 队列项目的大小 */
    volatile int8_t cRxLock;                /* 读取上锁计数器 */
    volatile int8_t cTxLock;                /* 写入上锁计数器 */
   /* 其他的一些条件编译 */
} xQUEUE;
```

当用于队列使用时：

```C
typedef struct QueuePointers
{
    int8_t * pcTail;                 /* 存储区的结束地址 */
    int8_t * pcReadFrom;             /* 最后一个读取队列的地址 */
} QueuePointers_t;
```

当用于互斥信号量和递归互斥信号量时：

```c
typedef struct SemaphoreData
{
    TaskHandle_t xMutexHolder;		    /* 互斥信号量持有者 */
    UBaseType_t uxRecursiveCallCount;	/* 递归互斥信号量的获取计数器 */
} SemaphoreData_t;
```

队列结构体示意图：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E7%BB%93%E6%9E%84%E4%BD%93%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

### 三、队列相关API函数介绍

#### 1、队列使用流程

答：使用队列的主要流程：创建队列 —> 写队列 —> 读队列。

#### 2、创建队列函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%88%9B%E5%BB%BA%E9%98%9F%E5%88%97%E5%87%BD%E6%95%B0.png)

动态和静态创建队列之间的区别：队列所需的内存空间由 FreeRTOS 从 FreeRTOS 管理的堆中分配，而静态创建需要用户自行分配内存。

```C
#define xQueueCreate (  uxQueueLength,   uxItemSize  )
        xQueueGenericCreate( ( uxQueueLength ), ( uxItemSize ), (queueQUEUE_TYPE_BASE )) 
```

此函数用于使用动态方式创建队列，队列所需的内存空间由 FreeRTOS 从 FreeRTOS 管理的堆中分配。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8A%A8%E6%80%81%E5%88%9B%E5%BB%BA%E9%98%9F%E5%88%97%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8A%A8%E6%80%81%E5%88%9B%E5%BB%BA%E9%98%9F%E5%88%97%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 3、各种功能所对应的队列

答：

```c
#define queueQUEUE_TYPE_BASE                    ( ( uint8_t ) 0U )  /* 队列 */
#define queueQUEUE_TYPE_SET                     ( ( uint8_t ) 0U )  /* 队列集 */
#define queueQUEUE_TYPE_MUTEX                   ( ( uint8_t ) 1U )  /* 互斥信号量 */
#define queueQUEUE_TYPE_COUNTING_SEMAPHORE      ( ( uint8_t ) 2U )  /* 计数型信号量 */
#define queueQUEUE_TYPE_BINARY_SEMAPHORE        ( ( uint8_t ) 3U )  /* 二值信号量 */
#define queueQUEUE_TYPE_RECURSIVE_MUTEX         ( ( uint8_t ) 4U )  /* 递归互斥信号量 */
```

#### 4、队列写入消息函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E5%86%99%E5%85%A5%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B0.png)

```c
#define  xQueueSend( xQueue, pvItemToQueue, xTicksToWait  )
         xQueueGenericSend( ( xQueue ), ( pvItemToQueue ), ( xTicksToWait ), queueSEND_TO_BACK )
```

```c
#define  xQueueSendToBack( xQueue, pvItemToQueue, xTicksToWait  )
         xQueueGenericSend( ( xQueue ), ( pvItemToQueue ), ( xTicksToWait ), queueSEND_TO_BACK )
```

```C
#define  xQueueSendToFront( xQueue, pvItemToQueue, xTicksToWait  )
         xQueueGenericSend( ( xQueue ), ( pvItemToQueue ), ( xTicksToWait ), queueSEND_TO_FRONT )
```

```C
#define  xQueueOverwrite(  xQueue,   pvItemToQueue  )
         xQueueGenericSend( ( xQueue ), ( pvItemToQueue ), 0, queueOVERWRITE )
```

可以看到这几个写入函数调用的是同一个函数xQueueGenericSend( )，只是指定了不同的写入位置！ 

队列一共有3种写入位置：

```c
#define queueSEND_TO_BACK                ( ( BaseType_t ) 0 )       /* 写入队列尾部 */
#define queueSEND_TO_FRONT              ( ( BaseType_t ) 1 )        /* 写入队列头部 */
#define queueOVERWRITE                  ( ( BaseType_t ) 2 )        /* 覆写队列*/
```

注意：覆写方式写入队列，只有在队列的队列长度为 1 时，才能够使用 。

往队列写入消息函数入口参数解析：

```C
BaseType_t      xQueueGenericSend(  QueueHandle_t       xQueue,
                                    const void * const  pvItemToQueue,
                                    TickType_t          xTicksToWait,
                                    const BaseType_t    xCopyPosition   );
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%BE%80%E9%98%9F%E5%88%97%E5%86%99%E5%85%A5%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%BE%80%E9%98%9F%E5%88%97%E5%86%99%E5%85%A5%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 5、队列读出消息函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E8%AF%BB%E5%87%BA%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B0.png)

```c
BaseType_t    xQueueReceive( QueueHandle_t  xQueue, 
                             void *   const pvBuffer,  
                             TickType_t     xTicksToWait )
```

此函数用于在任务中，从队列中读取消息，并且消息读取成功后，会将消息从队列中移除。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E8%AF%BB%E5%87%BA%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B01%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E8%AF%BB%E5%87%BA%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B01%E8%BF%94%E5%9B%9E%E5%80%BC.png)

```C
BaseType_t   xQueuePeek( QueueHandle_t   xQueue,
                         void * const   pvBuffer,
                         TickType_t   xTicksToWait )
```

此函数用于在任务中，从队列中读取消息， 但与函数 xQueueReceive()不同，此函数在成功读取消息后，并不会移除已读取的消息！ 

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E8%AF%BB%E5%87%BA%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B02%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E8%AF%BB%E5%87%BA%E6%B6%88%E6%81%AF%E5%87%BD%E6%95%B02%E8%BF%94%E5%9B%9E%E5%80%BC.png)

## 14.FreeRTOS信号量

### 一、信号量简介

#### 1、什么是信号量 

答：信号量是一种解决同步问题的机制，可以实现对共享资源的有序访问。

假设有一个人需要在停车场停车。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%BD%A6.png)

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%81%9C%E8%BD%A6%E5%9C%BA.png)

- 空车位：信号量资源数(计数值)
- 让出占用车位： 释放信号量(计数值++)
- 占用车位： 获取信号量(计数值--)

1. 首先判断停车场是否还有空车位(判断信号量是否有资源)。
2. 停车场正好有空车位(信号量有资源)，那么就可以直接将车开入停车位进行停车(获取信号量成功)。
3. 停车场已经没有空车位了(信号量没有资源)，那么可以选择不停车(获取信号量失败)，也可以选择等待(任务阻塞)其他人将车开出停车位(释放信号)，然后在将车停如空车位。

#### 2、信号量简介

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BF%A1%E5%8F%B7%E9%87%8F%E7%AE%80%E4%BB%8B.png)

- 当计数值大于0，表示有信号量资源。
- 当释放信号量，信号量计数值(资源数)加一。
- 当获取信号量，信号量计数值(资源数)减一。
- 信号量的计数值都是有限的：限定最大值。
- 如果最大值被限定为1，那么它就是二值信号量。
- 如果最大值不是1，它就是计数型信号量。

注意：信号量用于传递状态。

#### 3、队列与信号量的对比

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E4%B8%8E%E4%BF%A1%E5%8F%B7%E9%87%8F%E7%9A%84%E5%AF%B9%E6%AF%94.png)

### 二、二值信号量

#### 1、二值信号量介绍

答：二值信号量的本质是一个队列长度为1的队列，该队列就只有空和满两种情况。这就是二值信号量。

注意：二值信号量通常用于互斥访问或任务同步，与互斥信号量比较类似，但是二值信号量有可能会导致优先级翻转的问题，所以二值信号量更适合用于同步！！！

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E9%87%8F%E4%BB%8B%E7%BB%8D.png)

#### 2、二值信号量相关API函数

答：使用二值信号量的过程：创建二值信号量  ->  释放二值信号量  ->  获取二值信号量

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E9%87%8FAPI%E5%87%BD%E6%95%B0.png)

#### 3、创建二值信号量函数

答：创建二值信号量函数：

```c
SemaphoreHandle_t  xSemaphoreCreateBinary( void );
```

```c
#define   xSemaphoreCreateBinary()
xQueueGenericCreate(1 , semSEMAPHORE_QUEUE_ITEM_LENGTH, queueQUEUE_TYPE_BINARY_SEMAPHORE)
#define   semSEMAPHORE_QUEUE_ITEM_LENGTH   (( uint8_t ) 0U)
```

```c
#define   queueQUEUE_TYPE_BASE                           ( ( uint8_t ) 0U ) /* 队列 */
#define   queueQUEUE_TYPE_SET                            ( ( uint8_t ) 0U ) /* 队列集 */
#define   queueQUEUE_TYPE_MUTEX                          ( ( uint8_t ) 1U ) /* 互斥信号量 */
#define   queueQUEUE_TYPE_COUNTING_SEMAPHORE             ( ( uint8_t ) 2U ) /* 计数型信号量 */
#define   queueQUEUE_TYPE_BINARY_SEMAPHORE               ( ( uint8_t ) 3U ) /* 二值信号量 */
#define   queueQUEUE_TYPE_RECURSIVE_MUTEX                ( ( uint8_t ) 4U ) /* 递归互斥信号量 */
```

返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%88%9B%E5%BB%BA%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E9%87%8F%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 4、释放二值信号量函数

答：释放二值信号量函数：

```c
BaseType_t   xSemaphoreGive( xSemaphore ) 
```

```c
#define   xSemaphoreGive ( xSemaphore )
xQueueGenericSend((QueueHandle_t)( xSemaphore ), NULL, semGIVE_BLOCK_TIME, queueSEND_TO_BACK)
#define   semGIVE_BLOCK_TIME       ( ( TickType_t ) 0U )
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%87%8A%E6%94%BE%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%87%8A%E6%94%BE%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 5、获取二值信号量函数

答：获取二值信号量函数：

```c
BaseType_t   xSemaphoreTake( xSemaphore, xBlockTime ) 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E4%BA%8C%E5%80%BC%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

### 三、计数型信号量

#### 1、计数型信号量介绍

答：计数型信号量相当于队列长度为1的队列，因此计数型信号量能够容纳多个资源，这在计数型信号量被创建的时候确定的。

计数型信号量适用场合：

- 事件计数 ： 当每次事件发生后，在事件处理函数中释放计数型信号量(计数值+1)，其他任务会获取计数型信号量(计数值-1)，这种场合一般在创建时将初始化计数值设置为0.
- 资源管理 ： 信号量表示有效资源数量。任务必须先获取信号量(信号计数值-1)才能获取资源控制权。当计数值减为0时表示没有资源。当任务使用完资源后，必须释放信号量(信号量计数值+1)。信号量创建时计数值应等于最大资源数目。

#### 2、计数型信号量相关API函数

答：使用计数型信号量的过程：创建计数型信号量  ->  释放信号量  ->  获取信号量

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%AE%A1%E6%95%B0%E5%9E%8B%E4%BF%A1%E5%8F%B7%E9%87%8FAPI%E5%87%BD%E6%95%B0.png)

注意：计数型信号量的释放与获取的函数和二值信号量一样。

#### 3、计数型信号量创建函数

答：

```c
#define 	xSemaphoreCreateCounting( uxMaxCount , uxInitialCount )
            xQueueCreateCountingSemaphore( ( uxMaxCount ) , ( uxInitialCount ) ) 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%AE%A1%E6%95%B0%E5%9E%8B%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%88%9B%E5%BB%BA%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%AE%A1%E6%95%B0%E5%9E%8B%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%88%9B%E5%BB%BA%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 4、获取计数型信号量计数值函数

答：

```C
#define 	uxSemaphoreGetCount( xSemaphore ) 
            uxQueueMessagesWaiting(( QueueHandle_t )( xSemaphore ))
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E8%AE%A1%E6%95%B0%E5%80%BC%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%8E%B7%E5%8F%96%E8%AE%A1%E6%95%B0%E5%80%BC%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

### 四、优先级翻转介绍

#### 1、优先级翻转简介

答：优先级翻转：高优先级的任务反而慢执行，低优先级的任务反而优先执行。

优先级翻转在抢占式内核中是非常常见的，但是在实时操作系统中是不允许出现优先级翻转的，因为优先级翻转会破坏任务的预期顺序，可能会导致未知的严重后果。

在使用二值信号量的时候，经常会遇到优先级翻转的问题。

#### 2、优先级翻转的例子

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BC%98%E5%85%88%E7%BA%A7%E7%BF%BB%E8%BD%AC%E7%9A%84%E4%BE%8B%E5%AD%90.png)

高优先级任务被低优先级任务阻塞，导致高优先级任务迟迟得不到调度。但其他中等优先级的任务却能抢到CPU资源。从现象上看，就像是中等优先级的任务比高优先级任务具有更高的优先权(即优先级翻转)。

### 五、互斥信号量

#### 1、互斥信号量介绍

答：互斥信号量其实就是一个 拥有优先级翻转的二值信号量。

- 二值信号量更适用于同步的应用。
- 互斥信号量更适合那些需要互斥访问的应用(资源紧缺，需要资源保护)。

#### 2、什么是优先级继承

答：当一个互斥信号量正在被一个低优先级的任务持有时，如果此时有一个高优先级的任务也尝试获取这个互斥信号量，那么这个高优先级的任务就会被阻塞。不过这个高优先级的任务会将低优先级任务的优先级提升到与自己相同的优先级。

#### 3、优先级继承示例

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BC%98%E5%85%88%E7%BA%A7%E7%BB%A7%E6%89%BF%E7%A4%BA%E4%BE%8B.png)

此时任务H的阻塞时间仅仅是任务L的执行时间，将优先级翻转的危害降低到了最低。

#### 4、互斥信号量的注意事项

答：优先级继承并不能完全的消除优先级翻转的问题，它只是尽可能的降低优先级翻转带来的影响。

注意：互斥信号量不能用于中断服务函数中，原因如下：

1. 互斥信号量有优先级继承的机制，但是中断不是任务，没有任务优先级，所以互斥信号量只能用于任务中，不能用于中断服务函数中。
2. 中断服务函数中不能因为要等待互斥信号量而设置阻塞时间进入阻塞态。

#### 5、互斥信号量相关API函数

答：使用互斥信号量：首先将宏configUSE_MUTEXES置1.

使用流程：创建互斥信号量  ->  (task)获取信号量  ->  (give)释放信号量

创建互斥信号量函数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%88%9B%E5%BB%BA%E4%BA%92%E6%96%A5%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%87%BD%E6%95%B0.png)

互斥信号量的释放和获取函数与二值信号量相同！！！只不过互斥信号量不支持中断中调用。

注意：创建互斥信号量时，会主动释放一次信号量。

#### 6、创建互斥信号量函数

答：

```c
#define   xSemaphoreCreateMutex()      xQueueCreateMutex( queueQUEUE_TYPE_MUTEX )
```

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%88%9B%E5%BB%BA%E4%BA%92%E6%96%A5%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

## 15.FreeRTOS队列集

### 一、队列集简介

#### 1、队列集介绍

答：

- 一个队列只允许任务间传递的消息为同一种数据类型，如果需要再任务间传递不同数据类型的消息时，那么就可以使用队列集！！！
- 作用：用于对多个队列或信号量进行“监听”，其中不管哪一个消息到来，都可让任务退出阻塞状态。

假设：有一个任务，使用到队列接收和信号量的获取，如下：

不使用队列集：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%B8%8D%E4%BD%BF%E7%94%A8%E9%98%9F%E5%88%97%E9%9B%86.png)

使用队列集：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BD%BF%E7%94%A8%E9%98%9F%E5%88%97%E9%9B%86.png)

### 二、队列集相关API函数

#### 1、队列集相关API函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0.png)

#### 2、队列集创建函数

答：

```C
QueueSetHandle_t     xQueueCreateSet( const  UBaseType_t  uxEventQueueLength ); 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E5%88%9B%E5%BB%BA%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E5%88%9B%E5%BB%BA%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 3、队列集添加函数

答：

```C
BaseType_t xQueueAddToSet( QueueSetMemberHandle_t   xQueueOrSemaphore,
                           QueueSetHandle_t         xQueueSet); 
```

此函数用于往队列集中添加队列，要注意的时，队列在被添加到队列集之前队列中不能有有效的消息。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E6%B7%BB%E5%8A%A0%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E6%B7%BB%E5%8A%A0%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 4、队列集移除函数

答：

```C
BaseType_t   xQueueRemoveFromSet( QueueSetMemberHandle_t    xQueueOrSemaphore,
                                  QueueSetHandle_t          xQueueSet ); 
```

此函数用于从队列集中移除队列，要注意的是，队列在从队列集中移除之前，必须没有有效的消息。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E7%A7%BB%E9%99%A4%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E7%A7%BB%E9%99%A4%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 5、队列集获取函数

答：

```C
QueueSetMemberHandle_t   xQueueSelectFromSet( QueueSetHandle_t    xQueueSet,
                                              TickType_t const    xTicksToWait)
```

此函数用于在任务中获取队列集中有有效消息的队列。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E8%8E%B7%E5%8F%96%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%98%9F%E5%88%97%E9%9B%86%E8%8E%B7%E5%8F%96%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

## 16.FreeRTOS事件标志组

### 一、事件标志组简介

#### 1、事件标志组介绍

答：

事件标志位：用一个位，来表示事件是否发生。

事件标志组是一组事件标志位的合集，可以简单的理解事件标志组，就是一个整数。

#### 2、事件标志组的特点

答：

- 它的每一个位表示一个事件(高8位不算)。
- 每一位事件的含义，由用户自己决定，如：bit0表示按键是否按下，bit1表示是否接收到信息......
- 这些位的值为1表示事件发生了，值为0表示事件未发生。
- 任意任务或中断都可以读写这些位。
- 可以等待某一位成立，或者等待多位同时成立。

#### 3、事件标志组的标志变量

答：一个事件组就包含了一个EventBits_t数据类型的变量，变量类型EventBits_t数据类型的定义如下所示：

```C
typedef TickType_t EventBits_t;
#if ( configUSE_16_BIT_TICKS  ==  1 )
	typedef   uint16_t   TickType_t;
#else
	typedef   uint32_t   TickType_t;
#endif

#define  configUSE_16_BIT_TICKS    0 
```

EventBits_t 实际上是一个 16 位或 32 位无符号的数据类型。

注意：虽然使用了32为/16位无符号的数据类型变量来存储事件标志，但其中的高8为作用是存储事件标志组的控制信息，低24位/8位的作用才是存储事件标志，所以说一个事件标志组最多可以存储24个事件标志！！！

24位事件标志组示例图：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/24%E4%BD%8D%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E7%BB%84%E7%A4%BA%E4%BE%8B%E5%9B%BE.png)

#### 4、事件标志组与队列、信号量的区别

答：

<img src="https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E7%BB%84%E4%B8%8E%E9%98%9F%E5%88%97-%E4%BF%A1%E5%8F%B7%E9%87%8F%E5%8C%BA%E5%88%AB.png" style="zoom: 64%;" />

### 二、事件标志组相关API函数介绍

#### 1、事件标志组相关API函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E7%BB%84%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0.png)

#### 2、动态创建事件标志组函数

答：

```C
EventGroupHandle_t    xEventGroupCreate ( void ); 
```

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8A%A8%E6%80%81%E5%88%9B%E5%BB%BA%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E7%BB%84.png)

#### 3、消除事件标志位函数

答：

```C
EventBits_t  xEventGroupClearBits( EventGroupHandle_t   xEventGroup,
                                   const EventBits_t    uxBitsToClear) 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B8%85%E9%99%A4%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E4%BD%8D%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%B8%85%E9%99%A4%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E4%BD%8D%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 4、设置事件标志位函数

答：

```C
EventBits_t   xEventGroupSetBits(  EventGroupHandle_t   xEventGroup,
                                   const EventBits_t    uxBitsToSet ) 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%AE%BE%E7%BD%AE%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E4%BD%8D%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%AE%BE%E7%BD%AE%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E4%BD%8D%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 5、等待事件标志位函数

答：

```C
EventBits_t   xEventGroupWaitBits(   EventGroupHandle_t   xEventGroup,
                                     const EventBits_t    uxBitsToWaitFor,
                                     const BaseType_t     xClearOnExit,
                                     const BaseType_t     xWaitForAllBits,
                                     TickType_t           xTicksToWait  )
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E7%AD%89%E5%BE%85%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E4%BD%8D%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E7%AD%89%E5%BE%85%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E4%BD%8D%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

函数特点：

1. 可以等待某一为、也可以等待多位。
2. 等待期望的事件后，可以选择自动清除相关位或者不清除。

#### 6、同步事件标志组函数

答：

```C
EventBits_t     xEventGroupSync(   EventGroupHandle_t   xEventGroup,
                                   const EventBits_t    uxBitsToSet,
                                   const EventBits_t    uxBitsToWaitFor,
                                   TickType_t           xTicksToWait) 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%90%8C%E6%AD%A5%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E7%BB%84%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%90%8C%E6%AD%A5%E4%BA%8B%E4%BB%B6%E6%A0%87%E5%BF%97%E7%BB%84%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

## 17.FreeRTOS任务通知

### 一、任务通知的简介

#### 1、任务通知介绍

答：任务通知：用来通知任务的，任务控制块中的结构体成员变量ulNotifiedValue就是这个通知值。

使用队列、信号量、事件标志组时都需要另外创建一个结构体，通过中间的结构体进行间接通信。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E9%9D%9E%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E7%9A%84%E9%80%9A%E4%BF%A1.png)

使用任务通知时，任务结构体TCB中就包含了内部对象，可以直接接收别人发过来的“通知”。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E7%9A%84%E9%80%9A%E4%BF%A1.png)

#### 2、任务通知值的更新方式

答：

- 不覆盖接收任务的通知值。
- 覆盖接收任务的通知值。
- 更新接收任务通知值的一个或多个位。
- 增加接收任务的通知值。

只要合理，灵活的利用任务通知的特点，可以在一些场合中替代队列、信号量、事件标志组。

#### 3、任务通知值的优势及劣势

答：

任务通知的优势：

- 效率更高：使用任务通知向任务发送事件或数据比使用队列、事件标志组或信号量快得多。
- 使用内存更小：使用其他方法时都要先创建对应的结构体，使用任务通知时无需额外创建结构体。

任务通知的劣势：

- 无法发送数据给ISR：ISR没有任务结构体，所以无法给ISR发送数据。但是ISR可以使用任务通知的功能，发数据给任务。
- 无法广播给多个任务：任务通知只能是被指定的一个任务接收并处理 。
- 无法缓存多个数据：任务通知是通过更新任务通知值来发送数据的，任务结构体中只有一个任务通知值，只能保持一个数据。
- 发送受阻不支持阻塞：发送方无法进入阻塞状态等待。

### 二、任务通知值和通知状态

#### 1、任务通知结构体

答：

任务都有一个结构体---任务控制块，它里面有两个结构体成员变量：

```C
typedef  struct  tskTaskControlBlock 
{
      ......
      #if ( configUSE_TASK_NOTIFICATIONS  ==  1 )
           volatile  uint32_t    ulNotifiedValue [ configTASK_NOTIFICATION_ARRAY_ENTRIES ];
           volatile  uint8_t      ucNotifyState [ configTASK_NOTIFICATION_ARRAY_ENTRIES ];
      endif
      ......
} tskTCB;

#define  configTASK_NOTIFICATION_ARRAY_ENTRIES	1  	/* 定义任务通知数组的大小, 默认: 1 */
```

- 一个是uint32_t类型，用来表示任务通知值。
- 一个是uint16_t类型，用来表示任务通知状态。

#### 2、任务通知值

答：

任务通知值的更新方式有多种类型：

1. 计数值(数值累加，类似信号量)。
2. 相应位置1(类似事件标志组)。
3. 任意数值(支持覆写或不覆写，类似队列)。

#### 3、任务通知状态

答：任务通知状态共有3种。

```C
#define     taskNOT_WAITING_NOTIFICATION    ( ( uint8_t ) 0 )        /* 任务未等待通知 */
#define     taskWAITING_NOTIFICATION        ( ( uint8_t ) 1 )        /* 任务在等待通知 */
#define     taskNOTIFICATION_RECEIVED       ( ( uint8_t ) 2 )        /* 任务在等待接收 */
```

- 任务未等待通知：任务通知默认的初始化状态。
- 等待通知：接收方已经准备好了(调用了接收任务通知函数)，等待发送方给个通知。
- 等待接收：发送方已经发送过去(调用了发送任务通知函数)，等待接收方接收。

### 三、任务通知相关API函数

#### 1、任务通知相关API函数介绍

答：任务通知API函数主要有两类：1-发送通知，2-接收通知。

注意：发送通知API函数可以用于任务和中断函数中，但接受通知API函数只能用在任务中。

发送通知相关API函数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8F%91%E9%80%81%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%87%BD%E6%95%B0.png)

接收通知相关API函数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%8E%A5%E6%94%B6%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%87%BD%E6%95%B0.png)

#### 2、发送任务通知函数

答：

所有发送任务通知函数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8F%91%E9%80%81%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%87%BD%E6%95%B0.png)

```c
#define     xTaskNotifyAndQuery( xTaskToNotify,  ulValue,  eAction,  pulPreviousNotifyValue )
            xTaskGenericNotify( ( xTaskToNotify ), 
					          ( tskDEFAULT_INDEX_TO_NOTIFY ), 
					          ( ulValue ), 
					          ( eAction ),
					          ( pulPreviousNotifyValue ) )

#define     xTaskNotify  (xTaskToNotify ,  ulValue ,  eAction)
            xTaskGenericNotify(  ( xTaskToNotify ) ,  
                                 ( tskDEFAULT_INDEX_TO_NOTIFY ) ,  
                                 ( ulValue ) ,
                                 ( eAction ) , 
                                   NULL    )
 
#define     xTaskNotifyGive(  xTaskToNotify  )
            xTaskGenericNotify(  ( xTaskToNotify ) ,
                                 ( tskDEFAULT_INDEX_TO_NOTIFY ) ,
                                 ( 0 ) ,
                                 eIncrement ,
                                 NULL )
```

关键函数：

```c
BaseType_t     xTaskGenericNotify(  TaskHandle_t     xTaskToNotify,
                                    UBaseType_t      uxIndexToNotify,
                                    uint32_t         ulValue,
                                    eNotifyAction    eAction,
                                    uint32_t *       pulPreviousNotificationValue  )
```

发送任务通知的关键函数的参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8F%91%E9%80%81%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%85%B3%E9%94%AE%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

任务通知方式枚举：

```c
typedef  enum
{    
     eNoAction = 0,              /* 无操作 */
     eSetBits                    /* 更新指定bit */
     eIncrement                  /* 通知值加一 */
     eSetValueWithOverwrite      /* 覆写的方式更新通知值 */
     eSetValueWithoutOverwrite   /* 不覆写通知值 */
} eNotifyAction;
```

#### 3、接收任务通知函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%8E%A5%E6%94%B6%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%87%BD%E6%95%B0.png)

注意：

- 当任务通知用于信号量时，使用函数 ulTaskNotifyTake() 获取获取信号量。
- 当任务通知用于事件标志组或队列时，使用函数 xTaskNotifyWait() 来获取。

ulTaskNotifyTake()函数：

```c
#define     ulTaskNotifyTake( xClearCountOnExit  ,   xTicksToWait )
            ulTaskGenericNotifyTake( ( tskDEFAULT_INDEX_TO_NOTIFY ),
                                     ( xClearCountOnExit ),
                                     ( xTicksToWait ) )
```

此函数用于接收任务通知值，可以设置在退出此函数的时候将任务通知值清零或者减一。

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%8E%A5%E6%94%B6%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%80%BC%E5%87%BD%E6%95%B01%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%8E%A5%E6%94%B6%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%80%BC%E5%87%BD%E6%95%B01%E8%BF%94%E5%9B%9E%E5%80%BC.png)

xTaskNotifyWait()函数：

```C
#define     xTaskNotifyWait(    ulBitsToClearOnEntry,
                                ulBitsToClearOnExit,
                                pulNotificationValue,
                                xTicksToWait) 
            xTaskGenericNotifyWait(   tskDEFAULT_INDEX_TO_NOTIFY,
                                      ( ulBitsToClearOnEntry ),
                                      ( ulBitsToClearOnExit ),
                                      ( pulNotificationValue ),
                                      ( xTicksToWait )         ) 
```

```c
BaseType_t     xTaskGenericNotifyWait(     UBaseType_t     uxIndexToWaitOn,
                                           uint32_t        ulBitsToClearOnEntry,
                                           uint32_t        ulBitsToClearOnExit,
                                           uint32_t *      pulNotificationValue,
                                           TickType_t      xTicksToWait	    ); 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%8E%A5%E6%94%B6%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%80%BC%E5%87%BD%E6%95%B02%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%8E%A5%E6%94%B6%E4%BB%BB%E5%8A%A1%E9%80%9A%E7%9F%A5%E5%80%BC%E5%87%BD%E6%95%B02%E8%BF%94%E5%9B%9E%E5%80%BC.png)

## 18.FreeRTOS软件定时器

### 一、软件定时器的简介

#### 1、定时器介绍

答：

定时器：从指定的时刻开始，经过一个指定时间，然后触发一个超时事件，用户可以自定义定时器周期。

硬件定时器：芯片本身自带的定时器模块，硬件定时器的精度一般很高，每次在定时时间到达之后就会自动触发一个中断，用户在中断服务函数中处理信息。

软件定时器：是指具有定时功能的软件，可设置定时周期，当指定时间到达后要调用回调函数(也称超时函数)，用户在回调函数中处理信息。

#### 2、软件定时器优缺点

答：

优点：

- 硬件定时器数量有限，而软件定时器理论上只需有足够内存，就可以创建多个；
- 使用简单、成本低。

缺点：

- 软件定时器相对硬件定时器来说，精度没有那么高(因为它以系统时钟为基准，系统时钟中断优先级又是最低，容易被打断)。对于需要高精度要求的场合，不建议使用软件定时器。

#### 3、FreeRTOS软件定时器特点

答：

1. 可裁剪：软件定时器是可裁剪可配置的功能，如果要使能软件定时器，需将configUSE_TIMERS 配置项配置成 1 。
2. 单次和周期：软件定时器支持设置成  **单次定时器**  或  **周期定时器**  。

注意：软件定时器的超时回调函数是由软件定时器服务任务调用的，软件定时器的超时回调函数本身不是任务，因此不能在该回调函数中使用可能会导致任务阻塞的 API 函数。

软件定时器服务任务：在调用函数 vTaskStartScheduler()开启任务调度器的时候，会创建一个用于管理软件定时器的任务，这个任务就叫做**软件定时器服务任务**。

#### 4、软件定时器服务任务作用

答：

1. 负责软件定时器超时的逻辑判断。
2. 调用超时软件定时器的超时回调函数。
3. 处理软件定时器命令队列。

#### 5、软件定时器的命令队列

答：FreeRTOS 提供了许多软件定时器相关的 API 函数，这些 API 函数大多都是往定时器的队列中写入消息（发送命令），这个队列叫做软件定时器命令队列，是提供给 FreeRTOS 中的软件定时器使用的，用户是不能直接访问的。 

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E7%9A%84%E5%91%BD%E4%BB%A4%E9%98%9F%E5%88%97.png)

#### 6、软件定时器的相关配置

答：

- 当FreeRTOS 的配置项 configUSE_TIMERS 设置为1，在启动任务调度器时，会自动创建软件定时器的服务/守护任务prvTimerTask( )。
- 软件定时器服务任务的优先级为 configTIMER_TASK_PRIORITY = 31。
- 定时器的命令队列长度为 configTIMER_QUEUE_LENGTH = 5。

注意：软件定时器的超时回调函数是在软件定时器服务任务中被调用的，服务任务不是专为某个定时器服务的，它还要处理其他定时器。

因此定时器的回调函数不要影响其他“人”：

1. 回调函数要尽快实行，不能进入阻塞状态，即不能调用那些会阻塞任务的 API 函数，如：vTaskDelay() 。
2. 访问队列或者信号量的非零阻塞时间的 API 函数也不能调用。

### 二、软件定时器的状态

#### 1、软件定时器的状态

答：

- 休眠态：软件定时器可以通过其句柄被引用，但因为没有运行，所以其定时超时回调函数不会被执行。
- 运行态：运行态的定时器，当指定时间到达之后，它的超时回调函数会被调用。

注意：新创建的软件定时器处于休眠状态 ，也就是未运行的！ 

#### 2、如何让软件定时器从休眠态转变为运行态？

答：发送命令队列。

### 三、单次定时器和周期定时器

#### 1、单次定时器和周期定时器介绍

答：FreeRTOS提供了两种软件定时器：

- 单次定时器：单次定时器的一旦定时超时，只会执行一次其软件定时器超时回调函数，不会自动重新开启定时，不过可以被手动重新开启。
- 周期定时器：周期定时器的一旦启动以后就会在执行完回调函数以后自动的重新启动 ，从而周期地执行其软件定时器回调函数。

#### 2、单次定时器和周期定时器的对比示例

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8D%95%E6%AC%A1%E5%AE%9A%E6%97%B6%E5%99%A8%E5%92%8C%E5%91%A8%E6%9C%9F%E5%AE%9A%E6%97%B6%E5%99%A8%E7%9A%84%E5%AF%B9%E6%AF%94%E7%A4%BA%E4%BE%8B.png)

Timer1：周期定时器，定时超时时间为 2 个单位时间，开启后，一直以2个时间单位间隔重复执行。

Timer2：单次定时器，定时超时时间为 1 个单位时间，开启后，则在第一个超时后就不在执行了。

#### 3、软件定时器的状态转换图

答：

单次定时器状态转换图：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%8D%95%E6%AC%A1%E5%AE%9A%E6%97%B6%E5%99%A8%E7%8A%B6%E6%80%81%E8%BD%AC%E6%8D%A2%E5%9B%BE.png)

周期定时器状态转换图：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%91%A8%E6%9C%9F%E5%AE%9A%E6%97%B6%E5%99%A8%E7%8A%B6%E6%80%81%E8%BD%AC%E6%8D%A2%E5%9B%BE.png)

### 四、软件定时器结构体成员介绍

#### 1、软件定时器结构体介绍

答：

```c
typedef   struct
{
       const char *                  pcTimerName              /* 软件定时器名字 */
       ListItem_t                    xTimerListItem           /* 软件定时器列表项 */
       TickType_t                    xTimerPeriodInTicks;     /* 软件定时器的周期 */
       void *                        pvTimerID                /* 软件定时器的ID */
       TimerCallbackFunction_t       pxCallbackFunction;      /* 软件定时器的回调函数 */
       #if ( configUSE_TRACE_FACILITY == 1 )
              UBaseType_t 			uxTimerNumber            /*  软件定时器的编号，调试用  */
       #endif
       uint8_t                        ucStatus;               /*  软件定时器的状态  */
} xTIMER;
```

### 五、FreeRTOS软件定时器相关API函数

#### 1、软件定时器相关函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0.png)

#### 2、创建软件定时器函数

答：

```c
TimerHandle_t   xTimerCreate(     const char * const        pcTimerName,
                                  const TickType_t          xTimerPeriodInTicks,
                                  const UBaseType_t         uxAutoReload,
                                  void * const              pvTimerID,
                                  TimerCallbackFunction_t   pxCallbackFunction  ); 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%88%9B%E5%BB%BA%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%88%9B%E5%BB%BA%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 3、开启软件定时器函数

答：

```C
BaseType_t     xTimerStart(   TimerHandle_t        xTimer,
                              const TickType_t     xTicksToWait  ); 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%BC%80%E5%90%AF%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%BC%80%E5%90%AF%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 4、停止软件定时器函数

答：

```C
BaseType_t     xTimerStop(   TimerHandle_t      xTimer,
                             const TickType_t 	xTicksToWait   ); 
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%81%9C%E6%AD%A2%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%81%9C%E6%AD%A2%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 5、复位软件定时器函数

答：

```c
BaseType_t   xTimerReset( TimerHandle_t        xTimer,
                          const TickType_t     xTicksToWait); 
```

该函数将使软件定时器重新启动，复位后的软件定时器以复位是的时刻作为开启时刻重新定时。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%A4%8D%E4%BD%8D%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E6%95%88%E6%9E%9C.png)

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%A4%8D%E4%BD%8D%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%A4%8D%E4%BD%8D%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

#### 6、更改软件定时器超时时间函数

答：

```c
BaseType_t   xTimerChangePeriod( TimerHandle_t      xTimer,
                                 const TickType_t   xNewPeriod,
                                 const TickType_t   xTicksToWait );
```

函数参数：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%9B%B4%E6%94%B9%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E8%B6%85%E6%97%B6%E6%97%B6%E9%97%B4%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

函数返回值：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E6%9B%B4%E6%94%B9%E8%BD%AF%E4%BB%B6%E5%AE%9A%E6%97%B6%E5%99%A8%E8%B6%85%E6%97%B6%E6%97%B6%E9%97%B4%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

## 19.Tickless低功耗模式

### 一、低功耗模式简介

#### 1、低功耗介绍

答：很多应用场合对于功耗的要求很严格，比如可穿戴低功耗产品、物联网低功耗产品等；一般MCU都有相应的低功耗模式，裸机开发时可以使用MCU的低功耗模式。FreeRTOS也提供了一个叫Tickless的低功耗模式，方便带FreeRTOS操作系统的应用开发。

### 二、Tickless模式详解

#### 1、STM32低功耗模式

答：STM32低功耗模式有3种，分别是：睡眠模式、停止模式、待机模式。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/STM32%E4%BD%8E%E5%8A%9F%E8%80%97%E6%A8%A1%E5%BC%8F.png)

在FreeRTOS中主要使用的是睡眠模式：

- 进入睡眠模式：  WFI 指令(\__WFI )、WFE 指令(__WFE) 。
- 退出睡眠模式：任何中断或事件都可以唤醒睡眠模式。

#### 2、Tickless模式如何降低功耗

答：Tickless低功耗模式的本质是通过调用指令 WFI 实现睡眠模式！

#### 3、为什么要有Tickless模式

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/Tickless%E6%A8%A1%E5%BC%8F%E7%9A%84%E8%AE%BE%E8%AE%A1%E6%80%9D%E6%83%B3.png)

任务运行时间统计实验中，可以看出，在整个系统的运行过程中，其实大部分时间是在执行空闲任务。

空闲任务：是在系统中的所有其他任务都阻塞或挂起时才运行的。

#### 4、为了可以降低功耗，又不影响系统运行，该如何做？

答：可以在本该空闲任务执行的期间，让MCU 进入相应的低功耗模式；当其他任务准备运行的时候，唤醒MCU退出低功耗模式。

难点：

1. 进入低功耗之后，多久唤醒？也就是下一个要运行的任务如何被准确唤醒。
2. 任何中断均可唤醒MCU，若滴答定时器频繁中断则会影响低功耗的效果？

解决：将滴答定时器的中断周期修改为低功耗运行时间，退出低功耗后，需补上系统时钟节拍数。

值得庆幸的是：FreeRTOS 的低功耗 Tickless 模式机制已经处理好了这些难点。

### 三、Tickless模式相关配置项

#### 1、Tickless模式配置

答：

- configUSE_TICKLESS_IDLE        
  此宏用于使能低功耗 Tickless 模式 。
- configEXPECTED_IDLE_TIME_BEFORE_SLEEP     
  此宏用于定义系统进入相应低功耗模式的最短时长。
- configPRE_SLEEP_PROCESSING(x)
  此宏用于定义需要在系统进入低功耗模式前执行的事务，如：进入低功耗前关闭外设时钟，以达到降低功耗的目的。
- configPOSR_SLEEP_PROCESSING(x)
  此宏用于定义需要在系统退出低功耗模式后执行的事务，如：退出低功耗后开启之前关闭的外设时钟，以使系统能够正常运行。

#### 2、需要系统运行低功耗模式需满足以下几个条件

答：

1. 在 FreeRTOSConfig.h 文件中配置宏定义 configUSE_TICKLESS_IDLE 为 1 。
2. 满足当前空闲任务正在运行，所有其他任务处在挂起状态或阻塞状态。
3. 当系统可运行于低功耗模式的时钟节拍数大于等于configEXPECTED_IDLE_TIME_BEFORE_SLEEP（该宏默认为2个系统时钟节拍)。

#### 3、若想系统进入低功耗时功耗达到最低 

答：

1. 在进入睡眠模式前，可以关闭外设时钟、降低系统主频等，进一步降低系统功耗(调用函数configPRE_SLEEP_RPOCESSING()，需自行实现该函数的内部操作)。
2. 退出睡眠模式后，开启前面所关闭的外设时钟、恢复系统时钟主频等(退出睡眠模式后，开启前面所关闭的外设时钟、恢复系统时钟主频等)。

## 20.FreeRTOS内存管理

### 一、FreeRTOS内存管理简介

#### 1、FreeRTOS内存管理介绍

答：在使用 FreeRTOS 创建任务、队列、信号量等对象的时，一般都提供了两种方法：

- 动态方法创建：自动地从 FreeRTOS 管理的内存堆中申请创建对象所需的内存，并且在对象删除后，可将这块内存释放回FreeRTOS管理的内存堆 
- 静态方法创建：需用户提供各种内存空间，并且使用静态方式占用的内存空间一般固定下来了，即使任务、队列等被删除后，这些被占用的内存空间一般没有其他用途。

总结：**动态方式管理内存**相比与静态方式，更加灵活。

#### 2、为什么不用标准的C库自带的内存管理算法

答：因为标准 C 库的动态内存管理方法有如下几个缺点：

- 占用大量的代码空间 不适合用在资源紧缺的嵌入式系统中。
- 没有线程安全的相关机制。
- 运行有不确定性，每次调用这些函数时花费的时间可能都不相同。
- 内存碎片化。

因此，FreeRTOS 提供了多种动态内存管理的算法，可针对不同的嵌入式系统！

### 二、FreeRTOS内存管理算法

#### 1、FreeRTOS内存管理算法种类

答：FreeRTOS提供了5种动态内存管理算法，分别为： heap_1、heap_2、heap_3、heap_4、heap_5 。

如图所示：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%E7%AE%97%E6%B3%95%E7%A7%8D%E7%B1%BB.png)

在我们FreeRTOS例程中，使用的均为heap_4内存管理算法。

#### 2、heap_1内存管理算法

答：

##### heap_1的特点：

heap_1只实现了pvPortMalloc，没有实现vPortFree；也就是说，它只能申请内存，无法释放内存！如果你的工程，创建好的任务、队列、信号量等都不需要被删除，那么可以使用heap_1内存管理算法。

heap_1的实现最为简单，管理的内存堆是一个数组，在申请内存的时候， heap_1 内存管理算法只是简单地从数组中分出合适大小的内存，内存堆数组的定义如下所示 ：

```c
/* 定义一个大数组作为 FreeRTOS 管理的内存堆 */
static uint8_t ucHeap[ configTOTAL_HEAP_SIZE ];
```

##### heap_1内存管理算法的分配过程如下图所示：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/heap_1%E7%AE%97%E6%B3%95%E8%BF%87%E7%A8%8B.png)

注意： heap_1内存管理算法，只能申请无法释放！

#### 3、heap_2内存管理算法

答：

##### heap_2的特点：

- 相比于 heap_1 内存管理算法， heap_2 内存管理算法使用最适应算法，并且支持释放内存；
- heap_2 内存管理算法并不能将相邻的空闲内存块合并成一个大的空闲内存块；因此 heap_2 内存管理算法不可避免地会产生内存碎片；

##### 最适应算法：

假设heap有3块空闲内存（按内存块大小由小到大排序）：5字节、25字节、50字节。

现在新创建一个任务需要申请20字节的内存。

第一步：找出最小的、能满足pvPortMalloc的内存：25字节。

第二步：把它划分为20字节、5字节；返回这20字节的地址，剩下的5字节仍然是空闲状态，留给后续的pvPortMalloc使用。

##### heap_2内存管理算法的分配过程：

内存碎片是由于多次申请和释放内存，但释放的内存无法与相邻的空闲内存合并而产生的。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/heap_2%E7%AE%97%E6%B3%95%E8%BF%87%E7%A8%8B.png)

##### 适用场景：

频繁的创建和删除任务，且所创建的任务堆栈都相同，这类场景下Heap_2没有碎片化的问题。

#### 4、heap_4内存管理算法

答：

##### heap_4的特点：

heap_4 内存管理算法使用了首次适应算法，也支持内存的申请与释放，并且能够将空闲且相邻的内存进行合并，从而减少内存碎片的现象。

##### 首次适应算法：

- 假设heap有3块空闲内存（按内存块地址由低到高排序）：5字节、50字节、25字节。
- 现在新创建一个任务需要申请20字节的内存。
- 第一步：找出第一个能满足pvPortMalloc的内存：50字节。
- 第二步：把它划分为20字节、30字节；返回这20字节的地址，剩下30字节仍然是空闲状态，留给后续的pvPortMalloc使用。

##### heap_4内存管理算法的分配过程：

heap_4内存管理算法会把相邻的空闲内存合并为一个更大的空闲内存，这有助于减少内存的碎片问题。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/heap_4%E7%AE%97%E6%B3%95%E8%BF%87%E7%A8%8B.png)

##### 适用于这种场景：

频繁地分配、释放不同大小的内存。

#### 5、heap_5内存管理算法

答：

##### heap_5的特点：

heap_5 内存管理算法是在 heap_4 内存管理算法的基础上实现的，但是 heap_5 内存管理算法在 heap_4 内存管理算法的基础上实现了管理多个非连续内存区域的能力。

heap_5 内存管理算法默认并没有定义内存堆 ， 需要用户手动指定内存区域的信息，对其进行初始化。

##### 怎么指定一块内存？

使用如下结构体：

```c
typedef struct HeapRegion
{   
     uint8_t *     pucStartAddress;          /* 内存区域的起始地址 */
     size_t        xSizeInBytes;             /* 内存区域的大小，单位：字节 */
} HeapRegion_t; 
```

##### 怎么指定多块且不连续的内存？

```c
Const  HeapRegion_t  xHeapRegions[] = 
{
    {(uint8_t *)0x80000000, 0x10000 }, 	    /* 内存区域 1 */
    { (uint8_t *)0x90000000, 0xA0000 }, 	/* 内存区域 2 */
    { NULL, 0 }                             /* 数组终止标志 */
};
vPortDefineHeapRegions(xHeapRegions); 
```

##### 适用场景：

在嵌入式系统中，那些内存的地址并不连续的场景。

### 三、FreeRTOS内存管理相关API函数

#### 1、FreeRTOS内存管理相关函数

答：

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0.png)

```C
void * pvPortMalloc( size_t xWantedSize );
```

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/pvPortMalloc%E5%8F%82%E6%95%B0%E4%B8%8E%E8%BF%94%E5%9B%9E%E5%80%BC.png)

```C
void vPortFree( void * pv );
```

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/vPortFree%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0.png)

```C
size_t xPortGetFreeHeapSize( void );
```

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/xPortGetFreeHeapSize%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.png)

